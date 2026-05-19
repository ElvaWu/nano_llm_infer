#include <cerrno>
#include <chrono>
#include <cstring>
#include <memory>
#include <string>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include "generation.h"
#include "generation_server.h"

#include "test_util.h"

using namespace nano_llm_infer;

namespace {

#ifndef NANO_LLM_TEST_MODEL_PATH
#define NANO_LLM_TEST_MODEL_PATH "/root/autodl-tmp/repos/nano_llm_infer/model_file/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
#endif

class ScopedFd {
public:
    explicit ScopedFd(int fd) : fd_(fd) {}
    ~ScopedFd() {
        if (fd_ >= 0) {
            close(fd_);
        }
    }

    ScopedFd(const ScopedFd&) = delete;
    ScopedFd& operator=(const ScopedFd&) = delete;

    int get() const { return fd_; }

private:
    int fd_ = -1;
};

std::unique_ptr<ModelRuntime> make_runtime_or_skip() {
    const std::string model_path = NANO_LLM_TEST_MODEL_PATH;
    if (model_path.empty()) {
        std::cerr << "[SKIP] unit_generation_server: NANO_LLM_TEST_MODEL_PATH is empty\n";
        return nullptr;
    }

    try {
        RuntimeConfig config;
        config.model.path = model_path;
        config.model.compute_threads = 1;
        config.limits.context_tokens = 256;
        config.limits.default_max_new_tokens = 8;
        config.memory.prefill_ctx_bytes = 256ull * 1024 * 1024;
        config.memory.decode_ctx_bytes = 128ull * 1024 * 1024;
        return std::make_unique<ModelRuntime>(config);
    } catch (const std::exception& e) {
        std::cerr << "[SKIP] unit_generation_server: failed to load model at '" << model_path
                  << "' (" << e.what() << ")\n";
        return nullptr;
    }
}

int bind_local_ephemeral_port() {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        throw std::runtime_error(std::string("socket() failed: ") + std::strerror(errno));
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    if (bind(fd, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) != 0) {
        const std::string error = std::strerror(errno);
        close(fd);
        throw std::runtime_error("bind() failed: " + error);
    }
    if (listen(fd, 1) != 0) {
        const std::string error = std::strerror(errno);
        close(fd);
        throw std::runtime_error("listen() failed: " + error);
    }
    return fd;
}

int local_port_for_fd(int fd) {
    sockaddr_in addr{};
    socklen_t len = sizeof(addr);
    if (getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len) != 0) {
        throw std::runtime_error(std::string("getsockname() failed: ") + std::strerror(errno));
    }
    return static_cast<int>(ntohs(addr.sin_port));
}

void test_run_throws_when_port_is_already_in_use() {
    std::unique_ptr<ModelRuntime> runtime = make_runtime_or_skip();
    if (!runtime) {
        return;
    }

    ScopedFd occupied_fd(bind_local_ephemeral_port());

    ServerConfig config;
    config.host = "127.0.0.1";
    config.port = local_port_for_fd(occupied_fd.get());
    config.max_active_requests = 1;
    config.max_decode_batch_size = 1;
    config.enable_perf_logging = false;

    HttpStreamingServer server(runtime.get(), config);

    bool threw = false;
    std::string error;
    try {
        server.run();
    } catch (const std::exception& e) {
        threw = true;
        error = e.what();
    }

    EXPECT_TRUE(threw);
    EXPECT_TRUE(error.find("failed to listen on 127.0.0.1:") != std::string::npos);
}

GenerationRequest make_generation_request(const std::string& prompt, int max_tokens) {
    GenerationRequest request;
    request.messages.push_back({"user", prompt});
    request.options.max_tokens = max_tokens;
    return request;
}

void test_generation_stream_wait_next_timed(ModelRuntime* runtime) {
    GenerationService service(runtime, 1, 1);
    service.start();

    auto stream = service.submit(make_generation_request("Hello", 1));
    const auto submit_time = stream->submit_time();
    bool saw_event = false;
    bool saw_terminal = false;
    while (true) {
        GenerationEvent event;
        std::chrono::steady_clock::time_point event_time;
        if (!stream->wait_next_timed(&event, &event_time)) {
            break;
        }
        saw_event = true;
        EXPECT_TRUE(event_time >= submit_time);
        if (!event.finish_reason.empty() || !event.error.empty()) {
            saw_terminal = true;
        }
    }
    const auto close_time = stream->close_time();
    service.stop();

    EXPECT_TRUE(saw_event);
    EXPECT_TRUE(saw_terminal);
    EXPECT_TRUE(close_time >= submit_time);
}

void test_generation_stream_wait_next_still_returns_events(ModelRuntime* runtime) {
    GenerationService service(runtime, 1, 1);
    service.start();

    auto stream = service.submit(make_generation_request("Hi", 1));
    bool saw_event = false;
    bool saw_terminal = false;
    while (true) {
        GenerationEvent event;
        if (!stream->wait_next(&event)) {
            break;
        }
        saw_event = true;
        if (!event.finish_reason.empty() || !event.error.empty()) {
            saw_terminal = true;
        }
    }
    const auto close_time = stream->close_time();
    service.stop();

    EXPECT_TRUE(saw_event);
    EXPECT_TRUE(saw_terminal);
    EXPECT_TRUE(close_time != std::chrono::steady_clock::time_point{});
}

}  // namespace

int main() {
    const std::string body =
        "{\"model\":\"tiny\",\"stream\":true,\"max_tokens\":3,"
        "\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}";
    ParseResult parsed = parse_chat_completion_request(body);
    EXPECT_TRUE(parsed.ok);
    EXPECT_EQ(parsed.request.model, std::string("tiny"));
    EXPECT_TRUE(parsed.request.stream);
    EXPECT_EQ(parsed.request.generation.options.max_tokens, 3);
    EXPECT_EQ(parsed.request.generation.messages.size(), static_cast<size_t>(1));
    EXPECT_EQ(parsed.request.generation.messages[0].role, std::string("user"));
    EXPECT_EQ(parsed.request.generation.messages[0].content, std::string("Hello"));

    ParseResult no_stream = parse_chat_completion_request(
        "{\"model\":\"tiny\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}");
    EXPECT_TRUE(no_stream.ok);
    EXPECT_TRUE(!no_stream.request.stream);

    ParseResult stream_false = parse_chat_completion_request(
        "{\"model\":\"tiny\",\"stream\":false,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}");
    EXPECT_TRUE(stream_false.ok);
    EXPECT_TRUE(!stream_false.request.stream);

    ParseResult stream_non_bool = parse_chat_completion_request(
        "{\"model\":\"tiny\",\"stream\":\"yes\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}");
    EXPECT_TRUE(!stream_non_bool.ok);

    EXPECT_EQ(json_escape("a\"b\\c\n"), std::string("a\\\"b\\\\c\\n"));

    GenerationEvent token;
    token.text = "Hi";
    const std::string chunk = make_sse_chunk("tiny", token);
    EXPECT_TRUE(chunk.find("data: ") == 0);
    EXPECT_TRUE(chunk.find("\"content\":\"Hi\"") != std::string::npos);
    EXPECT_TRUE(chunk.find("\"finish_reason\":null") != std::string::npos);

    GenerationEvent done;
    done.finish_reason = "stop";
    EXPECT_TRUE(make_sse_chunk("tiny", done).find("\"finish_reason\":\"stop\"") != std::string::npos);
    EXPECT_EQ(make_sse_done(), std::string("data: [DONE]\n\n"));

    test_run_throws_when_port_is_already_in_use();
    std::unique_ptr<ModelRuntime> runtime = make_runtime_or_skip();
    if (runtime) {
        test_generation_stream_wait_next_timed(runtime.get());
        test_generation_stream_wait_next_still_returns_events(runtime.get());
    }

    return test_util::finalize();
}
