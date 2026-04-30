#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "generation.h"

namespace httplib {
class Server;
}

namespace nano_llm_infer {

struct ChatCompletionRequest {
    std::string model;
    bool stream = false;  // OpenAI-compatible default: return a single non-stream JSON completion.
    GenerationRequest generation;
};

struct ParseResult {
    bool ok = false;
    std::string error;
    ChatCompletionRequest request;
};

struct ServerConfig {
    std::string host = "127.0.0.1";
    int port = 8080;
    int max_active_requests = 2;
    int listen_backlog = 16;
};

ParseResult parse_chat_completion_request(const std::string& body);
std::string json_escape(const std::string& text);
std::string make_sse_chunk(const std::string& model_id, const GenerationEvent& event);
std::string make_sse_done();

class GenerationStream {
public:
    bool wait_next(GenerationEvent* event);
    void cancel();
    bool cancelled() const;

private:
    friend class GenerationService;

    void push(GenerationEvent event);
    void close();

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<GenerationEvent> events_;
    bool closed_ = false;
    bool cancelled_ = false;
};

class GenerationService {
public:
    GenerationService(ModelRuntime* runtime, int max_active_requests);
    ~GenerationService();

    GenerationService(const GenerationService&) = delete;
    GenerationService& operator=(const GenerationService&) = delete;

    void start();
    void stop();
    std::shared_ptr<GenerationStream> submit(GenerationRequest request);
    bool running() const;

private:
    struct Task {
        GenerationRequest request;
        std::shared_ptr<GenerationStream> stream;
        std::unique_ptr<GenerationSession> session;
    };

    void worker_loop();
    void admit_pending(std::vector<Task>* active);

    ModelRuntime* runtime_ = nullptr;
    int max_active_requests_ = 1;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<Task> pending_;
    std::thread worker_;
    bool running_ = false;
    bool stopping_ = false;
};

class HttpStreamingServer {
public:
    HttpStreamingServer(ModelRuntime* runtime, ServerConfig config);
    ~HttpStreamingServer();

    HttpStreamingServer(const HttpStreamingServer&) = delete;
    HttpStreamingServer& operator=(const HttpStreamingServer&) = delete;

    void run();
    void stop();

private:
    ModelRuntime* runtime_ = nullptr;
    ServerConfig config_;
    GenerationService service_;
    std::unique_ptr<httplib::Server> server_;
    int server_fd_ = -1;
    std::atomic<bool> stopping_{false};  // Guards idempotent shutdown across run/stop paths.
};

}  // namespace nano_llm_infer
