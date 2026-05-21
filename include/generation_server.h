#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "generation.h"
#include "perf_instrumentation.h"

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
    int max_decode_batch_size = 0;
    int listen_backlog = 16;
    bool enable_perf_logging = true;
    int perf_log_interval_ms = 1000;
};

ParseResult parse_chat_completion_request(const std::string& body);
std::string json_escape(const std::string& text);
std::string make_sse_chunk(const std::string& model_id, const GenerationEvent& event);
std::string make_sse_done();

class GenerationStream {
public:
    bool wait_next(GenerationEvent* event);
    bool wait_next_timed(GenerationEvent* event, std::chrono::steady_clock::time_point* time);
    void cancel();
    bool cancelled() const;
    std::chrono::steady_clock::time_point submit_time() const;
    std::chrono::steady_clock::time_point admit_time() const;
    std::chrono::steady_clock::time_point close_time() const;

private:
    friend class GenerationService;

    struct TimedEvent {
        GenerationEvent event;
        std::chrono::steady_clock::time_point time{};
    };

    void set_submit_time(std::chrono::steady_clock::time_point time);
    void set_admit_time(std::chrono::steady_clock::time_point time);
    void push(GenerationEvent event);
    void push(GenerationEvent event, std::chrono::steady_clock::time_point time);
    void close();
    void close_at(std::chrono::steady_clock::time_point time);

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<TimedEvent> events_;
    std::chrono::steady_clock::time_point submit_time_{};
    std::chrono::steady_clock::time_point admit_time_{};
    std::chrono::steady_clock::time_point close_time_{};
    bool closed_ = false;
    bool cancelled_ = false;
};

class GenerationService {
public:
    GenerationService(ModelRuntime* runtime, int max_active_requests, int max_decode_batch_size);
    GenerationService(
        ModelRuntime* runtime,
        int max_active_requests,
        int max_decode_batch_size,
        PerformanceInstrumentationConfig perf_config);
    ~GenerationService();

    GenerationService(const GenerationService&) = delete;
    GenerationService& operator=(const GenerationService&) = delete;

    void start();
    void stop();
    std::shared_ptr<GenerationStream> submit(GenerationRequest request);
    bool running() const;
    SystemMetricsSnapshot snapshot_metrics() const;

private:
    struct Task {
        enum class State {
            kNeedsPrefill,
            kReadyToDecode,
            kTerminal,
        };

        GenerationRequest request;
        std::shared_ptr<GenerationStream> stream;
        std::unique_ptr<GenerationSession> session;
        uint64_t request_id = 0;
        State state = State::kNeedsPrefill;  // Scheduler-owned lifecycle between admission and retirement.
    };

    void worker_loop();
    void admit_pending(std::vector<Task>* active);
    void close_cancelled_pending();
    void cleanup_retired(std::vector<Task>* active);
    bool prefill_next_batch(std::vector<Task>* active);
    void normalize_next_decode_index(size_t active_size);
    std::vector<size_t> select_decode_batch(const std::vector<Task>& active);

    ModelRuntime* runtime_ = nullptr;
    int max_active_requests_ = 1;
    int max_decode_batch_size_ = 1;
    int max_prefill_batch_size_ = 1;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<Task> pending_;
    std::thread worker_;
    bool running_ = false;
    bool stopping_ = false;
    size_t next_decode_index_ = 0;
    PerformanceInstrumentation instrumentation_;
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
