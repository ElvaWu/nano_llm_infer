#include "generation_server.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <csignal>
#include <exception>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace nano_llm_infer {

namespace {

using json = nlohmann::json;

constexpr auto kNoProgressWait = std::chrono::milliseconds(2);
constexpr int kMaxServicePrefillBatch = 16;

volatile std::sig_atomic_t g_signal_stop_requested = 0;  // Set in signal handler, read from run loop.

std::string json_error(const std::string& message) {
    json body;
    body["error"] = {
        {"message", message},
        {"type", "invalid_request_error"},
    };
    return body.dump();
}

void handle_signal(int) {
    // Async-signal-safe path: avoid allocations/locks and only set a flag.
    g_signal_stop_requested = 1;
}

std::string make_completion_json(const std::string& model_id, const std::string& text, const std::string& finish_reason) {
    json message;
    message["role"] = "assistant";
    message["content"] = text;

    json choice;
    choice["index"] = 0;
    choice["message"] = message;
    choice["finish_reason"] = finish_reason;

    json body;
    body["id"] = "chatcmpl-nano";
    body["object"] = "chat.completion";
    body["model"] = model_id;
    body["choices"] = json::array({choice});
    return body.dump();
}

}  // namespace

ParseResult parse_chat_completion_request(const std::string& body) {
    ParseResult result;
    try {
        const json root = json::parse(body);
        if (!root.is_object()) {
            result.error = "request body must be a JSON object";
            return result;
        }

        const auto model_it = root.find("model");
        if (model_it == root.end() || !model_it->is_string() || model_it->get<std::string>().empty()) {
            result.error = "field 'model' must be a non-empty string";
            return result;
        }
        result.request.model = model_it->get<std::string>();

        const auto stream_it = root.find("stream");
        if (stream_it != root.end() && !stream_it->is_null()) {
            if (!stream_it->is_boolean()) {
                result.error = "field 'stream' must be a boolean";
                return result;
            }
            result.request.stream = stream_it->get<bool>();
        }

        const auto messages_it = root.find("messages");
        if (messages_it == root.end() || !messages_it->is_array() || messages_it->empty()) {
            result.error = "field 'messages' must be a non-empty array";
            return result;
        }

        for (const auto& message : *messages_it) {
            if (!message.is_object()) {
                result.error = "each message must include string 'role' and 'content' fields";
                return result;
            }

            const auto role_it = message.find("role");
            const auto content_it = message.find("content");
            if (role_it == message.end() || content_it == message.end() ||
                !role_it->is_string() || !content_it->is_string()) {
                result.error = "each message must include string 'role' and 'content' fields";
                return result;
            }

            result.request.generation.messages.push_back({
                role_it->get<std::string>(),
                content_it->get<std::string>(),
            });
        }

        const auto max_tokens_it = root.find("max_tokens");
        if (max_tokens_it != root.end() && !max_tokens_it->is_null()) {
            if (!max_tokens_it->is_number_integer()) {
                result.error = "field 'max_tokens' must be a positive integer";
                return result;
            }

            const int max_tokens = max_tokens_it->get<int>();
            if (max_tokens <= 0) {
                result.error = "field 'max_tokens' must be a positive integer";
                return result;
            }
            result.request.generation.options.max_tokens = max_tokens;
        }

        result.ok = true;
        return result;
    } catch (const std::exception& e) {
        result.error = e.what();
        return result;
    }
}

std::string json_escape(const std::string& text) {
    const std::string encoded = json(text).dump();
    return encoded.substr(1, encoded.size() - 2);
}

std::string make_sse_chunk(const std::string& model_id, const GenerationEvent& event) {
    if (!event.error.empty()) {
        json body;
        body["error"] = {
            {"message", event.error},
            {"type", "server_error"},
        };
        return "data: " + body.dump() + "\n\n";
    }

    json choice;
    choice["index"] = 0;
    if (!event.finish_reason.empty()) {
        choice["delta"] = json::object();
        choice["finish_reason"] = event.finish_reason;
    } else {
        choice["delta"] = {{"content", event.text}};
        choice["finish_reason"] = nullptr;
    }

    json body;
    body["id"] = "chatcmpl-nano";
    body["object"] = "chat.completion.chunk";
    body["model"] = model_id;
    body["choices"] = json::array({choice});
    return "data: " + body.dump() + "\n\n";
}

std::string make_sse_done() {
    return "data: [DONE]\n\n";
}

bool GenerationStream::wait_next(GenerationEvent* event) {
    std::chrono::steady_clock::time_point ignored;
    return wait_next_timed(event, &ignored);
}

bool GenerationStream::wait_next_timed(
    GenerationEvent* event,
    std::chrono::steady_clock::time_point* time) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&] { return !events_.empty() || closed_ || cancelled_; });
    if (events_.empty()) {
        return false;
    }
    TimedEvent timed = std::move(events_.front());
    events_.pop_front();
    *event = std::move(timed.event);
    if (time != nullptr) {
        *time = timed.time;
    }
    return true;
}

void GenerationStream::cancel() {
    std::lock_guard<std::mutex> lock(mutex_);
    cancelled_ = true;
    cv_.notify_all();
}

bool GenerationStream::cancelled() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cancelled_;
}

std::chrono::steady_clock::time_point GenerationStream::submit_time() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return submit_time_;
}

std::chrono::steady_clock::time_point GenerationStream::admit_time() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return admit_time_;
}

std::chrono::steady_clock::time_point GenerationStream::close_time() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return close_time_;
}

void GenerationStream::set_submit_time(std::chrono::steady_clock::time_point time) {
    std::lock_guard<std::mutex> lock(mutex_);
    submit_time_ = time;
}

void GenerationStream::set_admit_time(std::chrono::steady_clock::time_point time) {
    std::lock_guard<std::mutex> lock(mutex_);
    admit_time_ = time;
}

void GenerationStream::push(GenerationEvent event) {
    push(std::move(event), std::chrono::steady_clock::now());
}

void GenerationStream::push(GenerationEvent event, std::chrono::steady_clock::time_point time) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (cancelled_) {
        return;
    }
    events_.push_back(TimedEvent{std::move(event), time});
    cv_.notify_all();
}

void GenerationStream::close() {
    close_at(std::chrono::steady_clock::now());
}

void GenerationStream::close_at(std::chrono::steady_clock::time_point time) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (close_time_ == std::chrono::steady_clock::time_point{}) {
        close_time_ = time;
    }
    closed_ = true;
    cv_.notify_all();
}

GenerationService::GenerationService(
    ModelRuntime* runtime,
    int max_active_requests,
    int max_decode_batch_size)
    : GenerationService(
          runtime,
          max_active_requests,
          max_decode_batch_size,
          PerformanceInstrumentationConfig{}) {
}

GenerationService::GenerationService(
    ModelRuntime* runtime,
    int max_active_requests,
    int max_decode_batch_size,
    PerformanceInstrumentationConfig perf_config)
    : runtime_(runtime),
      max_active_requests_(std::max(1, max_active_requests)),
      max_decode_batch_size_(std::max(
          1,
          max_decode_batch_size > 0 ? max_decode_batch_size : std::max(1, max_active_requests))),
      max_prefill_batch_size_(std::min(max_decode_batch_size_, kMaxServicePrefillBatch)),
      instrumentation_(std::move(perf_config)) {
    if (runtime_ == nullptr) {
        throw std::invalid_argument("GenerationService requires a non-null runtime");
    }
    runtime_->ensure_kv_arena_capacity(max_active_requests_);
}

GenerationService::~GenerationService() {
    stop();
}

void GenerationService::start() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (running_) {
        return;
    }
    stopping_ = false;
    running_ = true;
    worker_ = std::thread(&GenerationService::worker_loop, this);
}

void GenerationService::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!running_) {
            return;
        }
        stopping_ = true;
        cv_.notify_all();
    }
    if (worker_.joinable()) {
        worker_.join();
    }
    std::lock_guard<std::mutex> lock(mutex_);
    running_ = false;
}

std::shared_ptr<GenerationStream> GenerationService::submit(GenerationRequest request) {
    auto stream = std::make_shared<GenerationStream>();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        Task task;
        task.request = std::move(request);
        task.stream = stream;
        task.request_id = instrumentation_.next_request_id();
        const PerfClock::time_point submit_time = PerfClock::now();
        task.stream->set_submit_time(submit_time);
        instrumentation_.on_request_submitted(task.request_id, submit_time);
        pending_.push_back(std::move(task));
    }
    cv_.notify_all();
    return stream;
}

bool GenerationService::running() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return running_ && !stopping_;
}

SystemMetricsSnapshot GenerationService::snapshot_metrics() const {
    return instrumentation_.snapshot();
}

void GenerationService::admit_pending(std::vector<Task>* active) {
    while (static_cast<int>(active->size()) < max_active_requests_ && !pending_.empty()) {
        Task task = std::move(pending_.front());
        pending_.pop_front();
        task.session = runtime_->create_session();
        // Admission claims a stable shared KV arena slot for the whole active
        // request lifetime; cleanup releases it exactly once.
        runtime_->assign_kv_slot(task.session.get());
        task.state = Task::State::kNeedsPrefill;
        const PerfClock::time_point admit_time = PerfClock::now();
        task.stream->set_admit_time(admit_time);
        instrumentation_.on_request_admitted(task.request_id, admit_time);
        active->push_back(std::move(task));
    }
}

void GenerationService::close_cancelled_pending() {
    for (size_t i = 0; i < pending_.size();) {
        if (pending_[i].stream->cancelled()) {
            instrumentation_.on_request_cancelled(pending_[i].request_id, PerfClock::now(), false);
            pending_[i].stream->close();
            pending_.erase(pending_.begin() + static_cast<std::ptrdiff_t>(i));
            continue;
        }
        ++i;
    }
}

void GenerationService::normalize_next_decode_index(size_t active_size) {
    if (active_size == 0) {
        next_decode_index_ = 0;
        return;
    }
    next_decode_index_ %= active_size;
}

void GenerationService::cleanup_retired(std::vector<Task>* active) {
    for (size_t i = 0; i < active->size();) {
        Task& task = (*active)[i];
        if (task.stream->cancelled() || task.state == Task::State::kTerminal) {
            if (task.stream->cancelled()) {
                instrumentation_.on_request_cancelled(task.request_id, PerfClock::now(), true);
            }
            runtime_->release_kv_slot(task.session.get());
            task.stream->close();
            active->erase(active->begin() + static_cast<std::ptrdiff_t>(i));
            normalize_next_decode_index(active->size());
            continue;
        }
        ++i;
    }
}

bool GenerationService::prefill_next_batch(std::vector<Task>* active) {
    if (active->empty()) {
        return false;
    }

    // Collect pending-prefill tasks up to the runtime prefill graph limit. Variable-
    // length prompts are fine — prefill_batch handles them in one fused graph.
    std::vector<size_t> selected;
    selected.reserve(std::min(static_cast<size_t>(max_prefill_batch_size_), active->size()));

    const size_t start = next_decode_index_ % active->size();
    for (size_t offset = 0; offset < active->size(); ++offset) {
        if (selected.size() >= static_cast<size_t>(max_prefill_batch_size_)) {
            break;
        }
        const size_t index = (start + offset) % active->size();
        Task& task = (*active)[index];
        if (task.state != Task::State::kNeedsPrefill || task.stream->cancelled()) {
            continue;
        }
        selected.push_back(index);
    }

    if (selected.empty()) {
        return false;
    }

    std::vector<std::pair<GenerationSession*, const GenerationRequest*>> prefill_items;
    prefill_items.reserve(selected.size());
    for (size_t index : selected) {
        Task& task = (*active)[index];
        prefill_items.push_back({task.session.get(), &task.request});
    }

    try {
        runtime_->prefill_batch(prefill_items);
        for (size_t index : selected) {
            (*active)[index].state = Task::State::kReadyToDecode;
        }
    } catch (const std::exception& e) {
        const PerfClock::time_point error_time = PerfClock::now();
        for (size_t index : selected) {
            Task& task = (*active)[index];
            GenerationEvent event;
            event.error = e.what();
            task.stream->push(std::move(event), error_time);
            task.stream->close_at(error_time);
            task.state = Task::State::kTerminal;
            instrumentation_.on_request_completed(task.request_id, error_time, "error");
        }
    }
    return true;
}

std::vector<size_t> GenerationService::select_decode_batch(const std::vector<Task>& active) {
    std::vector<size_t> selected;
    if (active.empty()) {
        return selected;
    }

    selected.reserve(std::min(static_cast<size_t>(max_decode_batch_size_), active.size()));
    // Rotate the starting point so long-lived requests do not always lose the
    // last batch slot when there are more active streams than decode capacity.
    const size_t start = next_decode_index_ % active.size();
    size_t visited = 0;
    for (; visited < active.size() && selected.size() < static_cast<size_t>(max_decode_batch_size_); ++visited) {
        const size_t index = (start + visited) % active.size();
        if (active[index].state == Task::State::kReadyToDecode && !active[index].stream->cancelled()) {
            selected.push_back(index);
        }
    }

    const size_t advance = selected.empty() ? 1 : std::max<size_t>(selected.size(), 1);
    next_decode_index_ = (start + advance) % active.size();
    return selected;
}

void GenerationService::worker_loop() {
    std::vector<Task> active;

    while (true) {
        instrumentation_.maybe_log_periodic(PerfClock::now());
        bool made_progress = false;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [&] { return stopping_ || !pending_.empty() || !active.empty(); });
            const size_t pending_before = pending_.size();
            close_cancelled_pending();
            admit_pending(&active);
            if (pending_.size() != pending_before) {
                made_progress = true;
            }
            if (stopping_ && pending_.empty() && active.empty()) {
                break;
            }
        }

        const size_t active_before_cleanup = active.size();
        cleanup_retired(&active);
        if (active.size() != active_before_cleanup) {
            made_progress = true;
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            const size_t pending_before = pending_.size();
            close_cancelled_pending();
            admit_pending(&active);
            if (pending_.size() != pending_before) {
                made_progress = true;
            }
            if (stopping_ && pending_.empty() && active.empty()) {
                break;
            }
        }
        const size_t active_before_second_cleanup = active.size();
        cleanup_retired(&active);
        if (active.size() != active_before_second_cleanup) {
            made_progress = true;
        }

        if (active.empty()) {
            instrumentation_.on_decode_step(0, 0, PerfClock::now());
            continue;
        }

        // Give at most one newly admitted request its prefill work before the
        // next batched decode step so join/quit decisions are re-evaluated each
        // iteration instead of at whole-request boundaries.
        if (prefill_next_batch(&active)) {
            made_progress = true;
            const size_t active_before_prefill_cleanup = active.size();
            cleanup_retired(&active);
            if (active.size() != active_before_prefill_cleanup) {
                made_progress = true;
            }
            continue;
        }

        std::vector<size_t> decode_indices = select_decode_batch(active);
        if (decode_indices.empty()) {
            instrumentation_.on_decode_step(0, 0, PerfClock::now());
            if (!made_progress) {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait_for(lock, kNoProgressWait, [&] { return stopping_ || !pending_.empty(); });
            }
            continue;
        }

        std::vector<DecodeBatchItem> batch(decode_indices.size());
        for (size_t i = 0; i < decode_indices.size(); ++i) {
            Task& task = active[decode_indices[i]];
            batch[i].session = task.session.get();
            batch[i].request = &task.request;
        }

        try {
            runtime_->decode_batch(&batch);
        } catch (const std::exception& e) {
            // A batched decode failure should surface to every stream that
            // participated in that batch, not just the first task we inspect.
            for (DecodeBatchItem& item : batch) {
                item.event.error = e.what();
                item.produced = true;
            }
        }

        size_t tokens_generated_this_step = 0;
        const PerfClock::time_point step_now = PerfClock::now();

        for (size_t i = 0; i < decode_indices.size(); ++i) {
            Task& task = active[decode_indices[i]];
            const DecodeBatchItem& item = batch[i];
            if (!item.produced) {
                task.stream->close_at(step_now);
                task.state = Task::State::kTerminal;
                instrumentation_.on_request_completed(task.request_id, step_now, "closed");
                made_progress = true;
                continue;
            }

            if (item.event.token_generated) {
                instrumentation_.on_token_emitted(task.request_id, step_now);
                ++tokens_generated_this_step;
            }
            task.stream->push(item.event, step_now);
            made_progress = true;
            if (!item.event.finish_reason.empty() || !item.event.error.empty()) {
                task.stream->close_at(step_now);
                task.state = Task::State::kTerminal;
                const std::string terminal_reason =
                    !item.event.error.empty() ? "error" : item.event.finish_reason;
                instrumentation_.on_request_completed(task.request_id, step_now, terminal_reason);
            }
        }

        instrumentation_.on_decode_step(decode_indices.size(), tokens_generated_this_step, step_now);

        const size_t active_before_final_cleanup = active.size();
        cleanup_retired(&active);
        if (active.size() != active_before_final_cleanup) {
            made_progress = true;
        }

        if (!made_progress) {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait_for(lock, kNoProgressWait, [&] { return stopping_ || !pending_.empty(); });
        }
    }

    for (Task& task : active) {
        runtime_->release_kv_slot(task.session.get());
        task.stream->close();
    }

    std::lock_guard<std::mutex> lock(mutex_);
    while (!pending_.empty()) {
        pending_.front().stream->close();
        pending_.pop_front();
    }
}

HttpStreamingServer::HttpStreamingServer(ModelRuntime* runtime, ServerConfig config)
    : runtime_(runtime),
      config_(std::move(config)),
      service_(
          runtime,
          config_.max_active_requests,
          config_.max_decode_batch_size,
          PerformanceInstrumentationConfig{
              config_.enable_perf_logging,
              config_.perf_log_interval_ms,
          }) {
}

HttpStreamingServer::~HttpStreamingServer() {
    stop();
}

void HttpStreamingServer::run() {
    if (runtime_ == nullptr) {
        throw std::invalid_argument("runtime is null");
    }

    service_.start();
    stopping_.store(false, std::memory_order_relaxed);
    g_signal_stop_requested = 0;

    server_ = std::make_unique<httplib::Server>();
    server_->new_task_queue = [this] {
        return new httplib::ThreadPool(static_cast<size_t>(std::max(2, config_.listen_backlog)));
    };

    server_->Get("/healthz", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"ok\":true}", "application/json");
    });

    server_->Post("/v1/chat/completions", [this](const httplib::Request& req, httplib::Response& res) {
        ParseResult parsed = parse_chat_completion_request(req.body);
        if (!parsed.ok) {
            res.status = 400;
            res.set_content(json_error(parsed.error), "application/json");
            return;
        }

        auto stream = service_.submit(std::move(parsed.request.generation));
        const std::string model_id = parsed.request.model;

        if (!parsed.request.stream) {
            // Non-stream mode aggregates incremental events into one final response.
            std::string text;
            std::string finish_reason = "stop";
            while (true) {
                GenerationEvent event;
                if (!stream->wait_next(&event)) {
                    break;
                }
                if (!event.error.empty()) {
                    res.status = 500;
                    res.set_content(json_error(event.error), "application/json");
                    return;
                }
                text += event.text;
                if (!event.finish_reason.empty()) {
                    finish_reason = event.finish_reason;
                    break;
                }
            }
            res.set_content(make_completion_json(model_id, text, finish_reason), "application/json");
            return;
        }

        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");
        res.set_chunked_content_provider(
            "text/event-stream",
            [stream, model_id](size_t, httplib::DataSink& sink) {
                GenerationEvent event;
                if (!stream->wait_next(&event)) {
                    const std::string done = make_sse_done();
                    sink.write(done.data(), done.size());
                    sink.done();
                    return false;
                }

                const std::string chunk = make_sse_chunk(model_id, event);
                if (!sink.write(chunk.data(), chunk.size())) {
                    stream->cancel();
                    sink.done();
                    return false;
                }

                if (!event.finish_reason.empty() || !event.error.empty()) {
                    const std::string done = make_sse_done();
                    sink.write(done.data(), done.size());
                    sink.done();
                    return false;
                }
                return true;
            },
            [stream](bool) {
                stream->cancel();
            });
    });

    const auto previous_int = std::signal(SIGINT, handle_signal);
    const auto previous_term = std::signal(SIGTERM, handle_signal);

    std::mutex listener_mutex;
    std::condition_variable listener_cv;
    bool listener_finished = false;
    bool listen_ok = false;
    std::string listen_error;

    std::thread listener([this, &listener_mutex, &listener_cv, &listener_finished, &listen_ok, &listen_error] {
        try {
            listen_ok = server_->listen(config_.host.c_str(), config_.port);
            if (!listen_ok && !stopping_.load(std::memory_order_relaxed)) {
                listen_error =
                    "failed to listen on " + config_.host + ":" + std::to_string(config_.port);
            }
        } catch (const std::exception& e) {
            listen_error = e.what();
        } catch (...) {
            listen_error = "unknown listener failure";
        }

        {
            std::lock_guard<std::mutex> lock(listener_mutex);
            listener_finished = true;
        }
        listener_cv.notify_one();
    });
    server_->wait_until_ready();

    if (!server_->is_running()) {
        {
            std::unique_lock<std::mutex> lock(listener_mutex);
            listener_cv.wait(lock, [&] { return listener_finished; });
        }
        if (listener.joinable()) {
            listener.join();
        }

        std::signal(SIGINT, previous_int);
        std::signal(SIGTERM, previous_term);

        server_.reset();
        server_fd_ = -1;
        service_.stop();

        if (listen_error.empty()) {
            listen_error = "server failed to start listening";
        }
        throw std::runtime_error(listen_error);
    }

    while (server_->is_running()) {
        // Poll signal flag in normal thread context and stop gracefully.
        if (g_signal_stop_requested != 0) {
            stop();
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    if (listener.joinable()) {
        listener.join();
    }

    std::signal(SIGINT, previous_int);
    std::signal(SIGTERM, previous_term);

    server_.reset();
    server_fd_ = -1;
    service_.stop();

    if (!stopping_.load(std::memory_order_relaxed) && g_signal_stop_requested != 0) {
        stop();
    }
}

void HttpStreamingServer::stop() {
    // Idempotent stop: first caller performs shutdown, others no-op.
    const bool already_stopping = stopping_.exchange(true, std::memory_order_relaxed);
    if (already_stopping) {
        return;
    }
    if (server_) {
        server_->stop();
    }
    service_.stop();
}

}  // namespace nano_llm_infer
