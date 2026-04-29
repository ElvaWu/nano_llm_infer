#include "generation_server.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <csignal>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace nano_llm_infer {

namespace {

using json = nlohmann::json;

HttpStreamingServer* g_signal_server = nullptr;

std::string json_error(const std::string& message) {
    json body;
    body["error"] = {
        {"message", message},
        {"type", "invalid_request_error"},
    };
    return body.dump();
}

void handle_signal(int) {
    if (g_signal_server != nullptr) {
        g_signal_server->stop();
    }
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
        if (stream_it == root.end() || !stream_it->is_boolean() || !stream_it->get<bool>()) {
            result.error = "field 'stream' must be true";
            return result;
        }
        result.request.stream = true;

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
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&] { return !events_.empty() || closed_ || cancelled_; });
    if (events_.empty()) {
        return false;
    }
    *event = std::move(events_.front());
    events_.pop_front();
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

void GenerationStream::push(GenerationEvent event) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (cancelled_) {
        return;
    }
    events_.push_back(std::move(event));
    cv_.notify_all();
}

void GenerationStream::close() {
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
    cv_.notify_all();
}

GenerationService::GenerationService(ModelRuntime* runtime, int max_active_requests)
    : runtime_(runtime),
      max_active_requests_(std::max(1, max_active_requests)) {
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
        pending_.push_back(Task{std::move(request), stream, nullptr});
    }
    cv_.notify_all();
    return stream;
}

bool GenerationService::running() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return running_ && !stopping_;
}

void GenerationService::admit_pending(std::vector<Task>* active) {
    while (static_cast<int>(active->size()) < max_active_requests_ && !pending_.empty()) {
        Task task = std::move(pending_.front());
        pending_.pop_front();
        task.session = runtime_->create_session();
        active->push_back(std::move(task));
    }
}

void GenerationService::worker_loop() {
    std::vector<Task> active;

    while (true) {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [&] { return stopping_ || !pending_.empty() || !active.empty(); });
            admit_pending(&active);
            if (stopping_ && pending_.empty() && active.empty()) {
                break;
            }
        }

        for (size_t i = 0; i < active.size();) {
            Task& task = active[i];
            if (task.stream->cancelled()) {
                task.stream->close();
                active.erase(active.begin() + static_cast<std::ptrdiff_t>(i));
                continue;
            }

            GenerationEvent event;
            try {
                const bool produced = runtime_->decode_next(task.session.get(), task.request, &event);
                if (!produced) {
                    task.stream->close();
                    active.erase(active.begin() + static_cast<std::ptrdiff_t>(i));
                    continue;
                }
            } catch (const std::exception& e) {
                event.error = e.what();
            }

            task.stream->push(event);
            if (!event.finish_reason.empty() || !event.error.empty()) {
                task.stream->close();
                active.erase(active.begin() + static_cast<std::ptrdiff_t>(i));
                continue;
            }
            ++i;
        }
    }

    for (Task& task : active) {
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
      service_(runtime, config_.max_active_requests) {
}

HttpStreamingServer::~HttpStreamingServer() {
    stop();
}

void HttpStreamingServer::run() {
    if (runtime_ == nullptr) {
        throw std::invalid_argument("runtime is null");
    }

    service_.start();
    stopping_ = false;
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

    server_fd_ = 1;
    g_signal_server = this;
    const auto previous_int = std::signal(SIGINT, handle_signal);
    const auto previous_term = std::signal(SIGTERM, handle_signal);

    std::cout << "Listening on http://" << config_.host << ':' << config_.port << '\n';
    const bool ok = server_->listen(config_.host.c_str(), config_.port);

    std::signal(SIGINT, previous_int);
    std::signal(SIGTERM, previous_term);
    g_signal_server = nullptr;
    server_.reset();
    server_fd_ = -1;
    service_.stop();

    if (!ok && !stopping_) {
        throw std::runtime_error("failed to start HTTP server");
    }
}

void HttpStreamingServer::stop() {
    stopping_ = true;
    if (server_) {
        server_->stop();
    }
    service_.stop();
}

}  // namespace nano_llm_infer
