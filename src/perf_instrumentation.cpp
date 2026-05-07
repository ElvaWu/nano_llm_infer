#include "perf_instrumentation.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <utility>

namespace nano_llm_infer {

namespace {

std::ostream* default_log_stream() {
    return &std::clog;
}

}  // namespace

PerformanceInstrumentation::PerformanceInstrumentation(
    PerformanceInstrumentationConfig config,
    std::ostream* log_stream)
    : config_(std::move(config)),
      log_stream_(log_stream != nullptr ? log_stream : default_log_stream()) {
}

uint64_t PerformanceInstrumentation::next_request_id() {
    return next_request_id_++;
}

void PerformanceInstrumentation::on_request_submitted(uint64_t request_id, PerfClock::time_point now) {
    requests_[request_id] = RequestMetrics{
        request_id,
        now,
        PerfClock::time_point{},
        PerfClock::time_point{},
        0,
        false,
        false,
        false,
        false,
        "",
    };
}

void PerformanceInstrumentation::on_request_admitted(uint64_t request_id, PerfClock::time_point now) {
    RequestMetrics* metrics = find_request(request_id);
    if (metrics == nullptr) {
        on_request_submitted(request_id, now);
        metrics = find_request(request_id);
    }
    if (metrics != nullptr && !metrics->active) {
        metrics->active = true;
        set_active_request_count(active_request_count_.load(std::memory_order_relaxed) + 1);
    }
}

void PerformanceInstrumentation::on_request_cancelled(
    uint64_t request_id,
    PerfClock::time_point now,
    bool was_active) {
    RequestMetrics* metrics = find_request(request_id);
    if (metrics == nullptr) {
        return;
    }
    metrics->cancelled = true;
    on_request_completed(request_id, now, "cancelled");
    if (!was_active) {
        set_current_batch_size(0);
    }
}

void PerformanceInstrumentation::on_decode_step(
    size_t batch_size,
    size_t tokens_generated,
    PerfClock::time_point now) {
    set_current_batch_size(batch_size);
    set_tokens_generated_last_step(tokens_generated);
    total_tokens_generated_.fetch_add(static_cast<uint64_t>(tokens_generated), std::memory_order_relaxed);

    if (throughput_window_.window_start == PerfClock::time_point{}) {
        throughput_window_.window_start = now;
    }
    throughput_window_.tokens_in_window += static_cast<uint64_t>(tokens_generated);

    const auto elapsed = now - throughput_window_.window_start;
    if (elapsed > PerfClock::duration::zero()) {
        const double seconds = std::chrono::duration<double>(elapsed).count();
        set_tokens_per_second(static_cast<double>(throughput_window_.tokens_in_window) / seconds);
    }
}

void PerformanceInstrumentation::on_token_emitted(uint64_t request_id, PerfClock::time_point now) {
    RequestMetrics* metrics = find_request(request_id);
    if (metrics == nullptr) {
        return;
    }
    if (!metrics->saw_first_token) {
        metrics->first_token_time = now;
        metrics->saw_first_token = true;
    }
    metrics->total_tokens_generated++;
}

void PerformanceInstrumentation::on_request_completed(
    uint64_t request_id,
    PerfClock::time_point now,
    const std::string& terminal_reason) {
    auto it = requests_.find(request_id);
    if (it == requests_.end()) {
        return;
    }

    RequestMetrics metrics = it->second;
    metrics.completion_time = now;
    metrics.completed = true;
    metrics.terminal_reason = terminal_reason;

    log_request_completion(metrics);

    if (it->second.active) {
        const size_t active_count = active_request_count_.load(std::memory_order_relaxed);
        set_active_request_count(active_count > 0 ? active_count - 1 : 0);
    }
    requests_.erase(it);
    completed_request_count_.fetch_add(1, std::memory_order_relaxed);
}

void PerformanceInstrumentation::maybe_log_periodic(PerfClock::time_point now) {
    if (!config_.enable_periodic_logging || config_.log_interval_ms <= 0) {
        return;
    }

    if (last_periodic_log_time_ == PerfClock::time_point{}) {
        last_periodic_log_time_ = now;
        return;
    }

    const auto interval = std::chrono::milliseconds(config_.log_interval_ms);
    if (now - last_periodic_log_time_ < interval) {
        return;
    }

    const auto elapsed = now - throughput_window_.window_start;
    if (throughput_window_.window_start != PerfClock::time_point{} &&
        elapsed > PerfClock::duration::zero()) {
        const double seconds = std::chrono::duration<double>(elapsed).count();
        set_tokens_per_second(static_cast<double>(throughput_window_.tokens_in_window) / seconds);
    } else {
        set_tokens_per_second(0.0);
    }

    if (log_stream_ != nullptr) {
        std::lock_guard<std::mutex> lock(log_mutex_);
        (*log_stream_) << "[perf] active_requests=" << active_request_count_.load(std::memory_order_relaxed)
                       << " batch_size=" << current_batch_size_.load(std::memory_order_relaxed)
                       << " tokens_last_step=" << tokens_generated_last_step_.load(std::memory_order_relaxed)
                       << " tokens_per_sec="
                       << std::fixed << std::setprecision(2)
                       << snapshot().tokens_per_second
                       << " total_tokens=" << total_tokens_generated_.load(std::memory_order_relaxed)
                       << '\n';
    }

    throughput_window_.window_start = now;
    throughput_window_.tokens_in_window = 0;
    last_periodic_log_time_ = now;
}

SystemMetricsSnapshot PerformanceInstrumentation::snapshot() const {
    SystemMetricsSnapshot snapshot;
    snapshot.active_request_count = active_request_count_.load(std::memory_order_relaxed);
    snapshot.current_batch_size = current_batch_size_.load(std::memory_order_relaxed);
    snapshot.tokens_generated_last_step = tokens_generated_last_step_.load(std::memory_order_relaxed);
    snapshot.tokens_per_second =
        static_cast<double>(tokens_per_second_milli_.load(std::memory_order_relaxed)) / 1000.0;
    snapshot.total_tokens_generated = total_tokens_generated_.load(std::memory_order_relaxed);
    snapshot.completed_request_count = completed_request_count_.load(std::memory_order_relaxed);
    return snapshot;
}

void PerformanceInstrumentation::set_active_request_count(size_t count) {
    active_request_count_.store(count, std::memory_order_relaxed);
}

void PerformanceInstrumentation::set_current_batch_size(size_t batch_size) {
    current_batch_size_.store(batch_size, std::memory_order_relaxed);
}

void PerformanceInstrumentation::set_tokens_generated_last_step(size_t tokens_generated) {
    tokens_generated_last_step_.store(tokens_generated, std::memory_order_relaxed);
}

void PerformanceInstrumentation::set_tokens_per_second(double value) {
    const double clamped = std::max(0.0, value);
    tokens_per_second_milli_.store(
        static_cast<uint64_t>(clamped * 1000.0),
        std::memory_order_relaxed);
}

void PerformanceInstrumentation::log_request_completion(const RequestMetrics& metrics) const {
    if (log_stream_ == nullptr) {
        return;
    }

    std::optional<double> ttft_ms;
    std::optional<double> tpot_ms;
    if (metrics.saw_first_token) {
        ttft_ms = duration_ms(metrics.first_token_time - metrics.arrival_time);
        if (metrics.total_tokens_generated > 0) {
            tpot_ms = duration_ms(metrics.completion_time - metrics.first_token_time) /
                static_cast<double>(metrics.total_tokens_generated);
        }
    }

    std::lock_guard<std::mutex> lock(log_mutex_);
    (*log_stream_) << "[perf] request_id=" << metrics.request_id
                   << " reason=" << metrics.terminal_reason
                   << " total_tokens=" << metrics.total_tokens_generated
                   << " ttft_ms=";
    if (ttft_ms.has_value()) {
        (*log_stream_) << std::fixed << std::setprecision(2) << *ttft_ms;
    } else {
        (*log_stream_) << "n/a";
    }
    (*log_stream_) << " tpot_ms=";
    if (tpot_ms.has_value()) {
        (*log_stream_) << std::fixed << std::setprecision(2) << *tpot_ms;
    } else {
        (*log_stream_) << "n/a";
    }
    (*log_stream_) << '\n';
}

double PerformanceInstrumentation::duration_ms(PerfClock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

RequestMetrics* PerformanceInstrumentation::find_request(uint64_t request_id) {
    const auto it = requests_.find(request_id);
    if (it == requests_.end()) {
        return nullptr;
    }
    return &it->second;
}

}  // namespace nano_llm_infer
