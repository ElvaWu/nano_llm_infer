#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

namespace nano_llm_infer {

using PerfClock = std::chrono::steady_clock;

struct PerformanceInstrumentationConfig {
    bool enable_periodic_logging = true;
    int log_interval_ms = 1000;
};

struct RequestMetrics {
    uint64_t request_id = 0;
    PerfClock::time_point arrival_time{};
    PerfClock::time_point admit_time{};
    PerfClock::time_point first_token_time{};
    PerfClock::time_point last_token_time{};
    PerfClock::time_point completion_time{};
    PerfClock::duration inter_token_duration{};
    uint64_t total_tokens_generated = 0;
    uint64_t inter_token_interval_count = 0;
    bool saw_first_token = false;
    bool active = false;
    bool completed = false;
    bool cancelled = false;
    std::string terminal_reason;
};

struct SystemMetricsSnapshot {
    size_t active_request_count = 0;
    size_t current_batch_size = 0;
    size_t tokens_generated_last_step = 0;
    double tokens_per_second = 0.0;
    uint64_t total_tokens_generated = 0;
    uint64_t completed_request_count = 0;
    uint64_t decode_batch_size_total = 0;
    uint64_t decode_step_count = 0;
};

class PerformanceInstrumentation {
public:
    explicit PerformanceInstrumentation(
        PerformanceInstrumentationConfig config = PerformanceInstrumentationConfig{},
        std::ostream* log_stream = nullptr);

    uint64_t next_request_id();
    void on_request_submitted(uint64_t request_id, PerfClock::time_point now);
    void on_request_admitted(uint64_t request_id, PerfClock::time_point now);
    void on_request_cancelled(uint64_t request_id, PerfClock::time_point now, bool was_active);
    void on_decode_step(size_t batch_size, size_t tokens_generated, PerfClock::time_point now);
    void on_token_emitted(uint64_t request_id, PerfClock::time_point now);
    void on_request_completed(
        uint64_t request_id,
        PerfClock::time_point now,
        const std::string& terminal_reason);
    void maybe_log_periodic(PerfClock::time_point now);

    SystemMetricsSnapshot snapshot() const;

private:
    struct ThroughputWindow {
        PerfClock::time_point window_start{};
        uint64_t tokens_in_window = 0;
    };

    void set_active_request_count(size_t count);
    void set_current_batch_size(size_t batch_size);
    void set_tokens_generated_last_step(size_t tokens_generated);
    void set_tokens_per_second(double value);
    void log_request_completion(const RequestMetrics& metrics) const;
    static double duration_ms(PerfClock::duration duration);
    RequestMetrics* find_request(uint64_t request_id);
    std::optional<RequestMetrics> complete_request_locked(
        uint64_t request_id,
        PerfClock::time_point now,
        const std::string& terminal_reason);

    PerformanceInstrumentationConfig config_;
    std::ostream* log_stream_ = nullptr;
    uint64_t next_request_id_ = 1;
    std::unordered_map<uint64_t, RequestMetrics> requests_;
    ThroughputWindow throughput_window_;
    PerfClock::time_point last_periodic_log_time_{};

    mutable std::mutex state_mutex_;
    std::atomic<size_t> active_request_count_{0};
    std::atomic<size_t> current_batch_size_{0};
    std::atomic<size_t> tokens_generated_last_step_{0};
    std::atomic<uint64_t> total_tokens_generated_{0};
    std::atomic<uint64_t> completed_request_count_{0};
    std::atomic<uint64_t> decode_batch_size_total_{0};
    std::atomic<uint64_t> decode_step_count_{0};
    std::atomic<uint64_t> tokens_per_second_milli_{0};
    mutable std::mutex log_mutex_;
};

}  // namespace nano_llm_infer
