#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "generation.h"

namespace nano_llm_infer {

enum class BenchmarkMode {
    kOfflineThroughput,
    kOnlineServing,
};

struct BenchmarkRequestTemplate {
    std::string prompt;
    int max_tokens = 0;
};

struct BenchmarkConfig {
    BenchmarkMode mode = BenchmarkMode::kOfflineThroughput;
    size_t num_requests = 1;
    size_t max_concurrency = 1;
    double arrival_rate_qps = 1.0;
    uint64_t random_seed = 12345;
    int warmup_requests = 0;
    bool export_csv = false;
    std::string scheduler_policy = "round_robin";
    bool dynamic_batching_enabled = true;
    int max_active_requests = 1;
    int max_decode_batch_size = 0;
    int request_max_tokens = 16;
    std::vector<std::string> prompts;
};

struct BenchmarkRequestResult {
    size_t request_index = 0;
    size_t prompt_index = 0;
    uint64_t tokens_generated = 0;
    double latency_ms = 0.0;
    double ttft_ms = 0.0;
    double tpot_ms = 0.0;
    bool completed = false;
    std::string finish_reason;
    std::string error;
};

struct BenchmarkSummary {
    BenchmarkMode mode = BenchmarkMode::kOfflineThroughput;
    size_t num_requests = 0;
    size_t max_concurrency = 0;
    double avg_prompt_tokens = 0.0;
    double arrival_rate_qps = 0.0;
    uint64_t random_seed = 0;
    int warmup_requests = 0;
    std::string scheduler_policy;
    bool dynamic_batching_enabled = true;
    int max_active_requests = 0;
    int max_decode_batch_size = 0;
    int request_max_tokens = 0;
    double total_runtime_ms = 0.0;
    double throughput_tokens_per_sec = 0.0;
    double requests_per_sec = 0.0;
    double avg_batch_size = 0.0;
    double avg_latency_ms = 0.0;
    double p50_latency_ms = 0.0;
    double p95_latency_ms = 0.0;
    double avg_ttft_ms = 0.0;
    double p50_ttft_ms = 0.0;
    double p95_ttft_ms = 0.0;
    double avg_tpot_ms = 0.0;
    uint64_t total_tokens_generated = 0;
    size_t completed_requests = 0;
    std::vector<BenchmarkRequestResult> request_results;
};

struct BenchmarkRequestTrace {
    std::chrono::steady_clock::time_point submit_time{};
    std::chrono::steady_clock::time_point first_token_time{};
    std::chrono::steady_clock::time_point completion_time{};
    uint64_t tokens_generated = 0;
    bool saw_first_token = false;
    bool saw_terminal = false;
    std::string finish_reason;
    std::string error;

    void record_event(const GenerationEvent& event, std::chrono::steady_clock::time_point now);
    BenchmarkRequestResult finalize(size_t request_index, size_t prompt_index) const;
};

class ModelRuntime;

class BenchmarkRunner {
public:
    explicit BenchmarkRunner(ModelRuntime* runtime);

    BenchmarkSummary run(const BenchmarkConfig& config);

private:
    ModelRuntime* runtime_ = nullptr;
};

std::vector<BenchmarkRequestTemplate> default_benchmark_prompts();
std::vector<double> build_arrival_schedule_seconds(size_t num_requests, double arrival_rate_qps, uint64_t random_seed);
double benchmark_percentile(const std::vector<double>& values, double percentile);
int derive_max_decode_batch_size(const BenchmarkConfig& config);
std::string default_benchmark_csv_path();
void append_benchmark_summary_csv(const BenchmarkSummary& summary);
std::string format_benchmark_summary(const BenchmarkSummary& summary);
std::string benchmark_mode_to_string(BenchmarkMode mode);

}  // namespace nano_llm_infer
