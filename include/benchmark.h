#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "generation.h"

namespace nano_llm_infer {

enum class BenchmarkMode {
    kOfflineThroughput,
    kOnlineServing,
};

enum class BenchmarkPreset {
    kChat,
    kThroughput,
    kLatency,
    kLongContext,
};

enum class PromptCategory {
    Chat,
    Code,
    Reasoning,
};

enum class PromptLengthClass {
    Short,
    Medium,
    Long,
};

class LengthSampler {
public:
    explicit LengthSampler(uint64_t seed);

    int sample(PromptLengthClass length_class);

private:
    std::mt19937 rng_;
};

class Tokenizer;

class PromptGenerator {
public:
    explicit PromptGenerator(Tokenizer* tokenizer, uint64_t seed = 0);

    std::string generate(int target_tokens, PromptCategory category);

private:
    Tokenizer* tokenizer_ = nullptr;
    std::mt19937 rng_;
};

struct BenchmarkRequestTemplate {
    std::string prompt;
    int max_tokens = 0;
    uint64_t prompt_tokens = 0;
};

struct BenchmarkTrafficConfig {
    // offline_throughput submits as fast as concurrency allows; online_serving uses seeded Poisson arrivals.
    BenchmarkMode mode = BenchmarkMode::kOfflineThroughput;
    // Measured requests, excluding warmup.
    size_t request_count = 1;
    // Maximum client-side in-flight requests and service active-request limit for the benchmark.
    size_t concurrency = 1;
    // Online arrival rate in queries per second. 0.0 means unlimited and is valid only for offline mode.
    double request_rate_qps = 0.0;
    // Requests run before measurement.
    int warmup_requests = 0;
    // Seed for deterministic online arrivals.
    uint64_t random_seed = 12345;
};

struct BenchmarkWorkloadConfig {
    // Target synthetic prompt length when prompts is empty.
    int input_tokens = 128;
    // Per-request generation cap.
    int output_tokens = 16;
    // Text stop-sequence trimming. Disable for sustained decode benchmarks.
    bool enable_stop_sequences = true;
    // EOS handling. Disable EOS stops for sustained decode benchmarks.
    bool ignore_eos = false;
    // Synthetic-only controls used when prompts is empty.
    size_t synthetic_prompt_count = 1;
    bool synthetic_mixed_lengths = false;
    bool synthetic_mixed_categories = false;
    bool synthetic_random_output_tokens = false;
    PromptCategory synthetic_category = PromptCategory::Chat;
    PromptLengthClass synthetic_length_class = PromptLengthClass::Medium;
    int min_output_tokens = 0;
    int max_output_tokens = 0;
    // Optional explicit prompts; if set, synthetic prompt generation is skipped.
    std::vector<std::string> prompts;
};

struct BenchmarkServingConfig {
    // Maximum decode batch size. 0 means auto from concurrency; 1 disables decode batching.
    int decode_batch_size = 0;
};

struct BenchmarkOutputConfig {
    // Append summary metrics to the default timestamped CSV.
    bool export_csv = false;
};

struct BenchmarkConfig {
    BenchmarkPreset preset = BenchmarkPreset::kChat;
    BenchmarkTrafficConfig traffic;
    BenchmarkWorkloadConfig workload;
    BenchmarkServingConfig serving;
    BenchmarkOutputConfig output;
};

struct BenchmarkRequestResult {
    size_t request_index = 0;
    size_t prompt_index = 0;
    uint64_t prompt_tokens = 0;
    uint64_t tokens_generated = 0;
    double latency_ms = 0.0;
    double ttft_ms = 0.0;
    double queue_ms = 0.0;
    double service_ttft_ms = 0.0;
    double tpot_ms = 0.0;
    bool has_latency = false;
    bool has_ttft = false;
    bool has_queue = false;
    bool has_service_ttft = false;
    bool has_tpot = false;
    bool completed = false;
    std::string finish_reason;
    std::string error;
};

struct BenchmarkSummary {
    BenchmarkPreset preset = BenchmarkPreset::kChat;
    BenchmarkMode mode = BenchmarkMode::kOfflineThroughput;
    size_t request_count = 0;
    size_t concurrency = 0;
    double request_rate_qps = 0.0;
    uint64_t random_seed = 0;
    int warmup_requests = 0;
    int compute_threads = 0;
    int context_tokens = 0;
    double avg_input_tokens = 0.0;
    int output_tokens = 0;
    int decode_batch_size = 0;
    double total_runtime_ms = 0.0;
    double throughput_tokens_per_sec = 0.0;
    double prompt_tokens_per_sec = 0.0;
    double output_tokens_per_sec = 0.0;
    double total_model_tokens_per_sec = 0.0;
    double requests_per_sec = 0.0;
    double avg_batch_size = 0.0;
    double avg_latency_ms = 0.0;
    double p50_latency_ms = 0.0;
    double p95_latency_ms = 0.0;
    double avg_ttft_ms = 0.0;
    double p50_ttft_ms = 0.0;
    double p95_ttft_ms = 0.0;
    double avg_queue_ms = 0.0;
    double p50_queue_ms = 0.0;
    double p95_queue_ms = 0.0;
    double avg_service_ttft_ms = 0.0;
    double p50_service_ttft_ms = 0.0;
    double p95_service_ttft_ms = 0.0;
    double avg_tpot_ms = 0.0;
    bool gpu_monitoring_available = false;
    double gpu_util_avg_percent = 0.0;
    uint32_t gpu_util_peak_percent = 0;
    double gpu_memory_bw_util_avg_percent = 0.0;
    uint32_t gpu_memory_bw_util_peak_percent = 0;
    double gpu_vram_avg_percent = 0.0;
    double gpu_vram_peak_percent = 0.0;
    uint64_t total_prompt_tokens = 0;
    uint64_t total_tokens_generated = 0;
    uint64_t total_model_tokens = 0;
    double avg_output_tokens = 0.0;
    uint64_t length_limited_requests = 0;
    uint64_t early_stopped_requests = 0;
    std::map<std::string, uint64_t> finish_reason_counts;
    size_t completed_requests = 0;
    std::vector<BenchmarkRequestResult> request_results;
};

struct BenchmarkRequestTrace {
    std::chrono::steady_clock::time_point submit_time{};
    std::chrono::steady_clock::time_point admit_time{};
    std::chrono::steady_clock::time_point first_token_time{};
    std::chrono::steady_clock::time_point last_token_time{};
    std::chrono::steady_clock::time_point completion_time{};
    std::chrono::steady_clock::duration inter_token_duration{};
    uint64_t tokens_generated = 0;
    uint64_t inter_token_interval_count = 0;
    bool saw_first_token = false;
    bool saw_terminal = false;
    std::string finish_reason;
    std::string error;

    void record_event(const GenerationEvent& event, std::chrono::steady_clock::time_point now);
    BenchmarkRequestResult finalize(size_t request_index, size_t prompt_index, uint64_t prompt_tokens = 0) const;
};

class ModelRuntime;

class BenchmarkRunner {
public:
    explicit BenchmarkRunner(ModelRuntime* runtime);

    BenchmarkSummary run(const BenchmarkConfig& config);

private:
    ModelRuntime* runtime_ = nullptr;
};

BenchmarkConfig make_preset(BenchmarkPreset preset);
std::string benchmark_preset_to_string(BenchmarkPreset preset);
BenchmarkPreset parse_benchmark_preset(const std::string& value);
std::vector<double> build_arrival_schedule_seconds(size_t request_count, double request_rate_qps, uint64_t random_seed);
double benchmark_percentile(const std::vector<double>& values, double percentile);
double effective_request_rate_qps(const BenchmarkConfig& config);
int effective_decode_batch_size(const BenchmarkConfig& config);
int effective_active_request_limit(const BenchmarkConfig& config);
std::string default_benchmark_csv_path();
void append_benchmark_summary_csv(const BenchmarkSummary& summary);
std::string format_benchmark_summary(const BenchmarkSummary& summary);
std::string benchmark_mode_to_string(BenchmarkMode mode);

}  // namespace nano_llm_infer
