#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "benchmark.h"
#include "generation.h"

#include "test_util.h"

#ifndef NANO_LLM_TEST_MODEL_PATH
#define NANO_LLM_TEST_MODEL_PATH "/root/autodl-tmp/repos/nano_llm_infer/model_file/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
#endif

using namespace nano_llm_infer;

namespace {

ModelRuntime* make_runtime_or_skip(std::unique_ptr<ModelRuntime>* runtime) {
    const std::string model_path = NANO_LLM_TEST_MODEL_PATH;
    if (model_path.empty()) {
        std::cerr << "[SKIP] unit_benchmark: NANO_LLM_TEST_MODEL is empty\n";
        return nullptr;
    }

    try {
        RuntimeConfig config;
        config.model_path = model_path;
        config.compute_threads = 1;
        config.max_seq_len = 256;
        config.max_new_tokens = 8;
        config.prefill_ctx_bytes = 256ull * 1024 * 1024;
        config.decode_ctx_bytes = 128ull * 1024 * 1024;
        *runtime = std::make_unique<ModelRuntime>(config);
        return runtime->get();
    } catch (const std::exception& e) {
        std::cerr << "[SKIP] unit_benchmark: failed to load model at '" << model_path
                  << "' (" << e.what() << ")\n";
        return nullptr;
    }
}

void test_percentile_helpers() {
    const std::vector<double> values = {10.0, 20.0, 30.0, 40.0};
    EXPECT_NEAR(benchmark_percentile(values, 50.0), 25.0, 1e-9);
    EXPECT_NEAR(benchmark_percentile(values, 95.0), 38.5, 1e-9);
    EXPECT_NEAR(benchmark_percentile(values, 0.0), 10.0, 1e-9);
    EXPECT_NEAR(benchmark_percentile({}, 50.0), 0.0, 1e-9);
}

void test_arrival_schedule_is_seeded_and_monotonic() {
    const std::vector<double> first = build_arrival_schedule_seconds(5, 2.0, 123);
    const std::vector<double> second = build_arrival_schedule_seconds(5, 2.0, 123);
    EXPECT_EQ(first.size(), static_cast<size_t>(5));
    EXPECT_EQ(first.size(), second.size());
    for (size_t i = 0; i < first.size(); ++i) {
        EXPECT_NEAR(first[i], second[i], 1e-12);
        if (i > 0) {
            EXPECT_TRUE(first[i] >= first[i - 1]);
        }
    }
}

void test_request_trace_ttft_and_tpot() {
    BenchmarkRequestTrace trace;
    const auto submit = std::chrono::steady_clock::now();
    trace.submit_time = submit;
    trace.record_event(GenerationEvent{}, submit + std::chrono::milliseconds(5));

    GenerationEvent token;
    token.text = "hello";
    trace.record_event(token, submit + std::chrono::milliseconds(12));
    trace.record_event(token, submit + std::chrono::milliseconds(20));

    GenerationEvent done;
    done.finish_reason = "stop";
    trace.record_event(done, submit + std::chrono::milliseconds(30));
    trace.completion_time = submit + std::chrono::milliseconds(30);

    const BenchmarkRequestResult result = trace.finalize(3, 1);
    EXPECT_EQ(result.request_index, static_cast<size_t>(3));
    EXPECT_EQ(result.prompt_index, static_cast<size_t>(1));
    EXPECT_EQ(result.tokens_generated, static_cast<uint64_t>(2));
    EXPECT_NEAR(result.ttft_ms, 12.0, 0.5);
    EXPECT_NEAR(result.latency_ms, 30.0, 0.5);
    EXPECT_NEAR(result.tpot_ms, 9.0, 0.5);
    EXPECT_TRUE(result.completed);
}

void test_request_trace_without_tokens_has_zero_tpot() {
    BenchmarkRequestTrace trace;
    const auto submit = std::chrono::steady_clock::now();
    trace.submit_time = submit;

    GenerationEvent error;
    error.error = "boom";
    trace.record_event(error, submit + std::chrono::milliseconds(7));
    trace.completion_time = submit + std::chrono::milliseconds(10);

    const BenchmarkRequestResult result = trace.finalize(0, 0);
    EXPECT_EQ(result.tokens_generated, static_cast<uint64_t>(0));
    EXPECT_NEAR(result.ttft_ms, 0.0, 1e-9);
    EXPECT_NEAR(result.tpot_ms, 0.0, 1e-9);
    EXPECT_TRUE(!result.completed);
}

void test_csv_export_writes_header_and_row() {
    const std::string path = default_benchmark_csv_path();
    std::remove(path.c_str());

    BenchmarkSummary summary;
    summary.mode = BenchmarkMode::kOfflineThroughput;
    summary.max_concurrency = 3;
    summary.throughput_tokens_per_sec = 12.5;
    summary.avg_ttft_ms = 9.0;
    summary.p95_latency_ms = 44.0;
    summary.scheduler_policy = "round_robin";
    summary.dynamic_batching_enabled = true;
    summary.max_active_requests = 4;
    summary.max_decode_batch_size = 2;
    summary.requests_per_sec = 3.5;
    summary.avg_batch_size = 1.25;
    summary.total_tokens_generated = 50;
    summary.total_runtime_ms = 4000.0;

    append_benchmark_summary_csv(summary);

    std::ifstream input(path);
    std::string header;
    std::string row;
    std::getline(input, header);
    std::getline(input, row);
    EXPECT_EQ(
        header,
        std::string(
            "mode,concurrency,throughput,avg_ttft,p95_latency,scheduler_policy,dynamic_batching,"
            "max_active,max_decode_batch_size,requests_per_sec,avg_batch_size,total_tokens,total_runtime_ms"));
    EXPECT_TRUE(row.find("offline_throughput,3,12.500000,9.000000,44.000000,round_robin,true,4,2,3.500000,1.250000,50,4000.000000") == 0);
}

void test_human_summary_formatting() {
    BenchmarkSummary summary;
    summary.num_requests = 100;
    summary.max_concurrency = 16;
    summary.avg_prompt_tokens = 256.0;
    summary.request_max_tokens = 128;
    summary.dynamic_batching_enabled = true;
    summary.throughput_tokens_per_sec = 142.5;
    summary.requests_per_sec = 8.2;
    summary.avg_ttft_ms = 118.0;
    summary.p95_ttft_ms = 240.0;
    summary.avg_latency_ms = 2800.0;
    summary.p95_latency_ms = 4100.0;
    summary.avg_batch_size = 11.2;
    summary.completed_requests = 100;

    const std::string formatted = format_benchmark_summary(summary);
    EXPECT_TRUE(formatted.find("==============================\nBenchmark Summary\n==============================\n\n") == 0);
    EXPECT_TRUE(formatted.find("Requests:             100\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("Concurrency:          16\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("Prompt Length:        256.0 tokens\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("Generation Length:    128\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("Dynamic Batching:     ON\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("Throughput:           142.5 tok/s\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("Request Throughput:   8.2 req/s\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("Avg TTFT:             118 ms\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("P95 TTFT:             240 ms\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("Avg Latency:          2.8 s\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("P95 Latency:          4.1 s\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("Avg Batch Size:       11.2\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("Completed Requests:   100\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("p50") == std::string::npos);
}

void test_human_summary_uses_ms_for_short_latencies_and_off_toggle() {
    BenchmarkSummary summary;
    summary.num_requests = 2;
    summary.max_concurrency = 1;
    summary.avg_prompt_tokens = 8.0;
    summary.request_max_tokens = 4;
    summary.dynamic_batching_enabled = false;
    summary.avg_ttft_ms = 0.4;
    summary.p95_ttft_ms = 1.0;
    summary.avg_latency_ms = 900.0;
    summary.p95_latency_ms = 999.0;

    const std::string formatted = format_benchmark_summary(summary);
    EXPECT_TRUE(formatted.find("Dynamic Batching:     OFF\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("Avg TTFT:             0.4 ms\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("P95 TTFT:             1 ms\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("Avg Latency:          900 ms\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("P95 Latency:          999 ms\n") != std::string::npos);
}

void test_dynamic_batching_off_maps_to_batch_size_one() {
    BenchmarkConfig config;
    config.dynamic_batching_enabled = false;
    config.max_active_requests = 8;
    config.max_decode_batch_size = 4;
    EXPECT_EQ(derive_max_decode_batch_size(config), 1);
}

void test_average_batch_size_and_offline_runner(ModelRuntime* runtime) {
    BenchmarkConfig config;
    config.mode = BenchmarkMode::kOfflineThroughput;
    config.num_requests = 2;
    config.max_concurrency = 2;
    config.max_active_requests = 2;
    config.max_decode_batch_size = 2;
    config.request_max_tokens = 2;
    config.prompts = {"Hello.", "How are you?"};

    BenchmarkRunner runner(runtime);
    const BenchmarkSummary summary = runner.run(config);
    EXPECT_EQ(summary.request_results.size(), static_cast<size_t>(2));
    EXPECT_TRUE(summary.total_runtime_ms > 0.0);
    EXPECT_TRUE(summary.completed_requests >= 1);
    EXPECT_TRUE(summary.avg_batch_size >= 0.0);
}

void test_online_runner_is_deterministic_enough(ModelRuntime* runtime) {
    BenchmarkConfig config;
    config.mode = BenchmarkMode::kOnlineServing;
    config.num_requests = 3;
    config.max_concurrency = 1;
    config.arrival_rate_qps = 50.0;
    config.random_seed = 7;
    config.max_active_requests = 1;
    config.max_decode_batch_size = 1;
    config.request_max_tokens = 1;
    config.prompts = {"Hi", "Hello", "Hey"};

    BenchmarkRunner runner(runtime);
    const BenchmarkSummary summary = runner.run(config);
    EXPECT_EQ(summary.request_results.size(), static_cast<size_t>(3));
    EXPECT_TRUE(summary.total_runtime_ms > 0.0);
    for (size_t i = 0; i < summary.request_results.size(); ++i) {
        EXPECT_EQ(summary.request_results[i].request_index, i);
    }
}

}  // namespace

int main() {
    test_percentile_helpers();
    test_arrival_schedule_is_seeded_and_monotonic();
    test_request_trace_ttft_and_tpot();
    test_request_trace_without_tokens_has_zero_tpot();
    test_csv_export_writes_header_and_row();
    test_human_summary_formatting();
    test_human_summary_uses_ms_for_short_latencies_and_off_toggle();
    test_dynamic_batching_off_maps_to_batch_size_one();

    std::unique_ptr<ModelRuntime> runtime;
    ModelRuntime* runtime_ptr = make_runtime_or_skip(&runtime);
    if (runtime_ptr != nullptr) {
        test_average_batch_size_and_offline_runner(runtime_ptr);
        test_online_runner_is_deterministic_enough(runtime_ptr);
    }

    return test_util::finalize();
}
