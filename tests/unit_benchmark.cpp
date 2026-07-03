#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
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
        config.model.path = model_path;
        config.model.compute_threads = 1;
        config.limits.context_tokens = 256;
        config.limits.default_max_new_tokens = 8;
        config.memory.prefill_ctx_bytes = 256ull * 1024 * 1024;
        config.memory.decode_ctx_bytes = 128ull * 1024 * 1024;
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

void test_length_sampler_ranges() {
    LengthSampler sampler(123);
    for (int i = 0; i < 64; ++i) {
        const int short_length = sampler.sample(PromptLengthClass::Short);
        EXPECT_TRUE(short_length >= 32);
        EXPECT_TRUE(short_length <= 128);

        const int medium_length = sampler.sample(PromptLengthClass::Medium);
        EXPECT_TRUE(medium_length >= 256);
        EXPECT_TRUE(medium_length <= 512);

        const int long_length = sampler.sample(PromptLengthClass::Long);
        EXPECT_TRUE(long_length >= 1024);
        EXPECT_TRUE(long_length <= 2048);
    }
}

void test_preset_defaults() {
    const BenchmarkConfig chat = make_preset(BenchmarkPreset::kChat);
    EXPECT_EQ(chat.preset, BenchmarkPreset::kChat);
    EXPECT_EQ(chat.traffic.mode, BenchmarkMode::kOnlineServing);
    EXPECT_EQ(chat.traffic.request_count, static_cast<size_t>(40));
    EXPECT_EQ(chat.traffic.concurrency, static_cast<size_t>(4));
    EXPECT_EQ(effective_active_request_limit(chat), 4);
    EXPECT_EQ(chat.workload.input_tokens, 256);
    EXPECT_EQ(chat.workload.output_tokens, 128);
    EXPECT_TRUE(chat.workload.enable_stop_sequences);
    EXPECT_EQ(chat.workload.synthetic_prompt_count, static_cast<size_t>(8));
    EXPECT_EQ(chat.workload.synthetic_category, PromptCategory::Chat);
    EXPECT_EQ(chat.workload.synthetic_length_class, PromptLengthClass::Medium);
    EXPECT_TRUE(!chat.workload.synthetic_mixed_lengths);
    EXPECT_TRUE(!chat.workload.synthetic_mixed_categories);
    EXPECT_TRUE(!chat.workload.synthetic_random_output_tokens);
    EXPECT_NEAR(chat.traffic.request_rate_qps, 20.0, 1e-9);
    EXPECT_EQ(chat.traffic.warmup_requests, 1);
    EXPECT_EQ(chat.serving.decode_batch_size, 0);
    EXPECT_EQ(effective_decode_batch_size(chat), static_cast<int>(chat.traffic.concurrency));

    const BenchmarkConfig throughput = make_preset(BenchmarkPreset::kThroughput);
    EXPECT_EQ(throughput.traffic.mode, BenchmarkMode::kOfflineThroughput);
    EXPECT_EQ(throughput.traffic.request_count, static_cast<size_t>(96));
    EXPECT_EQ(throughput.traffic.concurrency, static_cast<size_t>(16));
    EXPECT_EQ(effective_active_request_limit(throughput), 16);
    EXPECT_EQ(throughput.workload.input_tokens, 128);
    EXPECT_EQ(throughput.workload.output_tokens, 128);
    EXPECT_TRUE(!throughput.workload.enable_stop_sequences);
    EXPECT_TRUE(throughput.workload.ignore_eos);
    EXPECT_EQ(throughput.workload.synthetic_prompt_count, throughput.traffic.request_count);
    EXPECT_TRUE(!throughput.workload.synthetic_mixed_lengths);
    EXPECT_TRUE(throughput.workload.synthetic_mixed_categories);
    EXPECT_TRUE(throughput.workload.synthetic_random_output_tokens);
    EXPECT_EQ(throughput.workload.min_output_tokens, 96);
    EXPECT_EQ(throughput.workload.max_output_tokens, 160);
    EXPECT_NEAR(throughput.traffic.request_rate_qps, 0.0, 1e-9);
    EXPECT_EQ(throughput.traffic.warmup_requests, 1);
    EXPECT_EQ(throughput.serving.decode_batch_size, 16);
    EXPECT_EQ(effective_decode_batch_size(throughput), static_cast<int>(throughput.traffic.concurrency));

    const BenchmarkConfig latency = make_preset(BenchmarkPreset::kLatency);
    EXPECT_EQ(latency.traffic.mode, BenchmarkMode::kOfflineThroughput);
    EXPECT_EQ(latency.traffic.request_count, static_cast<size_t>(8));
    EXPECT_EQ(latency.traffic.concurrency, static_cast<size_t>(1));
    EXPECT_EQ(effective_active_request_limit(latency), 1);
    EXPECT_EQ(latency.workload.input_tokens, 32);
    EXPECT_EQ(latency.workload.output_tokens, 16);
    EXPECT_TRUE(latency.workload.enable_stop_sequences);
    EXPECT_EQ(latency.workload.synthetic_prompt_count, static_cast<size_t>(1));
    EXPECT_EQ(latency.workload.synthetic_category, PromptCategory::Chat);
    EXPECT_EQ(latency.workload.synthetic_length_class, PromptLengthClass::Short);
    EXPECT_TRUE(!latency.workload.synthetic_mixed_lengths);
    EXPECT_TRUE(!latency.workload.synthetic_mixed_categories);
    EXPECT_TRUE(!latency.workload.synthetic_random_output_tokens);
    EXPECT_NEAR(latency.traffic.request_rate_qps, 0.0, 1e-9);
    EXPECT_EQ(latency.traffic.warmup_requests, 1);
    EXPECT_EQ(latency.serving.decode_batch_size, 0);
    EXPECT_EQ(effective_decode_batch_size(latency), static_cast<int>(latency.traffic.concurrency));

    const BenchmarkConfig long_context = make_preset(BenchmarkPreset::kLongContext);
    EXPECT_EQ(long_context.traffic.mode, BenchmarkMode::kOfflineThroughput);
    EXPECT_EQ(long_context.traffic.request_count, static_cast<size_t>(8));
    EXPECT_EQ(long_context.traffic.concurrency, static_cast<size_t>(1));
    EXPECT_EQ(effective_active_request_limit(long_context), 1);
    EXPECT_EQ(long_context.workload.input_tokens, 800);
    EXPECT_EQ(long_context.workload.output_tokens, 32);
    EXPECT_TRUE(long_context.workload.enable_stop_sequences);
    EXPECT_EQ(long_context.workload.synthetic_prompt_count, static_cast<size_t>(4));
    EXPECT_EQ(long_context.workload.synthetic_category, PromptCategory::Reasoning);
    EXPECT_EQ(long_context.workload.synthetic_length_class, PromptLengthClass::Long);
    EXPECT_TRUE(!long_context.workload.synthetic_mixed_lengths);
    EXPECT_TRUE(!long_context.workload.synthetic_mixed_categories);
    EXPECT_TRUE(!long_context.workload.synthetic_random_output_tokens);
    EXPECT_NEAR(long_context.traffic.request_rate_qps, 0.0, 1e-9);
    EXPECT_EQ(long_context.traffic.warmup_requests, 1);
    EXPECT_EQ(long_context.serving.decode_batch_size, 0);
    EXPECT_EQ(effective_decode_batch_size(long_context), static_cast<int>(long_context.traffic.concurrency));
}

void test_preset_parse_and_format() {
    EXPECT_EQ(benchmark_preset_to_string(BenchmarkPreset::kChat), std::string("chat"));
    EXPECT_EQ(benchmark_preset_to_string(BenchmarkPreset::kThroughput), std::string("throughput"));
    EXPECT_EQ(benchmark_preset_to_string(BenchmarkPreset::kLatency), std::string("latency"));
    EXPECT_EQ(benchmark_preset_to_string(BenchmarkPreset::kLongContext), std::string("long_context"));
    EXPECT_EQ(parse_benchmark_preset("chat"), BenchmarkPreset::kChat);
    EXPECT_EQ(parse_benchmark_preset("throughput"), BenchmarkPreset::kThroughput);
    EXPECT_EQ(parse_benchmark_preset("latency"), BenchmarkPreset::kLatency);
    EXPECT_EQ(parse_benchmark_preset("long_context"), BenchmarkPreset::kLongContext);

    bool threw = false;
    try {
        (void)parse_benchmark_preset("invalid");
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    EXPECT_TRUE(threw);
}

void test_request_trace_ttft_and_tpot() {
    BenchmarkRequestTrace trace;
    const auto submit = std::chrono::steady_clock::now();
    trace.submit_time = submit;
    trace.admit_time = submit + std::chrono::milliseconds(4);
    trace.record_event(GenerationEvent{}, submit + std::chrono::milliseconds(5));

    GenerationEvent token;
    token.text = "hello";
    token.token_generated = true;
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
    EXPECT_TRUE(result.has_ttft);
    EXPECT_NEAR(result.queue_ms, 4.0, 0.5);
    EXPECT_TRUE(result.has_queue);
    EXPECT_NEAR(result.service_ttft_ms, 8.0, 0.5);
    EXPECT_TRUE(result.has_service_ttft);
    EXPECT_NEAR(result.latency_ms, 30.0, 0.5);
    EXPECT_TRUE(result.has_latency);
    EXPECT_NEAR(result.tpot_ms, 8.0, 0.5);
    EXPECT_TRUE(result.has_tpot);
    EXPECT_TRUE(result.completed);
}

void test_request_trace_terminal_event_controls_latency() {
    BenchmarkRequestTrace trace;
    const auto submit = std::chrono::steady_clock::now();
    trace.submit_time = submit;

    GenerationEvent token;
    token.text = "hello";
    token.token_generated = true;
    trace.record_event(token, submit + std::chrono::milliseconds(12));

    GenerationEvent done;
    done.finish_reason = "stop";
    trace.record_event(done, submit + std::chrono::milliseconds(30));

    const BenchmarkRequestResult result = trace.finalize(1, 0);
    EXPECT_TRUE(result.has_latency);
    EXPECT_NEAR(result.latency_ms, 30.0, 0.5);
    EXPECT_TRUE(result.completed);
}

void test_request_trace_zero_queue_delay_is_present() {
    BenchmarkRequestTrace trace;
    const auto submit = std::chrono::steady_clock::now();
    trace.submit_time = submit;
    trace.admit_time = submit;

    GenerationEvent token;
    token.text = "hello";
    token.token_generated = true;
    trace.record_event(token, submit);

    GenerationEvent done;
    done.finish_reason = "stop";
    trace.record_event(done, submit);

    const BenchmarkRequestResult result = trace.finalize(1, 0);
    EXPECT_TRUE(result.has_ttft);
    EXPECT_TRUE(result.has_queue);
    EXPECT_TRUE(result.has_service_ttft);
    EXPECT_TRUE(result.has_latency);
    EXPECT_NEAR(result.ttft_ms, 0.0, 1e-9);
    EXPECT_NEAR(result.queue_ms, 0.0, 1e-9);
    EXPECT_NEAR(result.service_ttft_ms, 0.0, 1e-9);
    EXPECT_NEAR(result.latency_ms, 0.0, 1e-9);
}

void test_request_trace_single_token_has_zero_tpot() {
    BenchmarkRequestTrace trace;
    const auto submit = std::chrono::steady_clock::now();
    trace.submit_time = submit;

    GenerationEvent token;
    token.text = "hello";
    token.token_generated = true;
    trace.record_event(token, submit + std::chrono::milliseconds(12));

    GenerationEvent done;
    done.finish_reason = "stop";
    trace.record_event(done, submit + std::chrono::milliseconds(30));
    trace.completion_time = submit + std::chrono::milliseconds(30);

    const BenchmarkRequestResult result = trace.finalize(1, 0);
    EXPECT_EQ(result.tokens_generated, static_cast<uint64_t>(1));
    EXPECT_NEAR(result.ttft_ms, 12.0, 0.5);
    EXPECT_NEAR(result.tpot_ms, 0.0, 1e-9);
    EXPECT_TRUE(!result.has_tpot);
    EXPECT_TRUE(result.completed);
}

void test_request_trace_counts_empty_decoded_tokens() {
    BenchmarkRequestTrace trace;
    const auto submit = std::chrono::steady_clock::now();
    trace.submit_time = submit;

    GenerationEvent token;
    token.token_generated = true;
    trace.record_event(token, submit + std::chrono::milliseconds(12));

    GenerationEvent done;
    done.finish_reason = "length";
    trace.record_event(done, submit + std::chrono::milliseconds(20));

    const BenchmarkRequestResult result = trace.finalize(1, 0);
    EXPECT_EQ(result.tokens_generated, static_cast<uint64_t>(1));
    EXPECT_TRUE(result.has_ttft);
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
    EXPECT_TRUE(!result.has_ttft);
    EXPECT_TRUE(!result.has_tpot);
    EXPECT_TRUE(!result.completed);
}

void test_csv_export_writes_header_and_row() {
    const std::string path = default_benchmark_csv_path();
    std::remove(path.c_str());

    BenchmarkSummary summary;
    summary.preset = BenchmarkPreset::kChat;
    summary.mode = BenchmarkMode::kOfflineThroughput;
    summary.request_count = 7;
    summary.concurrency = 3;
    summary.request_rate_qps = 2.5;
    summary.random_seed = 99;
    summary.warmup_requests = 1;
    summary.compute_threads = 6;
    summary.context_tokens = 1024;
    summary.avg_input_tokens = 11.25;
    summary.output_tokens = 8;
    summary.decode_batch_size = 2;
    summary.throughput_tokens_per_sec = 12.5;
    summary.output_tokens_per_sec = 12.5;
    summary.prompt_tokens_per_sec = 19.25;
    summary.total_model_tokens_per_sec = 31.75;
    summary.requests_per_sec = 3.5;
    summary.avg_ttft_ms = 9.0;
    summary.avg_tpot_ms = 4.5;
    summary.p95_ttft_ms = 19.0;
    summary.avg_latency_ms = 22.0;
    summary.p95_latency_ms = 44.0;
    summary.avg_queue_ms = 1.0;
    summary.p50_queue_ms = 0.5;
    summary.p95_queue_ms = 2.0;
    summary.avg_service_ttft_ms = 8.0;
    summary.p50_service_ttft_ms = 7.0;
    summary.p95_service_ttft_ms = 18.0;
    summary.avg_batch_size = 1.25;
    summary.total_prompt_tokens = 77;
    summary.total_tokens_generated = 50;
    summary.total_model_tokens = 127;
    summary.total_runtime_ms = 4000.0;
    summary.completed_requests = 7;
    summary.gpu_monitoring_available = true;
    summary.gpu_util_avg_percent = 72.0;
    summary.gpu_util_peak_percent = 96;
    summary.gpu_memory_bw_util_avg_percent = 61.0;
    summary.gpu_memory_bw_util_peak_percent = 84;
    summary.gpu_vram_avg_percent = 16.75;
    summary.gpu_vram_peak_percent = 17.25;

    append_benchmark_summary_csv(summary);

    std::ifstream input(path);
    std::string header;
    std::string row;
    std::getline(input, header);
    std::getline(input, row);
    EXPECT_EQ(
        header,
        std::string(
            "preset,mode,request_count,concurrency,request_rate_qps,random_seed,warmup_requests,"
            "compute_threads,context_tokens,avg_input_tokens,output_tokens,decode_batch_size,"
            "requests_per_sec,avg_ttft_ms,avg_tpot_ms,"
            "avg_latency_ms,avg_batch_size,total_tokens,total_runtime_ms,completed_requests,"
            "total_prompt_tokens,total_model_tokens,prompt_tokens_per_sec,output_tokens_per_sec,"
            "total_model_tokens_per_sec,gpu_monitoring_available,gpu_vram_peak_percent"));
    EXPECT_TRUE(row.find("chat,offline_throughput,7,3,2.50,99,1,6,1024,11.25,8,2,3.50,9.00,4.50,22.00,1.25,50,4000.00,7,77,127,19.25,12.50,31.75,1,17.25") == 0);
}

void test_csv_export_preserves_unavailable_gpu_monitoring() {
    const std::string path = default_benchmark_csv_path();
    std::remove(path.c_str());

    BenchmarkSummary summary;
    summary.preset = BenchmarkPreset::kLatency;
    summary.mode = BenchmarkMode::kOnlineServing;
    summary.request_count = 1;
    summary.concurrency = 1;
    summary.gpu_monitoring_available = false;

    append_benchmark_summary_csv(summary);

    std::ifstream input(path);
    std::string header;
    std::string row;
    std::getline(input, header);
    std::getline(input, row);
    EXPECT_TRUE(header.find("gpu_monitoring_available,gpu_vram_peak_percent") != std::string::npos);
    EXPECT_TRUE(header.find("gpu_util_avg_percent") == std::string::npos);
    EXPECT_TRUE(row.find("latency,online_serving,1,1,0.00,0,0,0,0,0.00,0,0,0.00,0.00,0.00,0.00,0.00,0,0.00,0,0,0,0.00,0.00,0.00,0,") == 0);
}

void test_human_summary_formatting() {
    BenchmarkSummary summary;
    summary.preset = BenchmarkPreset::kChat;
    summary.request_count = 100;
    summary.mode = BenchmarkMode::kOnlineServing;
    summary.concurrency = 16;
    summary.request_rate_qps = 20.0;
    summary.random_seed = 123;
    summary.warmup_requests = 4;
    summary.compute_threads = 8;
    summary.context_tokens = 2048;
    summary.avg_input_tokens = 256.0;
    summary.output_tokens = 128;
    summary.decode_batch_size = 16;
    summary.throughput_tokens_per_sec = 142.5;
    summary.output_tokens_per_sec = 142.5;
    summary.prompt_tokens_per_sec = 284.0;
    summary.total_model_tokens_per_sec = 426.5;
    summary.requests_per_sec = 8.2;
    summary.total_prompt_tokens = 25600;
    summary.total_tokens_generated = 12800;
    summary.total_model_tokens = 38400;
    summary.avg_output_tokens = 128.0;
    summary.length_limited_requests = 100;
    summary.early_stopped_requests = 0;
    summary.finish_reason_counts = {{"length", 100}};
    summary.avg_ttft_ms = 118.0;
    summary.avg_tpot_ms = 22.0;
    summary.p95_ttft_ms = 240.0;
    summary.avg_queue_ms = 12.0;
    summary.p95_queue_ms = 21.0;
    summary.avg_service_ttft_ms = 106.0;
    summary.p95_service_ttft_ms = 219.0;
    summary.avg_latency_ms = 2800.0;
    summary.p95_latency_ms = 4100.0;
    summary.avg_batch_size = 11.2;
    summary.total_runtime_ms = 12345.0;
    summary.completed_requests = 100;
    summary.gpu_monitoring_available = true;
    summary.gpu_util_avg_percent = 72.0;
    summary.gpu_util_peak_percent = 96;
    summary.gpu_memory_bw_util_avg_percent = 61.0;
    summary.gpu_memory_bw_util_peak_percent = 84;
    summary.gpu_vram_avg_percent = 16.75;
    summary.gpu_vram_peak_percent = 17.25;

    const std::string formatted = format_benchmark_summary(summary);
    EXPECT_TRUE(formatted.find("==============================\nBenchmark Summary\n==============================\n\n") == 0);
    EXPECT_TRUE(formatted.find("Preset:               chat\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("Requests:             100\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("Mode:                 online_serving\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("Concurrency:          16\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("Request Rate:         20.00 req/s\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("Random Seed:          123\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("Warmup Requests:      4\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("Threads:              8\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("Context Tokens:       2048\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("Input Tokens:         256.0 tokens\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("Max Output Tokens:    128\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("Decode Batch Size:    16\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("Scheduler Policy") == std::string::npos);
    EXPECT_TRUE(formatted.find("Dynamic Batching") == std::string::npos);
    EXPECT_TRUE(formatted.find("Max Active") == std::string::npos);
    EXPECT_TRUE(formatted.find("Throughput:           142.5 tok/s\n") == std::string::npos);
    EXPECT_TRUE(formatted.find("Output Throughput:    142.5 tok/s\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("Prompt Throughput:    284.0 tok/s\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("Total Model Throughput: 426.5 tok/s\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("Request Throughput:   8.200 req/s\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("Total Prompt Tokens:  25600\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("Total Output Tokens:  12800\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("Total Model Tokens:   38400\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("Avg Output Tokens:    128.0 tokens\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("Length-Limited Requests: 100\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("Early-Stop Requests:  0\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("Finish Reasons:       length=100\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("Total Runtime:        12.3 s\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("Avg TTFT:             118 ms\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("Avg TPOT:             22 ms\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("P95 TTFT") == std::string::npos);
    EXPECT_TRUE(formatted.find("Avg Queue Delay") == std::string::npos);
    EXPECT_TRUE(formatted.find("P95 Queue Delay") == std::string::npos);
    EXPECT_TRUE(formatted.find("Avg Service TTFT") == std::string::npos);
    EXPECT_TRUE(formatted.find("P95 Service TTFT") == std::string::npos);
    EXPECT_TRUE(formatted.find("Avg Latency:          2.8 s\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("P95 Latency") == std::string::npos);
    EXPECT_TRUE(formatted.find("Avg Batch Size:       11.2\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("GPU Util Avg") == std::string::npos);
    EXPECT_TRUE(formatted.find("GPU Util Peak") == std::string::npos);
    EXPECT_TRUE(formatted.find("GPU Memory BW Util Avg") == std::string::npos);
    EXPECT_TRUE(formatted.find("GPU Memory BW Util Peak") == std::string::npos);
    EXPECT_TRUE(formatted.find("GPU VRAM Avg") == std::string::npos);
    EXPECT_TRUE(formatted.find("GPU VRAM Peak:        17.2%\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("Completed Requests:   100\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("p50") == std::string::npos);
}

void test_human_summary_uses_ms_for_short_latencies_and_off_toggle() {
    BenchmarkSummary summary;
    summary.preset = BenchmarkPreset::kThroughput;
    summary.request_count = 2;
    summary.concurrency = 1;
    summary.avg_input_tokens = 8.0;
    summary.output_tokens = 4;
    summary.avg_ttft_ms = 0.4;
    summary.avg_tpot_ms = 0.6;
    summary.p95_ttft_ms = 1.0;
    summary.avg_latency_ms = 900.0;
    summary.p95_latency_ms = 999.0;

    const std::string formatted = format_benchmark_summary(summary);
    EXPECT_TRUE(formatted.find("Request Rate:         unlimited\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("Avg TTFT:             0.4 ms\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("Avg TPOT:             0.6 ms\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("P95 TTFT") == std::string::npos);
    EXPECT_TRUE(formatted.find("Avg Latency:          900 ms\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("P95 Latency") == std::string::npos);
}

void test_human_summary_reports_unavailable_gpu_monitoring() {
    BenchmarkSummary summary;
    summary.preset = BenchmarkPreset::kLatency;
    summary.request_count = 1;
    summary.concurrency = 1;
    summary.avg_input_tokens = 8.0;
    summary.output_tokens = 1;
    summary.gpu_monitoring_available = false;

    const std::string formatted = format_benchmark_summary(summary);
    EXPECT_TRUE(formatted.find("GPU Monitoring:       unavailable\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("GPU Util Avg") == std::string::npos);
    EXPECT_TRUE(formatted.find("GPU VRAM Peak") == std::string::npos);
}

void test_human_summary_reports_early_stop_behavior() {
    BenchmarkSummary summary;
    summary.preset = BenchmarkPreset::kThroughput;
    summary.request_count = 4;
    summary.concurrency = 2;
    summary.output_tokens = 128;
    summary.total_tokens_generated = 240;
    summary.avg_output_tokens = 60.0;
    summary.completed_requests = 4;
    summary.length_limited_requests = 1;
    summary.early_stopped_requests = 3;
    summary.finish_reason_counts = {{"length", 1}, {"stop", 3}};

    const std::string formatted = format_benchmark_summary(summary);
    EXPECT_TRUE(formatted.find("Max Output Tokens:    128\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("Total Output Tokens:  240\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("Avg Output Tokens:    60.0 tokens\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("Length-Limited Requests: 1\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("Early-Stop Requests:  3\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("Finish Reasons:       length=1, stop=3\n") != std::string::npos);
}

void test_effective_decode_batch_size() {
    BenchmarkConfig config;
    config.traffic.concurrency = 8;
    config.serving.decode_batch_size = 0;
    EXPECT_EQ(effective_decode_batch_size(config), 8);
    config.serving.decode_batch_size = 4;
    EXPECT_EQ(effective_decode_batch_size(config), 4);
    config.serving.decode_batch_size = 1;
    EXPECT_EQ(effective_decode_batch_size(config), 1);
}

void test_average_batch_size_and_offline_runner(ModelRuntime* runtime) {
    BenchmarkConfig config;
    config.traffic.mode = BenchmarkMode::kOfflineThroughput;
    config.traffic.request_count = 2;
    config.traffic.concurrency = 2;
    config.serving.decode_batch_size = 2;
    config.workload.output_tokens = 2;
    config.workload.prompts = {"Hello.", "How are you?"};

    GenerationRequest first_request;
    first_request.messages.push_back({"user", config.workload.prompts[0]});
    first_request.options.max_tokens = config.workload.output_tokens;
    GenerationRequest second_request;
    second_request.messages.push_back({"user", config.workload.prompts[1]});
    second_request.options.max_tokens = config.workload.output_tokens;
    const double expected_avg_prompt_tokens =
        (static_cast<double>(runtime->request_prompt_token_count(first_request)) +
         static_cast<double>(runtime->request_prompt_token_count(second_request))) / 2.0;
    const uint64_t expected_total_prompt_tokens =
        runtime->request_prompt_token_count(first_request) +
        runtime->request_prompt_token_count(second_request);

    BenchmarkRunner runner(runtime);
    const BenchmarkSummary summary = runner.run(config);
    EXPECT_EQ(summary.request_results.size(), static_cast<size_t>(2));
    EXPECT_TRUE(summary.total_runtime_ms > 0.0);
    EXPECT_TRUE(summary.completed_requests >= 1);
    EXPECT_EQ(summary.total_model_tokens, summary.total_prompt_tokens + summary.total_tokens_generated);
    EXPECT_NEAR(summary.output_tokens_per_sec, summary.throughput_tokens_per_sec, 1e-9);
    EXPECT_TRUE(summary.prompt_tokens_per_sec >= 0.0);
    EXPECT_TRUE(summary.total_model_tokens_per_sec >= summary.output_tokens_per_sec);
    EXPECT_TRUE(summary.avg_batch_size >= 0.0);
    EXPECT_NEAR(summary.avg_input_tokens, expected_avg_prompt_tokens, 1e-9);
    EXPECT_EQ(summary.total_prompt_tokens, expected_total_prompt_tokens);
    EXPECT_EQ(summary.compute_threads, runtime->config().model.compute_threads);
    EXPECT_EQ(summary.context_tokens, runtime->config().limits.context_tokens);
    EXPECT_NEAR(
        summary.avg_output_tokens,
        summary.completed_requests == 0
            ? 0.0
            : static_cast<double>(summary.total_tokens_generated) / static_cast<double>(summary.completed_requests),
        1e-9);
    EXPECT_EQ(
        summary.length_limited_requests + summary.early_stopped_requests,
        static_cast<uint64_t>(summary.completed_requests));
    uint64_t finish_reason_total = 0;
    for (const auto& item : summary.finish_reason_counts) {
        finish_reason_total += item.second;
    }
    EXPECT_EQ(finish_reason_total, static_cast<uint64_t>(summary.completed_requests));
}

void test_prompt_generator_categories(ModelRuntime* runtime) {
    Tokenizer tokenizer = runtime->tokenizer();
    PromptGenerator generator(&tokenizer, 123);

    const std::string chat = generator.generate(32, PromptCategory::Chat);
    EXPECT_TRUE(!chat.empty());
    EXPECT_TRUE(tokenizer.encode(chat).size() <= static_cast<size_t>(32));

    const std::string code = generator.generate(32, PromptCategory::Code);
    EXPECT_TRUE(!code.empty());
    EXPECT_TRUE(tokenizer.encode(code).size() <= static_cast<size_t>(32));

    const std::string reasoning = generator.generate(32, PromptCategory::Reasoning);
    EXPECT_TRUE(!reasoning.empty());
    EXPECT_TRUE(tokenizer.encode(reasoning).size() <= static_cast<size_t>(32));
    EXPECT_TRUE(chat != code);
    EXPECT_TRUE(chat != reasoning);
    EXPECT_TRUE(code != reasoning);
}

void test_prompt_generator_seeded_diversity(ModelRuntime* runtime) {
    Tokenizer tokenizer = runtime->tokenizer();
    PromptGenerator generator(&tokenizer, 12345);
    std::vector<std::string> prompts;
    for (int i = 0; i < 24; ++i) {
        prompts.push_back(generator.generate(192, static_cast<PromptCategory>(i % 3)));
    }

    std::sort(prompts.begin(), prompts.end());
    const auto unique_end = std::unique(prompts.begin(), prompts.end());
    EXPECT_EQ(static_cast<size_t>(std::distance(prompts.begin(), unique_end)), prompts.size());
}

void test_prompt_generator_adds_filler_for_large_target(ModelRuntime* runtime) {
    Tokenizer tokenizer = runtime->tokenizer();
    PromptGenerator generator(&tokenizer, 7);

    const std::string prompt = generator.generate(256, PromptCategory::Chat);
    EXPECT_TRUE(prompt.find("Additional context:") != std::string::npos);
    EXPECT_TRUE(prompt.find("request") != std::string::npos ||
                prompt.find("benchmark") != std::string::npos ||
                prompt.find("service") != std::string::npos ||
                prompt.find("tokens") != std::string::npos);
    EXPECT_TRUE(tokenizer.encode(prompt).size() >= static_cast<size_t>(256));
}

void test_prompt_generator_caps_small_targets(ModelRuntime* runtime) {
    Tokenizer tokenizer = runtime->tokenizer();
    PromptGenerator generator(&tokenizer, 99);

    for (int i = 0; i < 12; ++i) {
        const std::string prompt = generator.generate(64, static_cast<PromptCategory>(i % 3));
        EXPECT_TRUE(tokenizer.encode(prompt).size() <= static_cast<size_t>(64));
    }
}

void test_synthetic_prompt_generation(ModelRuntime* runtime) {
    BenchmarkConfig config = make_preset(BenchmarkPreset::kLatency);
    config.traffic.request_count = 1;
    config.traffic.concurrency = 1;
    config.serving.decode_batch_size = 1;
    config.traffic.warmup_requests = 0;
    config.workload.input_tokens = 32;
    config.workload.output_tokens = 1;
    config.workload.prompts.clear();

    BenchmarkRunner runner(runtime);
    const BenchmarkSummary summary = runner.run(config);
    EXPECT_EQ(summary.request_results.size(), static_cast<size_t>(1));
    EXPECT_TRUE(summary.avg_input_tokens > 0.0);
    EXPECT_TRUE(summary.avg_input_tokens >= 32.0);
    EXPECT_TRUE(summary.avg_input_tokens <= 192.0);
}

void test_throughput_synthetic_prompts_fit_context(ModelRuntime* runtime) {
    BenchmarkConfig config = make_preset(BenchmarkPreset::kThroughput);
    config.traffic.request_count = 2;
    config.traffic.concurrency = 1;
    config.serving.decode_batch_size = 1;
    config.traffic.warmup_requests = 0;
    config.workload.output_tokens = 1;
    config.workload.synthetic_prompt_count = 2;
    config.workload.synthetic_random_output_tokens = false;

    BenchmarkRunner runner(runtime);
    const BenchmarkSummary summary = runner.run(config);
    EXPECT_EQ(summary.request_results.size(), static_cast<size_t>(2));
    EXPECT_TRUE(summary.avg_input_tokens > 0.0);
    EXPECT_TRUE(summary.avg_input_tokens < static_cast<double>(runtime->config().limits.context_tokens));
    EXPECT_TRUE(summary.avg_input_tokens <= 192.0);
}

void test_synthetic_prompts_fit_small_runtime_context() {
    const std::string model_path = NANO_LLM_TEST_MODEL_PATH;
    if (model_path.empty()) {
        std::cerr << "[SKIP] unit_benchmark: NANO_LLM_TEST_MODEL is empty\n";
        return;
    }

    std::unique_ptr<ModelRuntime> runtime;
    try {
        RuntimeConfig runtime_config;
        runtime_config.model.path = model_path;
        runtime_config.model.compute_threads = 1;
        runtime_config.limits.context_tokens = 96;
        runtime_config.limits.default_max_new_tokens = 2;
        runtime_config.memory.prefill_ctx_bytes = 256ull * 1024 * 1024;
        runtime_config.memory.decode_ctx_bytes = 128ull * 1024 * 1024;
        runtime = std::make_unique<ModelRuntime>(runtime_config);
    } catch (const std::exception& e) {
        std::cerr << "[SKIP] unit_benchmark: failed to load model at '" << model_path
                  << "' (" << e.what() << ")\n";
        return;
    }

    BenchmarkConfig config;
    config.traffic.mode = BenchmarkMode::kOfflineThroughput;
    config.traffic.request_count = 1;
    config.traffic.concurrency = 1;
    config.traffic.warmup_requests = 0;
    config.serving.decode_batch_size = 1;
    config.workload.input_tokens = 512;
    config.workload.output_tokens = 1;
    config.workload.synthetic_prompt_count = 1;
    config.workload.synthetic_category = PromptCategory::Chat;

    BenchmarkRunner runner(runtime.get());
    const BenchmarkSummary summary = runner.run(config);
    EXPECT_EQ(summary.request_results.size(), static_cast<size_t>(1));
    EXPECT_TRUE(summary.avg_input_tokens > 0.0);
    EXPECT_TRUE(summary.avg_input_tokens < static_cast<double>(runtime->config().limits.context_tokens));
}

void test_runner_rejects_prompt_that_exceeds_context(ModelRuntime* runtime) {
    BenchmarkConfig config;
    config.traffic.mode = BenchmarkMode::kOfflineThroughput;
    config.traffic.request_count = 1;
    config.traffic.concurrency = 1;
    config.traffic.warmup_requests = 0;
    config.serving.decode_batch_size = 1;
    config.workload.output_tokens = 1;
    config.workload.prompts = {std::string(4096, 'x')};

    BenchmarkRunner runner(runtime);
    bool threw = false;
    try {
        (void)runner.run(config);
    } catch (const std::invalid_argument& e) {
        threw = std::string(e.what()).find("benchmark prompt 0") != std::string::npos &&
            std::string(e.what()).find("context limit") != std::string::npos;
    }
    EXPECT_TRUE(threw);
}

void test_human_summary_reports_failed_requests_and_errors() {
    BenchmarkSummary summary;
    summary.preset = BenchmarkPreset::kThroughput;
    summary.request_count = 3;
    summary.concurrency = 1;
    summary.completed_requests = 1;
    summary.request_results = {
        BenchmarkRequestResult{0, 0, 16, 4, 10.0, 1.0, 0.0, 1.0, 2.0, true, true, true, true, true, true, "length", ""},
        BenchmarkRequestResult{1, 0, 32, 0, 20.0, 0.0, 0.0, 0.0, 0.0, true, false, true, false, false, false, "", "prompt exceeds max sequence length"},
        BenchmarkRequestResult{2, 0, 32, 0, 21.0, 0.0, 0.0, 0.0, 0.0, true, false, true, false, false, false, "", "prompt exceeds max sequence length"},
    };

    const std::string formatted = format_benchmark_summary(summary);
    EXPECT_TRUE(formatted.find("Completed Requests:   1\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("Failed Requests:      2\n") != std::string::npos);
    EXPECT_TRUE(formatted.find("Errors:               prompt exceeds max sequence length=2\n") != std::string::npos);
}

void test_online_runner_is_deterministic_enough(ModelRuntime* runtime) {
    BenchmarkConfig config;
    config.traffic.mode = BenchmarkMode::kOnlineServing;
    config.traffic.request_count = 3;
    config.traffic.concurrency = 1;
    config.traffic.request_rate_qps = 50.0;
    config.traffic.random_seed = 7;
    config.serving.decode_batch_size = 1;
    config.workload.output_tokens = 1;
    config.workload.prompts = {"Hi", "Hello", "Hey"};

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
    test_length_sampler_ranges();
    test_preset_defaults();
    test_preset_parse_and_format();
    test_request_trace_ttft_and_tpot();
    test_request_trace_terminal_event_controls_latency();
    test_request_trace_zero_queue_delay_is_present();
    test_request_trace_single_token_has_zero_tpot();
    test_request_trace_counts_empty_decoded_tokens();
    test_request_trace_without_tokens_has_zero_tpot();
    test_csv_export_writes_header_and_row();
    test_csv_export_preserves_unavailable_gpu_monitoring();
    test_human_summary_formatting();
    test_human_summary_uses_ms_for_short_latencies_and_off_toggle();
    test_human_summary_reports_unavailable_gpu_monitoring();
    test_human_summary_reports_early_stop_behavior();
    test_effective_decode_batch_size();

    std::unique_ptr<ModelRuntime> runtime;
    ModelRuntime* runtime_ptr = make_runtime_or_skip(&runtime);
    if (runtime_ptr != nullptr) {
        test_prompt_generator_categories(runtime_ptr);
        test_prompt_generator_seeded_diversity(runtime_ptr);
        test_prompt_generator_adds_filler_for_large_target(runtime_ptr);
        test_prompt_generator_caps_small_targets(runtime_ptr);
        test_average_batch_size_and_offline_runner(runtime_ptr);
        test_synthetic_prompt_generation(runtime_ptr);
        test_throughput_synthetic_prompts_fit_context(runtime_ptr);
        test_runner_rejects_prompt_that_exceeds_context(runtime_ptr);
        test_online_runner_is_deterministic_enough(runtime_ptr);
    }
    test_synthetic_prompts_fit_small_runtime_context();

    return test_util::finalize();
}
