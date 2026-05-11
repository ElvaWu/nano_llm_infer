#include "benchmark.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <future>
#include <iomanip>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <ctime>
#include <utility>

#include "generation_server.h"

namespace nano_llm_infer {

namespace {

using SteadyClock = std::chrono::steady_clock;

double duration_ms(SteadyClock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

double mean_or_zero(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    return sum / static_cast<double>(values.size());
}

std::string format_decimal(double value, int precision) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(precision) << value;
    return output.str();
}

std::string format_ms_value(double value_ms) {
    const int precision = value_ms < 1.0 ? 1 : 0;
    return format_decimal(value_ms, precision) + " ms";
}

std::string format_latency_value(double value_ms) {
    if (value_ms >= 1000.0) {
        return format_decimal(value_ms / 1000.0, 1) + " s";
    }
    const int precision = value_ms < 1.0 ? 1 : 0;
    return format_decimal(value_ms, precision) + " ms";
}

void append_summary_line(std::ostringstream* output, const std::string& label, const std::string& value) {
    *output << std::left << std::setw(22) << (label + ":") << value << '\n';
}

std::string csv_escape(const std::string& value) {
    bool needs_quotes = false;
    std::string escaped;
    escaped.reserve(value.size() + 4);
    for (char ch : value) {
        if (ch == '"' || ch == ',' || ch == '\n') {
            needs_quotes = true;
        }
        if (ch == '"') {
            escaped += "\"\"";
        } else {
            escaped.push_back(ch);
        }
    }
    if (!needs_quotes) {
        return escaped;
    }
    return "\"" + escaped + "\"";
}

GenerationRequest make_request(const BenchmarkRequestTemplate& request_template, int request_max_tokens) {
    GenerationRequest request;
    request.messages.push_back({"user", request_template.prompt});
    request.options.max_tokens = request_template.max_tokens > 0 ? request_template.max_tokens : request_max_tokens;
    return request;
}

std::vector<BenchmarkRequestTemplate> build_synthetic_prompts(
    ModelRuntime* runtime,
    int target_prompt_tokens,
    int request_max_tokens) {
    const int target = std::max(1, target_prompt_tokens);
    const std::string unit =
        "This benchmark prompt uses neutral repeated text for deterministic token length. ";
    std::string prompt = unit;
    size_t token_count = runtime->request_prompt_token_count(make_request({prompt, 0}, request_max_tokens));

    while (static_cast<int>(token_count) < target && prompt.size() < 1024 * 1024) {
        prompt += unit;
        token_count = runtime->request_prompt_token_count(make_request({prompt, 0}, request_max_tokens));
    }

    if (static_cast<int>(token_count) > target + 8) {
        std::string best = prompt;
        size_t best_tokens = token_count;
        size_t low = 0;
        size_t high = prompt.size();
        while (low < high) {
            const size_t mid = (low + high) / 2;
            std::string candidate = prompt.substr(0, mid);
            if (!candidate.empty() && candidate.back() != ' ') {
                candidate.push_back(' ');
            }
            const size_t candidate_tokens =
                runtime->request_prompt_token_count(make_request({candidate, 0}, request_max_tokens));
            if (std::abs(static_cast<int>(candidate_tokens) - target) <
                std::abs(static_cast<int>(best_tokens) - target)) {
                best = std::move(candidate);
                best_tokens = candidate_tokens;
            }
            if (candidate_tokens < static_cast<size_t>(target)) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        prompt = std::move(best);
    }

    return {BenchmarkRequestTemplate{prompt, 0}};
}

struct SubmittedRequest {
    std::future<BenchmarkRequestResult> future;
};

BenchmarkRequestResult wait_for_stream(
    size_t request_index,
    size_t prompt_index,
    std::shared_ptr<GenerationStream> stream,
    SteadyClock::time_point submit_time) {
    BenchmarkRequestTrace trace;
    trace.submit_time = submit_time;

    GenerationEvent event;
    while (stream->wait_next(&event)) {
        trace.record_event(event, SteadyClock::now());
    }
    trace.completion_time = SteadyClock::now();
    return trace.finalize(request_index, prompt_index);
}

bool collect_one_ready(
    std::vector<SubmittedRequest>* inflight,
    std::vector<BenchmarkRequestResult>* completed_results) {
    for (size_t i = 0; i < inflight->size(); ++i) {
        auto& submitted = (*inflight)[i];
        if (submitted.future.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
            continue;
        }
        completed_results->push_back(submitted.future.get());
        inflight->erase(inflight->begin() + static_cast<std::ptrdiff_t>(i));
        return true;
    }
    return false;
}

void drain_all(
    std::vector<SubmittedRequest>* inflight,
    std::vector<BenchmarkRequestResult>* completed_results) {
    while (!inflight->empty()) {
        if (collect_one_ready(inflight, completed_results)) {
            continue;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

BenchmarkSummary summarize_results(
    const BenchmarkConfig& config,
    const RuntimeConfig& runtime_config,
    double avg_prompt_tokens,
    int effective_max_decode_batch_size,
    double avg_batch_size,
    SteadyClock::time_point measurement_start,
    SteadyClock::time_point measurement_end,
    std::vector<BenchmarkRequestResult> request_results) {
    BenchmarkSummary summary;
    summary.preset = config.preset;
    summary.mode = config.traffic.mode;
    summary.request_count = config.traffic.request_count;
    summary.concurrency = config.traffic.concurrency;
    summary.request_rate_qps = effective_request_rate_qps(config);
    summary.random_seed = config.traffic.random_seed;
    summary.warmup_requests = config.traffic.warmup_requests;
    summary.compute_threads = runtime_config.model.compute_threads;
    summary.context_tokens = runtime_config.limits.context_tokens;
    summary.avg_input_tokens = avg_prompt_tokens;
    summary.output_tokens = config.workload.output_tokens;
    summary.decode_batch_size = effective_max_decode_batch_size;
    summary.avg_batch_size = avg_batch_size;
    summary.request_results = std::move(request_results);

    if (!summary.request_results.empty()) {
        std::sort(
            summary.request_results.begin(),
            summary.request_results.end(),
            [](const BenchmarkRequestResult& lhs, const BenchmarkRequestResult& rhs) {
                return lhs.request_index < rhs.request_index;
            });
    }

    std::vector<double> latencies;
    std::vector<double> ttfts;
    std::vector<double> tpots;
    latencies.reserve(summary.request_results.size());
    ttfts.reserve(summary.request_results.size());
    tpots.reserve(summary.request_results.size());

    for (const BenchmarkRequestResult& result : summary.request_results) {
        summary.total_tokens_generated += result.tokens_generated;
        if (result.completed) {
            ++summary.completed_requests;
        }
        latencies.push_back(result.latency_ms);
        if (result.ttft_ms > 0.0) {
            ttfts.push_back(result.ttft_ms);
        }
        if (result.tpot_ms > 0.0) {
            tpots.push_back(result.tpot_ms);
        }
    }

    if (measurement_end > measurement_start) {
        summary.total_runtime_ms = duration_ms(measurement_end - measurement_start);
        const double runtime_seconds = summary.total_runtime_ms / 1000.0;
        if (runtime_seconds > 0.0) {
            summary.throughput_tokens_per_sec =
                static_cast<double>(summary.total_tokens_generated) / runtime_seconds;
            summary.requests_per_sec =
                static_cast<double>(summary.completed_requests) / runtime_seconds;
        }
    }

    summary.avg_latency_ms = mean_or_zero(latencies);
    summary.p50_latency_ms = benchmark_percentile(latencies, 50.0);
    summary.p95_latency_ms = benchmark_percentile(latencies, 95.0);
    summary.avg_ttft_ms = mean_or_zero(ttfts);
    summary.p50_ttft_ms = benchmark_percentile(ttfts, 50.0);
    summary.p95_ttft_ms = benchmark_percentile(ttfts, 95.0);
    summary.avg_tpot_ms = mean_or_zero(tpots);
    return summary;
}

void run_warmup(
    GenerationService* service,
    const BenchmarkConfig& config,
    const std::vector<BenchmarkRequestTemplate>& prompts) {
    if (config.traffic.warmup_requests <= 0) {
        return;
    }

    std::vector<SubmittedRequest> inflight;
    std::vector<BenchmarkRequestResult> completed;
    inflight.reserve(static_cast<size_t>(config.traffic.warmup_requests));

    for (int request_index = 0; request_index < config.traffic.warmup_requests; ++request_index) {
        while (inflight.size() >= config.traffic.concurrency) {
            if (collect_one_ready(&inflight, &completed)) {
                continue;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        const size_t prompt_index = static_cast<size_t>(request_index) % prompts.size();
        const SteadyClock::time_point submit_time = SteadyClock::now();
        auto stream = service->submit(make_request(prompts[prompt_index], config.workload.output_tokens));
        inflight.push_back(SubmittedRequest{
            std::async(
                std::launch::async,
                wait_for_stream,
                static_cast<size_t>(request_index),
                prompt_index,
                stream,
                submit_time),
        });
    }

    drain_all(&inflight, &completed);
}

BenchmarkSummary run_measured_phase(
    GenerationService* service,
    const BenchmarkConfig& config,
    const RuntimeConfig& runtime_config,
    const std::vector<BenchmarkRequestTemplate>& prompts,
    double avg_prompt_tokens,
    int effective_max_decode_batch_size) {
    std::vector<SubmittedRequest> inflight;
    std::vector<BenchmarkRequestResult> completed_results;
    inflight.reserve(config.traffic.concurrency);
    completed_results.reserve(config.traffic.request_count);

    const std::vector<double> schedule_seconds =
        config.traffic.mode == BenchmarkMode::kOnlineServing
            ? build_arrival_schedule_seconds(
                  config.traffic.request_count,
                  effective_request_rate_qps(config),
                  config.traffic.random_seed)
            : std::vector<double>{};

    const SteadyClock::time_point measurement_clock_start = SteadyClock::now();
    const SystemMetricsSnapshot metrics_before = service->snapshot_metrics();
    SteadyClock::time_point first_submit_time{};
    SteadyClock::time_point last_completion_time = measurement_clock_start;

    for (size_t request_index = 0; request_index < config.traffic.request_count; ++request_index) {
        if (config.traffic.mode == BenchmarkMode::kOnlineServing) {
            const auto target = measurement_clock_start +
                std::chrono::duration_cast<SteadyClock::duration>(
                    std::chrono::duration<double>(schedule_seconds[request_index]));
            while (SteadyClock::now() < target) {
                if (collect_one_ready(&inflight, &completed_results)) {
                    continue;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }

        while (inflight.size() >= config.traffic.concurrency) {
            if (collect_one_ready(&inflight, &completed_results)) {
                continue;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        const size_t prompt_index = request_index % prompts.size();
        const SteadyClock::time_point submit_time = SteadyClock::now();
        auto stream = service->submit(make_request(prompts[prompt_index], config.workload.output_tokens));
        if (request_index == 0) {
            first_submit_time = submit_time;
        }
        inflight.push_back(SubmittedRequest{
            std::async(
                std::launch::async,
                wait_for_stream,
                request_index,
                prompt_index,
                stream,
                submit_time),
        });
    }

    drain_all(&inflight, &completed_results);
    last_completion_time = SteadyClock::now();
    const SystemMetricsSnapshot metrics_after = service->snapshot_metrics();
    const uint64_t decode_steps =
        metrics_after.decode_step_count >= metrics_before.decode_step_count
            ? metrics_after.decode_step_count - metrics_before.decode_step_count
            : 0;
    const uint64_t decode_batch_total =
        metrics_after.decode_batch_size_total >= metrics_before.decode_batch_size_total
            ? metrics_after.decode_batch_size_total - metrics_before.decode_batch_size_total
            : 0;
    const double avg_batch_size = decode_steps == 0
        ? 0.0
        : static_cast<double>(decode_batch_total) / static_cast<double>(decode_steps);

    return summarize_results(
        config,
        runtime_config,
        avg_prompt_tokens,
        effective_max_decode_batch_size,
        avg_batch_size,
        first_submit_time == SteadyClock::time_point{} ? measurement_clock_start : first_submit_time,
        last_completion_time,
        std::move(completed_results));
}

}  // namespace

void BenchmarkRequestTrace::record_event(const GenerationEvent& event, std::chrono::steady_clock::time_point now) {
    if (!event.text.empty()) {
        if (!saw_first_token) {
            first_token_time = now;
            saw_first_token = true;
        } else if (now >= last_token_time) {
            inter_token_duration += now - last_token_time;
            inter_token_interval_count++;
        }
        last_token_time = now;
        ++tokens_generated;
    }
    if (!event.finish_reason.empty()) {
        saw_terminal = true;
        finish_reason = event.finish_reason;
    }
    if (!event.error.empty()) {
        saw_terminal = true;
        error = event.error;
    }
}

BenchmarkRequestResult BenchmarkRequestTrace::finalize(size_t request_index, size_t prompt_index) const {
    BenchmarkRequestResult result;
    result.request_index = request_index;
    result.prompt_index = prompt_index;
    result.tokens_generated = tokens_generated;
    result.finish_reason = finish_reason;
    result.error = error;
    result.completed = saw_terminal && error.empty();

    if (completion_time > submit_time) {
        result.latency_ms = duration_ms(completion_time - submit_time);
    }
    if (saw_first_token && first_token_time >= submit_time) {
        result.ttft_ms = duration_ms(first_token_time - submit_time);
        if (inter_token_interval_count > 0) {
            result.tpot_ms = duration_ms(inter_token_duration) /
                static_cast<double>(inter_token_interval_count);
        }
    }
    return result;
}

BenchmarkRunner::BenchmarkRunner(ModelRuntime* runtime) : runtime_(runtime) {
}

BenchmarkSummary BenchmarkRunner::run(const BenchmarkConfig& config) {
    if (runtime_ == nullptr) {
        throw std::invalid_argument("BenchmarkRunner requires a non-null runtime");
    }
    if (config.traffic.request_count == 0) {
        throw std::invalid_argument("benchmark request_count must be greater than zero");
    }
    if (config.traffic.concurrency == 0) {
        throw std::invalid_argument("benchmark concurrency must be greater than zero");
    }
    if (config.workload.input_tokens <= 0) {
        throw std::invalid_argument("benchmark input_tokens must be greater than zero");
    }
    if (config.workload.output_tokens <= 0) {
        throw std::invalid_argument("benchmark output_tokens must be greater than zero");
    }
    if (config.traffic.mode == BenchmarkMode::kOnlineServing && config.traffic.request_rate_qps <= 0.0) {
        throw std::invalid_argument("online benchmark request_rate_qps must be greater than zero");
    }
    if (config.serving.decode_batch_size < 0) {
        throw std::invalid_argument("benchmark decode_batch_size must be non-negative");
    }

    std::vector<BenchmarkRequestTemplate> prompts;
    if (config.workload.prompts.empty()) {
        prompts = build_synthetic_prompts(runtime_, config.workload.input_tokens, config.workload.output_tokens);
    } else {
        prompts.reserve(config.workload.prompts.size());
        for (const std::string& prompt : config.workload.prompts) {
            prompts.push_back(BenchmarkRequestTemplate{prompt, 0});
        }
    }
    if (prompts.empty()) {
        throw std::invalid_argument("benchmark requires at least one prompt");
    }

    const int effective_max_decode_batch_size = effective_decode_batch_size(config);
    double total_prompt_tokens = 0.0;
    for (const BenchmarkRequestTemplate& prompt : prompts) {
        total_prompt_tokens += static_cast<double>(
            runtime_->request_prompt_token_count(make_request(prompt, config.workload.output_tokens)));
    }
    const double avg_prompt_tokens = prompts.empty() ? 0.0 : total_prompt_tokens / static_cast<double>(prompts.size());
    GenerationService service(
        runtime_,
        effective_active_request_limit(config),
        effective_max_decode_batch_size,
        PerformanceInstrumentationConfig{false, 1000});

    service.start();
    try {
        run_warmup(&service, config, prompts);
        BenchmarkSummary summary =
            run_measured_phase(
                &service,
                config,
                runtime_->config(),
                prompts,
                avg_prompt_tokens,
                effective_max_decode_batch_size);
        service.stop();
        if (config.output.export_csv) {
            append_benchmark_summary_csv(summary);
        }
        return summary;
    } catch (...) {
        service.stop();
        throw;
    }
}

BenchmarkConfig make_preset(BenchmarkPreset preset) {
    BenchmarkConfig config;
    config.preset = preset;
    config.traffic.random_seed = 12345;
    config.traffic.warmup_requests = 1;
    config.serving.decode_batch_size = 0;
    config.output.export_csv = false;

    switch (preset) {
        case BenchmarkPreset::kChat:
            config.traffic.mode = BenchmarkMode::kOnlineServing;
            config.traffic.request_count = 16;
            config.traffic.concurrency = 2;
            config.workload.input_tokens = 32;
            config.workload.output_tokens = 4;
            config.traffic.request_rate_qps = 8.0;
            break;
        case BenchmarkPreset::kThroughput:
            config.traffic.mode = BenchmarkMode::kOfflineThroughput;
            config.traffic.request_count = 12;
            config.traffic.concurrency = 4;
            config.workload.input_tokens = 32;
            config.workload.output_tokens = 4;
            config.traffic.request_rate_qps = 0.0;
            break;
        case BenchmarkPreset::kLatency:
            config.traffic.mode = BenchmarkMode::kOfflineThroughput;
            config.traffic.request_count = 4;
            config.traffic.concurrency = 1;
            config.workload.input_tokens = 16;
            config.workload.output_tokens = 2;
            config.traffic.request_rate_qps = 0.0;
            break;
        case BenchmarkPreset::kLongContext:
            config.traffic.mode = BenchmarkMode::kOfflineThroughput;
            config.traffic.request_count = 2;
            config.traffic.concurrency = 1;
            config.workload.input_tokens = 128;
            config.workload.output_tokens = 2;
            config.traffic.request_rate_qps = 0.0;
            break;
    }

    return config;
}

std::string benchmark_preset_to_string(BenchmarkPreset preset) {
    switch (preset) {
        case BenchmarkPreset::kChat:
            return "chat";
        case BenchmarkPreset::kThroughput:
            return "throughput";
        case BenchmarkPreset::kLatency:
            return "latency";
        case BenchmarkPreset::kLongContext:
            return "long_context";
    }
    return "unknown";
}

BenchmarkPreset parse_benchmark_preset(const std::string& value) {
    if (value == "chat") {
        return BenchmarkPreset::kChat;
    }
    if (value == "throughput") {
        return BenchmarkPreset::kThroughput;
    }
    if (value == "latency") {
        return BenchmarkPreset::kLatency;
    }
    if (value == "long_context") {
        return BenchmarkPreset::kLongContext;
    }
    throw std::invalid_argument("invalid benchmark preset: " + value);
}

std::string default_benchmark_csv_path() {
    const std::time_t now = std::time(nullptr);
    std::tm local_tm{};
#if defined(_WIN32)
    localtime_s(&local_tm, &now);
#else
    localtime_r(&now, &local_tm);
#endif

    std::ostringstream name;
    name << "benchmark_" << (local_tm.tm_year + 1900)
         << '_' << std::setw(2) << std::setfill('0') << (local_tm.tm_mon + 1)
         << '_' << std::setw(2) << std::setfill('0') << local_tm.tm_mday
         << '_' << std::setw(2) << std::setfill('0') << local_tm.tm_hour
         << '_' << std::setw(2) << std::setfill('0') << local_tm.tm_min
         << ".csv";
    return name.str();
}

std::vector<double> build_arrival_schedule_seconds(
    size_t request_count,
    double request_rate_qps,
    uint64_t random_seed) {
    std::vector<double> schedule(request_count, 0.0);
    if (request_count == 0 || request_rate_qps <= 0.0) {
        return schedule;
    }

    std::mt19937_64 rng(random_seed);
    std::exponential_distribution<double> inter_arrival(request_rate_qps);
    double current_time = 0.0;
    for (size_t i = 0; i < request_count; ++i) {
        if (i > 0) {
            current_time += inter_arrival(rng);
        }
        schedule[i] = current_time;
    }
    return schedule;
}

double benchmark_percentile(const std::vector<double>& values, double percentile) {
    if (values.empty()) {
        return 0.0;
    }
    if (percentile <= 0.0) {
        return *std::min_element(values.begin(), values.end());
    }
    if (percentile >= 100.0) {
        return *std::max_element(values.begin(), values.end());
    }

    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    const double position = (percentile / 100.0) * static_cast<double>(sorted.size() - 1);
    const size_t lower = static_cast<size_t>(std::floor(position));
    const size_t upper = static_cast<size_t>(std::ceil(position));
    if (lower == upper) {
        return sorted[lower];
    }
    const double fraction = position - static_cast<double>(lower);
    return sorted[lower] + (sorted[upper] - sorted[lower]) * fraction;
}

double effective_request_rate_qps(const BenchmarkConfig& config) {
    return config.traffic.request_rate_qps;
}

int effective_decode_batch_size(const BenchmarkConfig& config) {
    if (config.serving.decode_batch_size > 0) {
        return config.serving.decode_batch_size;
    }
    return std::max(1, static_cast<int>(config.traffic.concurrency));
}

int effective_active_request_limit(const BenchmarkConfig& config) {
    return std::max(1, static_cast<int>(config.traffic.concurrency));
}

void append_benchmark_summary_csv(const BenchmarkSummary& summary) {
    const std::string csv_path = default_benchmark_csv_path();
    bool needs_header = true;
    {
        std::ifstream input(csv_path);
        needs_header = !input.good() || input.peek() == std::ifstream::traits_type::eof();
    }

    std::ofstream output(csv_path, std::ios::app);
    if (!output) {
        throw std::runtime_error("failed to open benchmark CSV path: " + csv_path);
    }

    if (needs_header) {
        output
            << "preset,mode,request_count,concurrency,request_rate_qps,random_seed,warmup_requests,"
            << "compute_threads,context_tokens,avg_input_tokens,output_tokens,decode_batch_size,"
            << "throughput_tokens_per_sec,requests_per_sec,avg_ttft_ms,p95_ttft_ms,"
            << "avg_latency_ms,p95_latency_ms,avg_batch_size,total_tokens,total_runtime_ms,completed_requests\n";
    }

    output << csv_escape(benchmark_preset_to_string(summary.preset)) << ','
           << csv_escape(benchmark_mode_to_string(summary.mode)) << ','
           << summary.request_count << ','
           << summary.concurrency << ','
           << std::fixed << std::setprecision(6)
           << summary.request_rate_qps << ','
           << summary.random_seed << ','
           << summary.warmup_requests << ','
           << summary.compute_threads << ','
           << summary.context_tokens << ','
           << summary.avg_input_tokens << ','
           << summary.output_tokens << ','
           << summary.decode_batch_size << ','
           << summary.throughput_tokens_per_sec << ','
           << summary.requests_per_sec << ','
           << summary.avg_ttft_ms << ','
           << summary.p95_ttft_ms << ','
           << summary.avg_latency_ms << ','
           << summary.p95_latency_ms << ','
           << summary.avg_batch_size << ','
           << summary.total_tokens_generated << ','
           << summary.total_runtime_ms << ','
           << summary.completed_requests << '\n';
}

std::string format_benchmark_summary(const BenchmarkSummary& summary) {
    std::ostringstream output;
    output << "==============================\n";
    output << "Benchmark Summary\n";
    output << "==============================\n\n";

    append_summary_line(&output, "Preset", benchmark_preset_to_string(summary.preset));
    append_summary_line(&output, "Requests", std::to_string(summary.request_count));
    append_summary_line(&output, "Mode", benchmark_mode_to_string(summary.mode));
    append_summary_line(&output, "Concurrency", std::to_string(summary.concurrency));
    append_summary_line(
        &output,
        "Request Rate",
        summary.request_rate_qps > 0.0 ? format_decimal(summary.request_rate_qps, 2) + " req/s" : "unlimited");
    append_summary_line(&output, "Random Seed", std::to_string(summary.random_seed));
    append_summary_line(&output, "Warmup Requests", std::to_string(summary.warmup_requests));
    append_summary_line(&output, "Threads", std::to_string(summary.compute_threads));
    append_summary_line(&output, "Context Tokens", std::to_string(summary.context_tokens));
    append_summary_line(&output, "Input Tokens", format_decimal(summary.avg_input_tokens, 1) + " tokens");
    append_summary_line(&output, "Max Output Tokens", std::to_string(summary.output_tokens));
    append_summary_line(&output, "Decode Batch Size", std::to_string(summary.decode_batch_size));
    output << '\n';

    append_summary_line(&output, "Throughput", format_decimal(summary.throughput_tokens_per_sec, 1) + " tok/s");
    append_summary_line(&output, "Request Throughput", format_decimal(summary.requests_per_sec, 3) + " req/s");
    append_summary_line(&output, "Total Output Tokens", std::to_string(summary.total_tokens_generated));
    append_summary_line(&output, "Total Runtime", format_latency_value(summary.total_runtime_ms));
    output << '\n';

    append_summary_line(&output, "Avg TTFT", format_ms_value(summary.avg_ttft_ms));
    append_summary_line(&output, "P95 TTFT", format_ms_value(summary.p95_ttft_ms));
    output << '\n';

    append_summary_line(&output, "Avg Latency", format_latency_value(summary.avg_latency_ms));
    append_summary_line(&output, "P95 Latency", format_latency_value(summary.p95_latency_ms));
    output << '\n';

    append_summary_line(&output, "Avg Batch Size", format_decimal(summary.avg_batch_size, 1));
    output << '\n';
    append_summary_line(&output, "Completed Requests", std::to_string(summary.completed_requests));
    return output.str();
}

std::string benchmark_mode_to_string(BenchmarkMode mode) {
    switch (mode) {
        case BenchmarkMode::kOfflineThroughput:
            return "offline_throughput";
        case BenchmarkMode::kOnlineServing:
            return "online_serving";
    }
    return "unknown";
}

}  // namespace nano_llm_infer
