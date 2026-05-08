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

constexpr auto kBatchSampleInterval = std::chrono::milliseconds(5);

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
    double avg_prompt_tokens,
    int effective_max_decode_batch_size,
    double avg_batch_size,
    SteadyClock::time_point measurement_start,
    SteadyClock::time_point measurement_end,
    std::vector<BenchmarkRequestResult> request_results) {
    BenchmarkSummary summary;
    summary.mode = config.mode;
    summary.num_requests = config.num_requests;
    summary.max_concurrency = config.max_concurrency;
    summary.avg_prompt_tokens = avg_prompt_tokens;
    summary.arrival_rate_qps = config.arrival_rate_qps;
    summary.random_seed = config.random_seed;
    summary.warmup_requests = config.warmup_requests;
    summary.scheduler_policy = config.scheduler_policy;
    summary.dynamic_batching_enabled = config.dynamic_batching_enabled;
    summary.max_active_requests = config.max_active_requests;
    summary.max_decode_batch_size = effective_max_decode_batch_size;
    summary.request_max_tokens = config.request_max_tokens;
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

class BatchSampler {
public:
    explicit BatchSampler(GenerationService* service) : service_(service) {}

    void start() {
        running_ = true;
        worker_ = std::thread([this] {
            while (running_.load(std::memory_order_relaxed)) {
                const SystemMetricsSnapshot snapshot = service_->snapshot_metrics();
                total_ += static_cast<double>(snapshot.current_batch_size);
                ++samples_;
                std::this_thread::sleep_for(kBatchSampleInterval);
            }
        });
    }

    double stop_and_average() {
        running_.store(false, std::memory_order_relaxed);
        if (worker_.joinable()) {
            worker_.join();
        }
        if (samples_ == 0) {
            return 0.0;
        }
        return total_ / static_cast<double>(samples_);
    }

private:
    GenerationService* service_ = nullptr;
    std::atomic<bool> running_{false};
    std::thread worker_;
    double total_ = 0.0;
    size_t samples_ = 0;
};

void run_warmup(
    GenerationService* service,
    const BenchmarkConfig& config,
    const std::vector<BenchmarkRequestTemplate>& prompts) {
    if (config.warmup_requests <= 0) {
        return;
    }

    std::vector<SubmittedRequest> inflight;
    std::vector<BenchmarkRequestResult> completed;
    inflight.reserve(static_cast<size_t>(config.warmup_requests));

    for (int request_index = 0; request_index < config.warmup_requests; ++request_index) {
        while (inflight.size() >= config.max_concurrency) {
            if (collect_one_ready(&inflight, &completed)) {
                continue;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        const size_t prompt_index = static_cast<size_t>(request_index) % prompts.size();
        auto stream = service->submit(make_request(prompts[prompt_index], config.request_max_tokens));
        const SteadyClock::time_point submit_time = SteadyClock::now();
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
    const std::vector<BenchmarkRequestTemplate>& prompts,
    double avg_prompt_tokens,
    int effective_max_decode_batch_size) {
    std::vector<SubmittedRequest> inflight;
    std::vector<BenchmarkRequestResult> completed_results;
    inflight.reserve(config.max_concurrency);
    completed_results.reserve(config.num_requests);

    const std::vector<double> schedule_seconds =
        config.mode == BenchmarkMode::kOnlineServing
            ? build_arrival_schedule_seconds(config.num_requests, config.arrival_rate_qps, config.random_seed)
            : std::vector<double>{};

    BatchSampler sampler(service);
    const SteadyClock::time_point measurement_clock_start = SteadyClock::now();
    SteadyClock::time_point first_submit_time{};
    SteadyClock::time_point last_completion_time = measurement_clock_start;
    sampler.start();

    for (size_t request_index = 0; request_index < config.num_requests; ++request_index) {
        if (config.mode == BenchmarkMode::kOnlineServing) {
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

        while (inflight.size() >= config.max_concurrency) {
            if (collect_one_ready(&inflight, &completed_results)) {
                continue;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        const size_t prompt_index = request_index % prompts.size();
        auto stream = service->submit(make_request(prompts[prompt_index], config.request_max_tokens));
        const SteadyClock::time_point submit_time = SteadyClock::now();
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
    const double avg_batch_size = sampler.stop_and_average();

    return summarize_results(
        config,
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
        }
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
        if (tokens_generated > 0 && completion_time >= first_token_time) {
            result.tpot_ms = duration_ms(completion_time - first_token_time) /
                static_cast<double>(tokens_generated);
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
    if (config.num_requests == 0) {
        throw std::invalid_argument("benchmark num_requests must be greater than zero");
    }
    if (config.max_concurrency == 0) {
        throw std::invalid_argument("benchmark max_concurrency must be greater than zero");
    }
    if (config.mode == BenchmarkMode::kOnlineServing && config.arrival_rate_qps <= 0.0) {
        throw std::invalid_argument("online benchmark arrival_rate_qps must be greater than zero");
    }

    std::vector<BenchmarkRequestTemplate> prompts = default_benchmark_prompts();
    if (!config.prompts.empty()) {
        prompts.clear();
        prompts.reserve(config.prompts.size());
        for (const std::string& prompt : config.prompts) {
            prompts.push_back(BenchmarkRequestTemplate{prompt, 0});
        }
    }
    if (prompts.empty()) {
        throw std::invalid_argument("benchmark requires at least one prompt");
    }

    const int effective_max_decode_batch_size = derive_max_decode_batch_size(config);
    double total_prompt_tokens = 0.0;
    for (const BenchmarkRequestTemplate& prompt : prompts) {
        total_prompt_tokens += static_cast<double>(runtime_->prompt_token_count(prompt.prompt));
    }
    const double avg_prompt_tokens = prompts.empty() ? 0.0 : total_prompt_tokens / static_cast<double>(prompts.size());
    GenerationService service(
        runtime_,
        std::max(1, config.max_active_requests),
        effective_max_decode_batch_size,
        PerformanceInstrumentationConfig{false, 1000});

    service.start();
    try {
        run_warmup(&service, config, prompts);
        BenchmarkSummary summary =
            run_measured_phase(&service, config, prompts, avg_prompt_tokens, effective_max_decode_batch_size);
        service.stop();
        if (config.export_csv) {
            append_benchmark_summary_csv(summary);
        }
        return summary;
    } catch (...) {
        service.stop();
        throw;
    }
}

std::vector<BenchmarkRequestTemplate> default_benchmark_prompts() {
    return {
        {"Write one short sentence about the ocean.", 0},
        {"Explain what a compiler does in one sentence.", 0},
        {"List two benefits of regular exercise.", 0},
        {"Reply with a brief greeting.", 0},
    };
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
    size_t num_requests,
    double arrival_rate_qps,
    uint64_t random_seed) {
    std::vector<double> schedule(num_requests, 0.0);
    if (num_requests == 0 || arrival_rate_qps <= 0.0) {
        return schedule;
    }

    std::mt19937_64 rng(random_seed);
    std::exponential_distribution<double> inter_arrival(arrival_rate_qps);
    double current_time = 0.0;
    for (size_t i = 0; i < num_requests; ++i) {
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

int derive_max_decode_batch_size(const BenchmarkConfig& config) {
    if (!config.dynamic_batching_enabled) {
        return 1;
    }
    if (config.max_decode_batch_size > 0) {
        return config.max_decode_batch_size;
    }
    return std::max(1, config.max_active_requests);
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
            << "mode,concurrency,throughput,avg_ttft,p95_latency,scheduler_policy,dynamic_batching,"
            << "max_active,max_decode_batch_size,requests_per_sec,avg_batch_size,total_tokens,total_runtime_ms\n";
    }

    output << csv_escape(benchmark_mode_to_string(summary.mode)) << ','
           << summary.max_concurrency << ','
           << std::fixed << std::setprecision(6)
           << summary.throughput_tokens_per_sec << ','
           << summary.avg_ttft_ms << ','
           << summary.p95_latency_ms << ','
           << csv_escape(summary.scheduler_policy) << ','
           << (summary.dynamic_batching_enabled ? "true" : "false") << ','
           << summary.max_active_requests << ','
           << summary.max_decode_batch_size << ','
           << summary.requests_per_sec << ','
           << summary.avg_batch_size << ','
           << summary.total_tokens_generated << ','
           << summary.total_runtime_ms << '\n';
}

std::string format_benchmark_summary(const BenchmarkSummary& summary) {
    std::ostringstream output;
    output << "==============================\n";
    output << "Benchmark Summary\n";
    output << "==============================\n\n";

    append_summary_line(&output, "Requests", std::to_string(summary.num_requests));
    append_summary_line(&output, "Concurrency", std::to_string(summary.max_concurrency));
    append_summary_line(&output, "Prompt Length", format_decimal(summary.avg_prompt_tokens, 1) + " tokens");
    append_summary_line(&output, "Generation Length", std::to_string(summary.request_max_tokens));
    output << '\n';

    append_summary_line(&output, "Dynamic Batching", summary.dynamic_batching_enabled ? "ON" : "OFF");
    output << '\n';

    append_summary_line(&output, "Throughput", format_decimal(summary.throughput_tokens_per_sec, 1) + " tok/s");
    append_summary_line(&output, "Request Throughput", format_decimal(summary.requests_per_sec, 1) + " req/s");
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
