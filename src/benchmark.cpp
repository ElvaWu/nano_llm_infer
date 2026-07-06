#include "benchmark.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <ctime>
#include <utility>

#if defined(NANO_LLM_ENABLE_NVML)
#include <nvml.h>
#endif
#if defined(NANO_LLM_ENABLE_CUDA)
#include "ggml-backend.h"
#include "ggml-cuda.h"
#endif

#include "generation_server.h"

namespace nano_llm_infer {

LengthSampler::LengthSampler(uint64_t seed) : rng_(static_cast<std::mt19937::result_type>(seed)) {
}

int LengthSampler::sample(PromptLengthClass length_class) {
    switch (length_class) {
        case PromptLengthClass::Short: {
            std::uniform_int_distribution<int> distribution(32, 128);
            return distribution(rng_);
        }
        case PromptLengthClass::Medium: {
            std::uniform_int_distribution<int> distribution(256, 512);
            return distribution(rng_);
        }
        case PromptLengthClass::Long: {
            std::uniform_int_distribution<int> distribution(1024, 2048);
            return distribution(rng_);
        }
    }
    return 256;
}

namespace {

using SteadyClock = std::chrono::steady_clock;

const std::vector<std::string>& prompt_templates_for_category(PromptCategory category) {
    static const std::vector<std::string> kChatTemplates = {
        "Explain how transformer attention works to a software engineer who understands matrices but has not built a language model before.",
        "A team is adopting machine learning for document classification. Describe the practical steps they should take to collect data, train a baseline, and evaluate quality.",
        "I am planning a weekend project and need a practical assistant-style checklist for choosing tools, estimating effort, and avoiding common mistakes.",
        "A product manager asks whether a customer support chatbot should use retrieval, fine tuning, or prompt engineering. Give a practical recommendation.",
        "Summarize the operational risks of deploying a new language model endpoint for a small engineering team that owns latency, cost, and reliability.",
        "Help a developer debug why an assistant sometimes gives inconsistent answers when the same user question is sent with slightly different context.",
        "Draft a concise onboarding note for engineers who need to interpret benchmark results, compare runs, and decide whether a model change is acceptable.",
        "Explain how to choose between a smaller fast model and a larger accurate model for an internal knowledge assistant used by several departments.",
        "A user wants to migrate a batch summarization workflow from a script to a service. Outline the design decisions and failure cases to consider.",
        "Describe a practical process for evaluating answer quality when the prompts include logs, configuration snippets, and support-ticket history.",
        "Give advice to a team that needs to reduce first-token latency without making generated answers noticeably shorter or less useful.",
        "Explain why tokenization can make two prompts with similar character counts behave differently during inference and capacity planning.",
    };
    static const std::vector<std::string> kCodeTemplates = {
        "Write a C++ quicksort implementation for a vector of integers, then explain the partition step and the average-case complexity.",
        "Design a thread-safe queue in modern C++ using a mutex and condition variable. Include the public methods and describe shutdown behavior.",
        "Implement a small parser helper function that reads comma-separated key value pairs, validates required fields, and returns structured errors.",
        "Review a C++ function that reuses a buffer across requests and identify ownership, lifetime, and data-race risks before suggesting a fix.",
        "Write pseudocode for a scheduler that admits requests up to a concurrency limit, batches decode steps, and records queue delay metrics.",
        "Implement a helper that parses command-line options for request count, concurrency, random seed, and output token limits with clear errors.",
        "Design a compact metrics accumulator for latency percentiles, token totals, and finish-reason counts without storing every generated token.",
        "Explain how to structure a unit test for a seeded random workload generator so the test is deterministic but not tied to exact text.",
        "Create a C++ RAII wrapper for a background sampling thread that starts on construction, records periodic observations, and joins on destruction.",
        "Refactor a loop that drains completed futures from an in-flight request vector while preserving request order in the final report.",
        "Describe how to validate that a tokenizer-backed prompt generator hits a minimum token target without trimming strings by byte count.",
        "Write a small utility function that rotates through workload categories, samples target lengths, and returns request templates for a benchmark.",
    };
    static const std::vector<std::string> kReasoningTemplates = {
        "Solve this math problem step by step: a service processes 18 requests per second, each request generates 64 tokens, and capacity improves by 25 percent. What is the new token throughput?",
        "Reason step by step about whether a benchmark should prioritize lower latency, higher throughput, or more predictable tail latency for an interactive coding assistant.",
        "Compare the tradeoffs between batching more requests together and keeping the first-token delay low. Give a structured analysis with assumptions.",
        "A benchmark sends 96 requests with concurrency 16. Each request has a different prompt length and output cap. Reason about how this affects average latency.",
        "Estimate how prefill cost changes when prompt length increases from 128 tokens to 512 tokens, assuming decode length stays fixed.",
        "Analyze whether a workload with repeated identical prompts is likely to overstate, understate, or hide real serving bottlenecks.",
        "A model server improves output throughput but worsens p95 time to first token. Explain how to decide whether the change is acceptable.",
        "Work through a capacity-planning scenario where a team must choose between doubling batch size or adding another worker process.",
        "Compare two benchmark runs where total generated tokens are equal but one run has many short prompts and the other has fewer long prompts.",
        "Reason about how ignoring EOS changes sustained decode measurements and why that setting may be useful for a throughput benchmark.",
        "Evaluate the tradeoff between deterministic benchmark seeds and realistic prompt variety in repeated performance experiments.",
        "Given a fixed context window, decide how to allocate tokens between retrieved documents, user instructions, and generation budget.",
    };

    switch (category) {
        case PromptCategory::Chat:
            return kChatTemplates;
        case PromptCategory::Code:
            return kCodeTemplates;
        case PromptCategory::Reasoning:
            return kReasoningTemplates;
    }
    return kChatTemplates;
}

const std::vector<std::string>& filler_paragraphs() {
    static const std::vector<std::string> kFillers = {
        "The system records each request with its prompt length, queue delay, first token latency, generated token count, and finish reason so that repeated runs can be compared without relying on wall-clock impressions alone.",
        "Additional background describes deployment constraints, model context limits, cache residency, request admission, and the difference between prefill work and decode work when several sessions are active at the same time.",
        "The benchmark documentation should include hardware details, thread counts, memory limits, scheduling parameters, and any sampling settings that may affect deterministic reproduction of the observed measurements.",
        "For long-context evaluation, the prompt may contain technical notes, configuration snippets, operational runbooks, and implementation details that force the tokenizer to produce a large input before generation begins.",
        "CUDA kernels are launched asynchronously on the GPU stream. Memory bandwidth utilization becomes critical during attention computation for long sequence inference workloads.",
        "The service receives a mix of interactive questions, code review tasks, summarization requests, and diagnostic prompts that vary in vocabulary, formatting, and expected answer shape.",
        "A realistic run should include short direct requests, medium procedural questions, and longer context-heavy prompts so prefill and decode phases are both represented.",
        "Operators compare the request timeline with model-token throughput, GPU memory pressure, and batching efficiency before deciding whether a change is a regression.",
        "Some prompts include lists, constraints, and domain terms while others use plain conversational language, which creates different token boundaries and cache-access patterns.",
        "The generated text is intentionally neutral but varied enough to avoid measuring the same tokenizer path, attention pattern, and prompt prefix on every request.",
        "Production traces often contain repeated concepts with different wording, so the benchmark uses seeded variation instead of loading external datasets or replay files.",
        "The report should make it possible to separate queueing effects from model execution time and to understand whether longer prompts are dominating the run.",
    };
    return kFillers;
}

const std::vector<std::string>& prompt_constraints() {
    static const std::vector<std::string> kConstraints = {
        "Focus on practical steps and mention one risk that could invalidate the recommendation.",
        "Use concise paragraphs and include the assumptions that would change your answer.",
        "Include a small example, then explain how the example generalizes to a larger system.",
        "Prioritize operational details such as latency, memory use, scheduling, observability, and error handling.",
        "Compare at least two approaches and state which one you would choose for a benchmark-oriented environment.",
        "Call out the measurement signals that would prove whether the proposed approach is working.",
        "Avoid generic advice; make the answer specific to model serving, token budgets, and request concurrency.",
        "Explain the main tradeoff first, then give a short implementation or evaluation plan.",
        "Include one edge case that would be easy to miss during testing.",
        "Write for an engineer who needs enough detail to turn the answer into code or a runbook.",
    };
    return kConstraints;
}

const std::vector<std::string>& prompt_scenarios() {
    static const std::vector<std::string> kScenarios = {
        "Context: the service runs on a single GPU with bursty traffic and a fixed context window.",
        "Context: the team is comparing two commits and needs a deterministic but varied benchmark workload.",
        "Context: request prompts come from documentation, code snippets, bug reports, and user questions.",
        "Context: the deployment has strict p95 latency goals and must keep total generated tokens predictable.",
        "Context: the benchmark is run before release to catch regressions in batching, prefill, and decode behavior.",
        "Context: explicit prompt traces are unavailable, so synthetic prompts must approximate several real task shapes.",
        "Context: the model may stop early unless the workload disables stop sequences and ignores EOS.",
        "Context: engineers will inspect average input tokens, output throughput, queue delay, and finish reasons after the run.",
        "Context: some requests are simple assistant turns while others carry enough background to stress the KV cache.",
        "Context: the system should remain reproducible when the same random seed is used on the same model tokenizer.",
    };
    return kScenarios;
}

PromptCategory category_for_index(size_t index) {
    static constexpr std::array<PromptCategory, 3> kCategories = {
        PromptCategory::Chat,
        PromptCategory::Code,
        PromptCategory::Reasoning,
    };
    return kCategories[index % kCategories.size()];
}

PromptLengthClass length_class_for_index(size_t index) {
    static constexpr std::array<PromptLengthClass, 3> kLengths = {
        PromptLengthClass::Short,
        PromptLengthClass::Medium,
        PromptLengthClass::Long,
    };
    return kLengths[index % kLengths.size()];
}

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
    const std::string label_text = label + ":";
    *output << std::left << std::setw(static_cast<int>(std::max<size_t>(22, label_text.size() + 1))) << label_text
            << value << '\n';
}

std::string format_finish_reason_counts(const std::map<std::string, uint64_t>& counts) {
    if (counts.empty()) {
        return "none";
    }

    std::ostringstream output;
    bool first = true;
    for (const auto& item : counts) {
        if (!first) {
            output << ", ";
        }
        output << item.first << '=' << item.second;
        first = false;
    }
    return output.str();
}

std::string format_error_counts(const std::vector<BenchmarkRequestResult>& results) {
    std::map<std::string, uint64_t> counts;
    for (const BenchmarkRequestResult& result : results) {
        if (!result.error.empty()) {
            ++counts[result.error];
        }
    }
    return format_finish_reason_counts(counts);
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

std::string truncate_prompt_to_tokens(Tokenizer* tokenizer, const std::string& prompt, int target_tokens) {
    std::vector<int> tokens = tokenizer->encode(prompt);
    const size_t target = static_cast<size_t>(std::max(1, target_tokens));
    if (tokens.size() <= target) {
        return prompt;
    }
    tokens.resize(target);
    std::string truncated = tokenizer->decode(tokens);
    while (tokenizer->encode(truncated).size() > target && !tokens.empty()) {
        tokens.pop_back();
        truncated = tokenizer->decode(tokens);
    }
    return truncated;
}

void validate_prompt_context_fit(const std::vector<BenchmarkRequestTemplate>& prompts, int context_tokens) {
    for (size_t index = 0; index < prompts.size(); ++index) {
        if (prompts[index].prompt_tokens >= static_cast<uint64_t>(context_tokens)) {
            std::ostringstream message;
            message << "benchmark prompt " << index
                    << " has " << prompts[index].prompt_tokens
                    << " tokens, which does not fit context limit "
                    << context_tokens;
            throw std::invalid_argument(message.str());
        }
    }
}

BenchmarkRequestTemplate fit_synthetic_prompt_to_context(
    ModelRuntime* runtime,
    Tokenizer* tokenizer,
    std::string prompt,
    int template_max_tokens,
    int request_max_tokens) {
    const int context_tokens = runtime->config().limits.context_tokens;
    int raw_target = static_cast<int>(tokenizer->encode(prompt).size());

    while (true) {
        BenchmarkRequestTemplate request_template{prompt, template_max_tokens, 0};
        const size_t wrapped_tokens =
            runtime->request_prompt_token_count(make_request(request_template, request_max_tokens));
        if (wrapped_tokens < static_cast<size_t>(context_tokens)) {
            request_template.prompt_tokens = static_cast<uint64_t>(wrapped_tokens);
            return request_template;
        }

        if (raw_target <= 1) {
            throw std::invalid_argument(
                "synthetic benchmark prompt cannot fit runtime chat template within context limit");
        }
        --raw_target;
        prompt = truncate_prompt_to_tokens(tokenizer, prompt, raw_target);
    }
}

GenerationRequest make_request(
    const BenchmarkRequestTemplate& request_template,
    int request_max_tokens,
    bool enable_stop_sequences,
    bool ignore_eos) {
    GenerationRequest request = make_request(request_template, request_max_tokens);
    request.options.enable_stop_sequences = enable_stop_sequences;
    request.options.ignore_eos = ignore_eos;
    return request;
}

std::vector<BenchmarkRequestTemplate> build_synthetic_prompts(
    ModelRuntime* runtime,
    const BenchmarkConfig& config) {
    Tokenizer tokenizer = runtime->tokenizer();
    PromptGenerator generator(&tokenizer, config.traffic.random_seed);
    LengthSampler length_sampler(config.traffic.random_seed ^ 0x9e3779b97f4a7c15ull);
    std::mt19937 output_rng(static_cast<std::mt19937::result_type>(
        config.traffic.random_seed ^ 0xbf58476d1ce4e5b9ull));

    const size_t prompt_count = std::max<size_t>(1, config.workload.synthetic_prompt_count);
    std::vector<BenchmarkRequestTemplate> prompts;
    prompts.reserve(prompt_count);

    const int min_output_tokens = std::max(1, config.workload.min_output_tokens);
    const int max_output_tokens = std::max(min_output_tokens, config.workload.max_output_tokens);
    std::uniform_int_distribution<int> output_distribution(min_output_tokens, max_output_tokens);

    for (size_t index = 0; index < prompt_count; ++index) {
        const PromptCategory category = config.workload.synthetic_mixed_categories
            ? category_for_index(index)
            : config.workload.synthetic_category;
        const PromptLengthClass length_class = config.workload.synthetic_mixed_lengths
            ? length_class_for_index(index)
            : config.workload.synthetic_length_class;
        const int target_tokens = config.workload.synthetic_mixed_lengths
            ? length_sampler.sample(length_class)
            : std::max(1, config.workload.input_tokens);
        const int max_tokens = config.workload.synthetic_random_output_tokens
            ? output_distribution(output_rng)
            : 0;
        prompts.push_back(fit_synthetic_prompt_to_context(
            runtime,
            &tokenizer,
            generator.generate(target_tokens, category),
            max_tokens,
            config.workload.output_tokens));
    }

    return prompts;
}

struct SubmittedRequest {
    size_t request_index = 0;
    size_t prompt_index = 0;
    uint64_t prompt_tokens = 0;
    SteadyClock::time_point submit_time{};
    std::shared_ptr<GenerationStream> stream;
};

struct GpuMonitorStats {
    bool available = false;
    uint64_t sample_count = 0;
    uint64_t gpu_util_total = 0;
    uint32_t gpu_util_peak = 0;
    uint64_t gpu_memory_bw_util_total = 0;
    uint32_t gpu_memory_bw_util_peak = 0;
    double gpu_vram_util_total_percent = 0.0;
    double gpu_vram_util_peak_percent = 0.0;
};

#if defined(NANO_LLM_ENABLE_NVML)
bool get_cuda_visible_nvml_device(nvmlDevice_t* device) {
#if defined(NANO_LLM_ENABLE_CUDA)
    if (device == nullptr) {
        return false;
    }

    ggml_backend_reg_t cuda_reg = ggml_backend_cuda_reg();
    if (cuda_reg == nullptr || ggml_backend_reg_dev_count(cuda_reg) == 0) {
        return false;
    }

    ggml_backend_dev_t cuda_device = ggml_backend_reg_dev_get(cuda_reg, 0);
    if (cuda_device == nullptr) {
        return false;
    }

    ggml_backend_dev_props props{};
    ggml_backend_dev_get_props(cuda_device, &props);
    if (props.device_id == nullptr || props.device_id[0] == '\0') {
        return false;
    }

    return nvmlDeviceGetHandleByPciBusId_v2(props.device_id, device) == NVML_SUCCESS;
#else
    (void)device;
    return false;
#endif
}
#endif

class GpuMonitor {
public:
    GpuMonitor() {
#if defined(NANO_LLM_ENABLE_NVML)
        // NVML is initialized once for this benchmark monitor and shut down
        // in the destructor after the sampler has stopped.
        if (nvmlInit_v2() != NVML_SUCCESS) {
            return;
        }
        nvml_initialized_ = true;
        if (!get_cuda_visible_nvml_device(&device_)) {
            return;
        }
        available_ = true;
#endif
    }

    ~GpuMonitor() {
        stop();
#if defined(NANO_LLM_ENABLE_NVML)
        if (nvml_initialized_) {
            nvmlShutdown();
        }
#endif
    }

    bool available() const {
        return available_;
    }

    void start() {
        if (!available_ || running_) {
            return;
        }
        // The sampler thread starts immediately before measured benchmark work
        // and is joined as soon as that measured work is drained.
        running_ = true;
        sampler_ = std::thread([this]() {
            while (running_) {
                sample_once();
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        });
    }

    void stop() {
        if (!running_) {
            return;
        }
        running_ = false;
        if (sampler_.joinable()) {
            sampler_.join();
        }
    }

    GpuMonitorStats stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        GpuMonitorStats snapshot = stats_;
        snapshot.available = available_ && snapshot.sample_count > 0;
        return snapshot;
    }

private:
    void sample_once() {
#if defined(NANO_LLM_ENABLE_NVML)
        nvmlUtilization_t utilization{};
        nvmlMemory_t memory{};
        if (nvmlDeviceGetUtilizationRates(device_, &utilization) != NVML_SUCCESS ||
            nvmlDeviceGetMemoryInfo(device_, &memory) != NVML_SUCCESS) {
            return;
        }

        const double vram_util_percent =
            memory.total > 0 ? (static_cast<double>(memory.used) * 100.0 / static_cast<double>(memory.total)) : 0.0;
        std::lock_guard<std::mutex> lock(mutex_);
        // Keep metric collection low-overhead by storing totals and peaks
        // instead of retaining every raw 100 ms sample.
        ++stats_.sample_count;
        stats_.gpu_util_total += utilization.gpu;
        stats_.gpu_util_peak = std::max(stats_.gpu_util_peak, utilization.gpu);
        stats_.gpu_memory_bw_util_total += utilization.memory;
        stats_.gpu_memory_bw_util_peak = std::max(stats_.gpu_memory_bw_util_peak, utilization.memory);
        stats_.gpu_vram_util_total_percent += vram_util_percent;
        stats_.gpu_vram_util_peak_percent = std::max(stats_.gpu_vram_util_peak_percent, vram_util_percent);
#endif
    }

    bool available_ = false;
    bool nvml_initialized_ = false;
    std::atomic<bool> running_{false};
    std::thread sampler_;
    mutable std::mutex mutex_;
    GpuMonitorStats stats_;
#if defined(NANO_LLM_ENABLE_NVML)
    nvmlDevice_t device_{};
#endif
};

BenchmarkRequestResult collect_stream_result(const SubmittedRequest& submitted) {
    BenchmarkRequestTrace trace;
    trace.submit_time = submitted.stream->submit_time();
    if (trace.submit_time == SteadyClock::time_point{}) {
        trace.submit_time = submitted.submit_time;
    }

    GenerationEvent event;
    SteadyClock::time_point event_time{};
    while (submitted.stream->wait_next_timed(&event, &event_time)) {
        if (trace.admit_time == SteadyClock::time_point{}) {
            trace.admit_time = submitted.stream->admit_time();
        }
        trace.record_event(event, event_time);
    }
    if (trace.admit_time == SteadyClock::time_point{}) {
        trace.admit_time = submitted.stream->admit_time();
    }
    if (trace.completion_time == SteadyClock::time_point{}) {
        trace.completion_time = submitted.stream->close_time();
    }
    if (trace.completion_time == SteadyClock::time_point{}) {
        trace.completion_time = SteadyClock::now();
    }
    return trace.finalize(submitted.request_index, submitted.prompt_index, submitted.prompt_tokens);
}

void collect_one_inflight(
    std::vector<SubmittedRequest>* inflight,
    std::vector<BenchmarkRequestResult>* completed_results) {
    completed_results->push_back(collect_stream_result(inflight->front()));
    inflight->erase(inflight->begin());
}

void drain_all(
    std::vector<SubmittedRequest>* inflight,
    std::vector<BenchmarkRequestResult>* completed_results) {
    while (!inflight->empty()) {
        collect_one_inflight(inflight, completed_results);
    }
}

BenchmarkSummary summarize_results(
    const BenchmarkConfig& config,
    const RuntimeConfig& runtime_config,
    double avg_prompt_tokens,
    int effective_max_decode_batch_size,
    double avg_batch_size,
    const GpuMonitorStats& gpu_stats,
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
    std::vector<double> queues;
    std::vector<double> service_ttfts;
    std::vector<double> tpots;
    latencies.reserve(summary.request_results.size());
    ttfts.reserve(summary.request_results.size());
    queues.reserve(summary.request_results.size());
    service_ttfts.reserve(summary.request_results.size());
    tpots.reserve(summary.request_results.size());

    for (const BenchmarkRequestResult& result : summary.request_results) {
        if (result.completed) {
            ++summary.completed_requests;
            summary.total_tokens_generated += result.tokens_generated;
            summary.total_prompt_tokens += result.prompt_tokens;
            const std::string finish_reason = result.finish_reason.empty() ? "unknown" : result.finish_reason;
            ++summary.finish_reason_counts[finish_reason];
            if (finish_reason == "length") {
                ++summary.length_limited_requests;
            } else {
                ++summary.early_stopped_requests;
            }
            if (result.has_latency) {
                latencies.push_back(result.latency_ms);
            }
            if (result.has_ttft) {
                ttfts.push_back(result.ttft_ms);
            }
            if (result.has_queue) {
                queues.push_back(result.queue_ms);
            }
            if (result.has_service_ttft) {
                service_ttfts.push_back(result.service_ttft_ms);
            }
            if (result.has_tpot) {
                tpots.push_back(result.tpot_ms);
            }
        }
    }
    summary.total_model_tokens = summary.total_prompt_tokens + summary.total_tokens_generated;
    if (summary.completed_requests > 0) {
        summary.avg_output_tokens =
            static_cast<double>(summary.total_tokens_generated) / static_cast<double>(summary.completed_requests);
    }

    if (measurement_end > measurement_start) {
        summary.total_runtime_ms = duration_ms(measurement_end - measurement_start);
        const double runtime_seconds = summary.total_runtime_ms / 1000.0;
        if (runtime_seconds > 0.0) {
            summary.throughput_tokens_per_sec =
                static_cast<double>(summary.total_tokens_generated) / runtime_seconds;
            summary.total_model_tokens_per_sec =
                static_cast<double>(summary.total_model_tokens) / runtime_seconds;
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
    summary.avg_queue_ms = mean_or_zero(queues);
    summary.p50_queue_ms = benchmark_percentile(queues, 50.0);
    summary.p95_queue_ms = benchmark_percentile(queues, 95.0);
    summary.avg_service_ttft_ms = mean_or_zero(service_ttfts);
    summary.p50_service_ttft_ms = benchmark_percentile(service_ttfts, 50.0);
    summary.p95_service_ttft_ms = benchmark_percentile(service_ttfts, 95.0);
    summary.avg_tpot_ms = mean_or_zero(tpots);

    // Aggregate GPU stats from the sampler's compact counters after the
    // measured phase finishes.
    summary.gpu_monitoring_available = gpu_stats.available && gpu_stats.sample_count > 0;
    if (summary.gpu_monitoring_available) {
        summary.gpu_util_avg_percent =
            static_cast<double>(gpu_stats.gpu_util_total) / static_cast<double>(gpu_stats.sample_count);
        summary.gpu_util_peak_percent = gpu_stats.gpu_util_peak;
        summary.gpu_memory_bw_util_avg_percent =
            static_cast<double>(gpu_stats.gpu_memory_bw_util_total) / static_cast<double>(gpu_stats.sample_count);
        summary.gpu_memory_bw_util_peak_percent = gpu_stats.gpu_memory_bw_util_peak;
        summary.gpu_vram_avg_percent =
            gpu_stats.gpu_vram_util_total_percent / static_cast<double>(gpu_stats.sample_count);
        summary.gpu_vram_peak_percent = gpu_stats.gpu_vram_util_peak_percent;
    }
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
            collect_one_inflight(&inflight, &completed);
        }
        const size_t prompt_index = static_cast<size_t>(request_index) % prompts.size();
        const SteadyClock::time_point submit_time = SteadyClock::now();
        auto stream = service->submit(make_request(
            prompts[prompt_index],
            config.workload.output_tokens,
            config.workload.enable_stop_sequences,
            config.workload.ignore_eos));
        inflight.push_back(SubmittedRequest{
            static_cast<size_t>(request_index),
            prompt_index,
            prompts[prompt_index].prompt_tokens,
            submit_time,
            std::move(stream),
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
    GpuMonitor gpu_monitor;
    gpu_monitor.start();

    try {
        for (size_t request_index = 0; request_index < config.traffic.request_count; ++request_index) {
            if (config.traffic.mode == BenchmarkMode::kOnlineServing) {
                const auto target = measurement_clock_start +
                    std::chrono::duration_cast<SteadyClock::duration>(
                        std::chrono::duration<double>(schedule_seconds[request_index]));
                const auto guard = std::chrono::microseconds(300);
                while (SteadyClock::now() + guard < target) {
                    std::this_thread::sleep_until(target - guard);
                }
                while (SteadyClock::now() < target) {
                    std::this_thread::yield();
                }
            }

            while (inflight.size() >= config.traffic.concurrency) {
                collect_one_inflight(&inflight, &completed_results);
            }

            const size_t prompt_index = request_index % prompts.size();
            const SteadyClock::time_point submit_time = SteadyClock::now();
            auto stream = service->submit(make_request(
                prompts[prompt_index],
                config.workload.output_tokens,
                config.workload.enable_stop_sequences,
                config.workload.ignore_eos));
            if (request_index == 0) {
                first_submit_time = submit_time;
            }
            inflight.push_back(SubmittedRequest{
                request_index,
                prompt_index,
                prompts[prompt_index].prompt_tokens,
                submit_time,
                std::move(stream),
            });
        }
        drain_all(&inflight, &completed_results);
        last_completion_time = SteadyClock::now();
        gpu_monitor.stop();
    } catch (...) {
        gpu_monitor.stop();
        throw;
    }
    const GpuMonitorStats gpu_stats = gpu_monitor.stats();
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
        gpu_stats,
        first_submit_time == SteadyClock::time_point{} ? measurement_clock_start : first_submit_time,
        last_completion_time,
        std::move(completed_results));
}

}  // namespace

PromptGenerator::PromptGenerator(Tokenizer* tokenizer, uint64_t seed)
    : tokenizer_(tokenizer),
      rng_(static_cast<std::mt19937::result_type>(seed)) {
    if (tokenizer_ == nullptr) {
        throw std::invalid_argument("PromptGenerator requires a non-null tokenizer");
    }
}

std::string PromptGenerator::generate(int target_tokens, PromptCategory category) {
    const std::vector<std::string>& templates = prompt_templates_for_category(category);
    std::uniform_int_distribution<size_t> template_distribution(0, templates.size() - 1);
    std::uniform_int_distribution<size_t> filler_distribution(0, filler_paragraphs().size() - 1);
    std::uniform_int_distribution<size_t> constraint_distribution(0, prompt_constraints().size() - 1);
    std::uniform_int_distribution<size_t> scenario_distribution(0, prompt_scenarios().size() - 1);

    // Prompt diversity improves benchmark quality because repeated identical prompts can hide
    // realistic variation in prefill cost, batching behavior, and cache behavior.
    const std::string base_template = templates[template_distribution(rng_)];
    std::string prompt = base_template;
    prompt += "\n";
    prompt += prompt_scenarios()[scenario_distribution(rng_)];
    prompt += "\n";
    prompt += prompt_constraints()[constraint_distribution(rng_)];

    const int target = std::max(1, target_tokens);
    int token_count = static_cast<int>(tokenizer_->encode(prompt).size());
    if (token_count > target) {
        prompt = base_template;
        token_count = static_cast<int>(tokenizer_->encode(prompt).size());
        if (token_count > target) {
            prompt = truncate_prompt_to_tokens(tokenizer_, prompt, target);
            token_count = static_cast<int>(tokenizer_->encode(prompt).size());
        }
    }

    if (token_count < target) {
        prompt += "\nAdditional context:\n";
    }

    // Token-based length control matters because model prefill, KV-cache usage, and context
    // limits are determined by tokenizer output, not by bytes or characters.
    size_t guard = 0;
    while (token_count < target && prompt.size() < 1024 * 1024 && guard < 4096) {
        prompt += filler_paragraphs()[filler_distribution(rng_)];
        prompt += "\n\n";
        token_count = static_cast<int>(tokenizer_->encode(prompt).size());
        ++guard;
    }
    if (token_count > target) {
        prompt = truncate_prompt_to_tokens(tokenizer_, prompt, target);
    }

    return prompt;
}

void BenchmarkRequestTrace::record_event(const GenerationEvent& event, std::chrono::steady_clock::time_point now) {
    if (event.token_generated) {
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
        completion_time = now;
    }
    if (!event.error.empty()) {
        saw_terminal = true;
        error = event.error;
        completion_time = now;
    }
}

BenchmarkRequestResult BenchmarkRequestTrace::finalize(
    size_t request_index,
    size_t prompt_index,
    uint64_t prompt_tokens) const {
    BenchmarkRequestResult result;
    result.request_index = request_index;
    result.prompt_index = prompt_index;
    result.prompt_tokens = prompt_tokens;
    result.tokens_generated = tokens_generated;
    result.finish_reason = finish_reason;
    result.error = error;
    result.completed = saw_terminal && error.empty();

    if (completion_time != SteadyClock::time_point{} && completion_time >= submit_time) {
        result.latency_ms = duration_ms(completion_time - submit_time);
        result.has_latency = true;
    }
    if (admit_time != SteadyClock::time_point{} && admit_time >= submit_time) {
        result.queue_ms = duration_ms(admit_time - submit_time);
        result.has_queue = true;
    }
    if (saw_first_token && first_token_time >= submit_time) {
        result.ttft_ms = duration_ms(first_token_time - submit_time);
        result.has_ttft = true;
        if (result.has_queue && first_token_time >= admit_time) {
            result.service_ttft_ms = duration_ms(first_token_time - admit_time);
            result.has_service_ttft = true;
        }
        if (inter_token_interval_count > 0) {
            result.tpot_ms = duration_ms(inter_token_duration) /
                static_cast<double>(inter_token_interval_count);
            result.has_tpot = true;
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
        prompts = build_synthetic_prompts(runtime_, config);
    } else {
        prompts.reserve(config.workload.prompts.size());
        for (const std::string& prompt : config.workload.prompts) {
            prompts.push_back(BenchmarkRequestTemplate{prompt, 0, 0});
        }
    }
    if (prompts.empty()) {
        throw std::invalid_argument("benchmark requires at least one prompt");
    }

    const int effective_max_decode_batch_size = effective_decode_batch_size(config);
    double total_prompt_tokens = 0.0;
    for (BenchmarkRequestTemplate& prompt : prompts) {
        prompt.prompt_tokens =
            runtime_->request_prompt_token_count(make_request(prompt, config.workload.output_tokens));
        total_prompt_tokens += static_cast<double>(prompt.prompt_tokens);
    }
    validate_prompt_context_fit(prompts, runtime_->config().limits.context_tokens);
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
            // Online serving: realistic chat workload with queuing pressure.
            // 256 input / 128 output matches typical assistant turn lengths.
            // 20 QPS with concurrency=4 creates meaningful queue delay at
            // ~600 tok/s aggregate throughput on this system.
            config.traffic.mode = BenchmarkMode::kOnlineServing;
            config.traffic.request_count = 40;
            config.traffic.concurrency = 4;
            config.workload.input_tokens = 256;
            config.workload.output_tokens = 128;
            config.workload.synthetic_prompt_count = 8;
            config.workload.synthetic_category = PromptCategory::Chat;
            config.workload.synthetic_length_class = PromptLengthClass::Medium;
            config.traffic.request_rate_qps = 20.0;
            break;
        case BenchmarkPreset::kThroughput:
            // Offline throughput: saturate the decode batch at 128/128 tokens.
            // concurrency=12 fills the batch; 96 requests gives stable stats
            // over 8 full batch rounds.
            config.traffic.mode = BenchmarkMode::kOfflineThroughput;
            config.traffic.request_count = 96;
            config.traffic.concurrency = 16;
            config.workload.input_tokens = 128;
            config.workload.output_tokens = 128;
            config.workload.enable_stop_sequences = false;
            config.workload.ignore_eos = true;
            config.workload.synthetic_prompt_count = config.traffic.request_count;
            config.workload.synthetic_mixed_lengths = false;
            config.workload.synthetic_mixed_categories = true;
            config.workload.synthetic_random_output_tokens = true;
            config.workload.min_output_tokens = 96;
            config.workload.max_output_tokens = 160;
            config.serving.decode_batch_size = 16;
            config.traffic.request_rate_qps = 0.0;
            break;
        case BenchmarkPreset::kLatency:
            // Single-request latency: no batching, 32 input / 16 output gives
            // a real TTFT measurement and enough decode steps for stable TPOT.
            config.traffic.mode = BenchmarkMode::kOfflineThroughput;
            config.traffic.request_count = 8;
            config.traffic.concurrency = 1;
            config.workload.input_tokens = 32;
            config.workload.output_tokens = 16;
            config.workload.synthetic_prompt_count = 1;
            config.workload.synthetic_category = PromptCategory::Chat;
            config.workload.synthetic_length_class = PromptLengthClass::Short;
            config.traffic.request_rate_qps = 0.0;
            break;
        case BenchmarkPreset::kLongContext:
            // Long-context: large prompt with short output to stress prefill
            // and KV cache fill.
            config.traffic.mode = BenchmarkMode::kOfflineThroughput;
            config.traffic.request_count = 8;
            config.traffic.concurrency = 1;
            config.workload.input_tokens = 800;
            config.workload.output_tokens = 32;
            config.workload.synthetic_prompt_count = 4;
            config.workload.synthetic_category = PromptCategory::Reasoning;
            config.workload.synthetic_length_class = PromptLengthClass::Long;
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
            << "requests_per_sec,avg_ttft_ms,avg_tpot_ms,"
            << "avg_latency_ms,avg_batch_size,total_tokens,total_runtime_ms,completed_requests,"
            << "total_prompt_tokens,total_model_tokens,total_model_tokens_per_sec,"
            << "gpu_monitoring_available,gpu_vram_peak_percent\n";
    }

    output << csv_escape(benchmark_preset_to_string(summary.preset)) << ','
           << csv_escape(benchmark_mode_to_string(summary.mode)) << ','
           << summary.request_count << ','
           << summary.concurrency << ','
           << std::fixed << std::setprecision(2)
           << summary.request_rate_qps << ','
           << summary.random_seed << ','
           << summary.warmup_requests << ','
           << summary.compute_threads << ','
           << summary.context_tokens << ','
           << summary.avg_input_tokens << ','
           << summary.output_tokens << ','
           << summary.decode_batch_size << ','
           << summary.requests_per_sec << ','
           << summary.avg_ttft_ms << ','
           << summary.avg_tpot_ms << ','
           << summary.avg_latency_ms << ','
           << summary.avg_batch_size << ','
           << summary.total_tokens_generated << ','
           << summary.total_runtime_ms << ','
           << summary.completed_requests << ','
           << summary.total_prompt_tokens << ','
           << summary.total_model_tokens << ','
           << summary.total_model_tokens_per_sec << ','
           << (summary.gpu_monitoring_available ? 1 : 0) << ',';
    if (summary.gpu_monitoring_available) {
        output << summary.gpu_vram_peak_percent;
    } else {
        output << "";
    }
    output << '\n';
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

    append_summary_line(&output, "Total Model Throughput", format_decimal(summary.total_model_tokens_per_sec, 1) + " tok/s");
    append_summary_line(&output, "Request Throughput", format_decimal(summary.requests_per_sec, 3) + " req/s");
    output << '\n';
    append_summary_line(&output, "Total Prompt Tokens", std::to_string(summary.total_prompt_tokens));
    append_summary_line(&output, "Total Output Tokens", std::to_string(summary.total_tokens_generated));
    append_summary_line(&output, "Total Model Tokens", std::to_string(summary.total_model_tokens));
    append_summary_line(&output, "Avg Output Tokens", format_decimal(summary.avg_output_tokens, 1) + " tokens");
    output << '\n';
    append_summary_line(&output, "Length-Limited Requests", std::to_string(summary.length_limited_requests));
    append_summary_line(&output, "Early-Stop Requests", std::to_string(summary.early_stopped_requests));
    append_summary_line(&output, "Finish Reasons", format_finish_reason_counts(summary.finish_reason_counts));
    output << '\n';
    append_summary_line(&output, "Total Runtime", format_latency_value(summary.total_runtime_ms));
    output << '\n';

    append_summary_line(&output, "Avg TTFT", format_ms_value(summary.avg_ttft_ms));
    append_summary_line(&output, "Avg TPOT", format_ms_value(summary.avg_tpot_ms));
    output << '\n';

    append_summary_line(&output, "Avg Latency", format_latency_value(summary.avg_latency_ms));
    output << '\n';

    append_summary_line(&output, "Avg Batch Size", format_decimal(summary.avg_batch_size, 1));
    output << '\n';

    // GPU metrics are reported after normal benchmark stats so they read as a
    // lightweight hardware summary for the same measured window.
    if (summary.gpu_monitoring_available) {
        append_summary_line(&output, "GPU VRAM Peak", format_decimal(summary.gpu_vram_peak_percent, 1) + "%");
    } else {
        append_summary_line(&output, "GPU Monitoring", "unavailable");
    }
    output << '\n';

    append_summary_line(&output, "Completed Requests", std::to_string(summary.completed_requests));
    const size_t failed_requests =
        summary.request_results.size() >= summary.completed_requests
            ? summary.request_results.size() - summary.completed_requests
            : 0;
    if (failed_requests > 0) {
        append_summary_line(&output, "Failed Requests", std::to_string(failed_requests));
        append_summary_line(&output, "Errors", format_error_counts(summary.request_results));
    }
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
