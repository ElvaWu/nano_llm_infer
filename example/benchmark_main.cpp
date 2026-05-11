#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>

#include "benchmark.h"
#include "generation.h"

namespace {

int parse_int_arg(const char* value, const char* name) {
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0) {
        throw std::runtime_error(std::string("invalid value for ") + name);
    }
    return static_cast<int>(parsed);
}

int parse_non_negative_int_arg(const char* value, const char* name) {
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed < 0) {
        throw std::runtime_error(std::string("invalid value for ") + name);
    }
    return static_cast<int>(parsed);
}

size_t parse_size_arg(const char* value, const char* name) {
    return static_cast<size_t>(parse_int_arg(value, name));
}

double parse_double_arg(const char* value, const char* name) {
    char* end = nullptr;
    const double parsed = std::strtod(value, &end);
    if (end == value || *end != '\0' || parsed < 0.0) {
        throw std::runtime_error(std::string("invalid value for ") + name);
    }
    return parsed;
}

uint64_t parse_u64_arg(const char* value, const char* name) {
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0') {
        throw std::runtime_error(std::string("invalid value for ") + name);
    }
    return static_cast<uint64_t>(parsed);
}

nano_llm_infer::BenchmarkMode parse_mode(const std::string& value) {
    if (value == "offline") {
        return nano_llm_infer::BenchmarkMode::kOfflineThroughput;
    }
    if (value == "online") {
        return nano_llm_infer::BenchmarkMode::kOnlineServing;
    }
    throw std::runtime_error("invalid value for --mode");
}

bool parse_toggle(const std::string& value, const char* name) {
    if (value == "on") {
        return true;
    }
    if (value == "off") {
        return false;
    }
    throw std::runtime_error(std::string("invalid value for ") + name);
}

void print_usage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0 << " [options]\n\n"
        << "Runtime:\n"
        << "  --model PATH              GGUF model path\n"
        << "  --threads N               ggml graph execution threads\n"
        << "  --context-tokens N        KV-cache context tokens\n\n"
        << "Benchmark:\n"
        << "  --preset chat|throughput|latency|long_context\n"
        << "  --mode offline|online\n\n"
        << "Traffic:\n"
        << "  --requests N              measured requests\n"
        << "  --concurrency N           client in-flight and service active request limit\n"
        << "  --request-rate QPS        online request rate; offline uses 0 for unlimited\n"
        << "  --warmup-requests N       requests excluded from measurement\n"
        << "  --seed N                  deterministic online arrival seed\n\n"
        << "Workload:\n"
        << "  --input-tokens N          target synthetic prompt tokens\n"
        << "  --output-tokens N         per-request generated token cap\n\n"
        << "Serving:\n"
        << "  --decode-batch-size N     0 auto, 1 disables decode batching\n\n"
        << "Output:\n"
        << "  --export-csv on|off       append summary metrics to CSV\n";
}

bool is_runtime_arg(const std::string& arg) {
    return arg == "--model" || arg == "--threads" || arg == "--context-tokens";
}

bool is_benchmark_arg(const std::string& arg) {
    return arg == "--preset" || arg == "--mode" || arg == "--requests" || arg == "--concurrency" ||
        arg == "--request-rate" || arg == "--warmup-requests" || arg == "--seed" ||
        arg == "--input-tokens" || arg == "--output-tokens" || arg == "--decode-batch-size" ||
        arg == "--export-csv";
}

}  // namespace

int main(int argc, char** argv) {
    nano_llm_infer::RuntimeConfig runtime_config;
    runtime_config.memory.prefill_ctx_bytes = 768ull * 1024 * 1024;
    runtime_config.memory.decode_ctx_bytes = 512ull * 1024 * 1024;
    nano_llm_infer::BenchmarkPreset preset = nano_llm_infer::BenchmarkPreset::kChat;

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                print_usage(argv[0]);
                return 0;
            }
            if (arg != "--preset") {
                if (is_runtime_arg(arg) || is_benchmark_arg(arg)) {
                    ++i;
                    continue;
                }
                throw std::runtime_error("unknown argument: " + arg);
            }
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for --preset");
            }
            preset = nano_llm_infer::parse_benchmark_preset(argv[++i]);
        }

        nano_llm_infer::BenchmarkConfig benchmark_config = nano_llm_infer::make_preset(preset);
        bool context_tokens_overridden = false;

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto require_value = [&](const char* name) -> const char* {
                if (i + 1 >= argc) {
                    throw std::runtime_error(std::string("missing value for ") + name);
                }
                return argv[++i];
            };

            if (arg == "--model") {
                runtime_config.model.path = require_value("--model");
            } else if (arg == "--threads") {
                runtime_config.model.compute_threads = parse_int_arg(require_value("--threads"), "--threads");
            } else if (arg == "--context-tokens") {
                runtime_config.limits.context_tokens = parse_int_arg(require_value("--context-tokens"), "--context-tokens");
                context_tokens_overridden = true;
            } else if (arg == "--preset") {
                require_value("--preset");
            } else if (arg == "--mode") {
                benchmark_config.traffic.mode = parse_mode(require_value("--mode"));
            } else if (arg == "--requests") {
                benchmark_config.traffic.request_count = parse_size_arg(require_value("--requests"), "--requests");
            } else if (arg == "--concurrency") {
                benchmark_config.traffic.concurrency = parse_size_arg(require_value("--concurrency"), "--concurrency");
            } else if (arg == "--request-rate") {
                benchmark_config.traffic.request_rate_qps =
                    parse_double_arg(require_value("--request-rate"), "--request-rate");
            } else if (arg == "--input-tokens") {
                benchmark_config.workload.input_tokens = parse_int_arg(require_value("--input-tokens"), "--input-tokens");
            } else if (arg == "--output-tokens") {
                benchmark_config.workload.output_tokens =
                    parse_int_arg(require_value("--output-tokens"), "--output-tokens");
            } else if (arg == "--decode-batch-size") {
                benchmark_config.serving.decode_batch_size =
                    parse_non_negative_int_arg(require_value("--decode-batch-size"), "--decode-batch-size");
            } else if (arg == "--warmup-requests") {
                benchmark_config.traffic.warmup_requests =
                    parse_non_negative_int_arg(require_value("--warmup-requests"), "--warmup-requests");
            } else if (arg == "--seed") {
                benchmark_config.traffic.random_seed = parse_u64_arg(require_value("--seed"), "--seed");
            } else if (arg == "--export-csv") {
                benchmark_config.output.export_csv = parse_toggle(require_value("--export-csv"), "--export-csv");
            } else if (arg == "--help" || arg == "-h") {
                print_usage(argv[0]);
                return 0;
            } else {
                throw std::runtime_error("unknown argument: " + arg);
            }
        }

        runtime_config.limits.default_max_new_tokens = benchmark_config.workload.output_tokens;
        if (!context_tokens_overridden) {
            runtime_config.limits.context_tokens =
                std::max(
                    runtime_config.limits.context_tokens,
                    benchmark_config.workload.input_tokens + benchmark_config.workload.output_tokens + 64);
        }

        // This benchmark targets GenerationService directly so scheduler and batching
        // behavior can be measured without HTTP serialization and network overhead.
        nano_llm_infer::ModelRuntime runtime(runtime_config);
        nano_llm_infer::BenchmarkRunner runner(&runtime);
        const nano_llm_infer::BenchmarkSummary summary = runner.run(benchmark_config);
        std::cout << nano_llm_infer::format_benchmark_summary(summary) << '\n';
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
