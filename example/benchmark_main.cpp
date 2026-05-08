#include <cstdlib>
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

size_t parse_size_arg(const char* value, const char* name) {
    return static_cast<size_t>(parse_int_arg(value, name));
}

double parse_double_arg(const char* value, const char* name) {
    char* end = nullptr;
    const double parsed = std::strtod(value, &end);
    if (end == value || *end != '\0' || parsed <= 0.0) {
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
    std::cerr << "Usage: " << argv0 << " [--model PATH] [--threads N] [--max-seq-len N] "
              << "[--mode offline|online] [--num-requests N] [--concurrency N] [--arrival-rate QPS] "
              << "[--max-active N] [--max-decode-batch N] [--dynamic-batching on|off] "
              << "[--scheduler-policy NAME] [--max-tokens N] [--seed N] [--export-csv on|off]\n";
}

}  // namespace

int main(int argc, char** argv) {
    nano_llm_infer::RuntimeConfig runtime_config;
    nano_llm_infer::BenchmarkConfig benchmark_config;

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto require_value = [&](const char* name) -> const char* {
                if (i + 1 >= argc) {
                    throw std::runtime_error(std::string("missing value for ") + name);
                }
                return argv[++i];
            };

            if (arg == "--model") {
                runtime_config.model_path = require_value("--model");
            } else if (arg == "--threads") {
                runtime_config.compute_threads = parse_int_arg(require_value("--threads"), "--threads");
            } else if (arg == "--max-seq-len") {
                runtime_config.max_seq_len = parse_int_arg(require_value("--max-seq-len"), "--max-seq-len");
            } else if (arg == "--mode") {
                benchmark_config.mode = parse_mode(require_value("--mode"));
            } else if (arg == "--num-requests") {
                benchmark_config.num_requests = parse_size_arg(require_value("--num-requests"), "--num-requests");
            } else if (arg == "--concurrency") {
                benchmark_config.max_concurrency = parse_size_arg(require_value("--concurrency"), "--concurrency");
            } else if (arg == "--arrival-rate") {
                benchmark_config.arrival_rate_qps = parse_double_arg(require_value("--arrival-rate"), "--arrival-rate");
            } else if (arg == "--max-active") {
                benchmark_config.max_active_requests = parse_int_arg(require_value("--max-active"), "--max-active");
            } else if (arg == "--max-decode-batch") {
                benchmark_config.max_decode_batch_size =
                    parse_int_arg(require_value("--max-decode-batch"), "--max-decode-batch");
            } else if (arg == "--dynamic-batching") {
                benchmark_config.dynamic_batching_enabled =
                    parse_toggle(require_value("--dynamic-batching"), "--dynamic-batching");
            } else if (arg == "--scheduler-policy") {
                benchmark_config.scheduler_policy = require_value("--scheduler-policy");
            } else if (arg == "--max-tokens") {
                benchmark_config.request_max_tokens = parse_int_arg(require_value("--max-tokens"), "--max-tokens");
                runtime_config.max_new_tokens = benchmark_config.request_max_tokens;
            } else if (arg == "--seed") {
                benchmark_config.random_seed = parse_u64_arg(require_value("--seed"), "--seed");
            } else if (arg == "--export-csv") {
                benchmark_config.export_csv = parse_toggle(require_value("--export-csv"), "--export-csv");
            } else if (arg == "--help" || arg == "-h") {
                print_usage(argv[0]);
                return 0;
            } else {
                throw std::runtime_error("unknown argument: " + arg);
            }
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
