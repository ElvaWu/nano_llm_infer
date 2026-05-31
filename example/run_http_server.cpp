#include <cstdlib>
#include <iostream>
#include <string>

#include "generation.h"
#include "generation_server.h"

namespace {

int parse_int_arg(const char* value, const char* name) {
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0) {
        throw std::runtime_error(std::string("invalid value for ") + name);
    }
    return static_cast<int>(parsed);
}

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " [--model PATH] [--host IPv4] [--port N] "
              << "[--threads N] [--max-active N] [--max-decode-batch N] "
              << "[--max-tokens N] [--max-seq-len N]\n";
}

}  // namespace

int main(int argc, char** argv) {
    nano_llm_infer::RuntimeConfig runtime_config;
    nano_llm_infer::ServerConfig server_config;

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
                runtime_config.model.path = require_value("--model");
            } else if (arg == "--host") {
                server_config.host = require_value("--host");
            } else if (arg == "--port") {
                server_config.port = parse_int_arg(require_value("--port"), "--port");
            } else if (arg == "--threads") {
                runtime_config.model.compute_threads = parse_int_arg(require_value("--threads"), "--threads");
            } else if (arg == "--max-active") {
                server_config.max_active_requests = parse_int_arg(require_value("--max-active"), "--max-active");
            } else if (arg == "--max-decode-batch") {
                server_config.max_decode_batch_size =
                    parse_int_arg(require_value("--max-decode-batch"), "--max-decode-batch");
            } else if (arg == "--max-tokens") {
                runtime_config.limits.default_max_new_tokens = parse_int_arg(require_value("--max-tokens"), "--max-tokens");
            } else if (arg == "--max-seq-len") {
                runtime_config.limits.context_tokens = parse_int_arg(require_value("--max-seq-len"), "--max-seq-len");
            } else if (arg == "--help" || arg == "-h") {
                print_usage(argv[0]);
                return 0;
            } else {
                throw std::runtime_error("unknown argument: " + arg);
            }
        }

        if (server_config.max_decode_batch_size <= 0) {
            server_config.max_decode_batch_size = server_config.max_active_requests;
        }

        nano_llm_infer::ModelRuntime runtime(runtime_config);
        nano_llm_infer::HttpStreamingServer server(&runtime, server_config);
        server.run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
