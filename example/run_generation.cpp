#include <iostream>
#include <stdexcept>
#include <string>

#include "generation.h"

namespace {

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " [--model PATH] [PROMPT]\n";
}

}  // namespace

int main(int argc, char** argv) {
    nano_llm_infer::RuntimeConfig config;
    std::string user_prompt = "Who are you?";

    try {
        bool has_prompt = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--model") {
                if (i + 1 >= argc) {
                    throw std::runtime_error("missing value for --model");
                }
                config.model.path = argv[++i];
            } else if (arg == "--help" || arg == "-h") {
                print_usage(argv[0]);
                return 0;
            } else if (!arg.empty() && arg[0] == '-') {
                throw std::runtime_error("unknown argument: " + arg);
            } else if (has_prompt) {
                throw std::runtime_error("only one positional prompt is supported");
            } else {
                user_prompt = arg;
                has_prompt = true;
            }
        }

        nano_llm_infer::ModelRuntime runtime(config);
        nano_llm_infer::GenerationRequest request;
        request.messages.push_back({"user", user_prompt});
        request.options.max_tokens = config.limits.default_max_new_tokens;

        std::cout << "Response: " << std::flush;
        runtime.generate_stream(request, [](const nano_llm_infer::GenerationEvent& event) {
            if (!event.error.empty()) {
                std::cerr << "\nError: " << event.error << '\n';
                return false;
            }
            if (!event.text.empty()) {
                std::cout << event.text << std::flush;
            }
            return event.finish_reason.empty();
        });
        std::cout << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
