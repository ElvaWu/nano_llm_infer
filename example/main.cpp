#include <iostream>
#include <string>

#include "generation.h"

int main(int argc, char** argv) {
    nano_llm_infer::RuntimeConfig config;
    nano_llm_infer::ModelRuntime runtime(config);

    std::string user_prompt = "Who are you?";
    if (argc > 1) {
        user_prompt = argv[1];
    }

    nano_llm_infer::GenerationRequest request;
    request.messages.push_back({"user", user_prompt});
    request.options.max_tokens = config.max_new_tokens;

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
    return 0;
}
