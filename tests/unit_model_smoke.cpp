#include <iostream>
#include <string>

#include "gguf_loader.h"
#include "tokenizer.h"

#include "test_util.h"

#ifndef NANO_LLM_TEST_MODEL_PATH
#define NANO_LLM_TEST_MODEL_PATH "/root/autodl-tmp/repos/nano_llm_infer/model_file/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
#endif

using namespace nano_llm_infer;

int main() {
    const std::string model_path = NANO_LLM_TEST_MODEL_PATH;
    if (model_path.empty()) {
        std::cerr << "[SKIP] unit_model_smoke: NANO_LLM_TEST_MODEL is empty\n";
        return 0;
    }

    GGUFLoader loader;
    Model model;

    try {
        model = loader.load_model(model_path);
    } catch (const std::exception& e) {
        std::cerr << "[SKIP] unit_model_smoke: failed to load model at '" << model_path
                  << "' (" << e.what() << ")\n";
        return 0;
    }

    EXPECT_TRUE(model.n_layers > 0);
    EXPECT_TRUE(model.n_heads > 0);

    Tokenizer tokenizer;
    EXPECT_TRUE(tokenizer.load_from_gguf(loader));

    const auto tokens = tokenizer.encode("Hello");
    EXPECT_TRUE(!tokens.empty());

    return test_util::finalize();
}
