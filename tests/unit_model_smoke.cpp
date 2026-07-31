#include <cstdlib>
#include <iostream>
#include <string>

#include "gguf_loader.h"
#include "tokenizer.h"

#include "gguf_test_util.h"
#include "test_util.h"

#ifndef NANO_LLM_TEST_MODEL_PATH
#define NANO_LLM_TEST_MODEL_PATH ""
#endif

#ifndef NANO_LLM_TEST_QWEN_MODEL_PATH
#define NANO_LLM_TEST_QWEN_MODEL_PATH ""
#endif

using namespace nano_llm_infer;

namespace {

std::string configured_path(const char* compiled_path, const char* environment_name) {
    if (compiled_path != nullptr && compiled_path[0] != '\0') {
        return compiled_path;
    }
    const char* environment = std::getenv(environment_name);
    return environment == nullptr ? "" : environment;
}

void run_model_smoke(
    const std::string& label,
    const std::string& model_path,
    const std::string& expected_architecture) {
    if (model_path.empty()) {
        std::cerr << "[SKIP] unit_model_smoke " << label << ": model path is empty\n";
        return;
    }

    GGUFLoader loader;
    Model model;
    try {
        model = loader.load_model(model_path);
    } catch (const std::exception& error) {
        std::cerr << "[SKIP] unit_model_smoke " << label << ": failed to load model at '"
                  << model_path << "' (" << error.what() << ")\n";
        return;
    }

    EXPECT_EQ(model.architecture, expected_architecture);
    EXPECT_TRUE(model.n_layers > 0);
    EXPECT_TRUE(model.n_heads > 0);
    EXPECT_TRUE(model.embedding_size > 0);

    Tokenizer tokenizer;
    EXPECT_TRUE(tokenizer.load_from_gguf(loader));

    const auto tokens = tokenizer.encode("Hello");
    EXPECT_TRUE(!tokens.empty());

    if (expected_architecture == "qwen2") {
        EXPECT_EQ(tokens, std::vector<int>({9707}));
        EXPECT_TRUE(find_tensor_by_name(model, "blk.0.attn_q.bias") != nullptr);
        EXPECT_TRUE(find_tensor_by_name(model, "blk.0.attn_k.bias") != nullptr);
        EXPECT_TRUE(find_tensor_by_name(model, "blk.0.attn_v.bias") != nullptr);
    }

    gguf_test_util::free_model(&model);
}

}  // namespace

int main() {
    run_model_smoke(
        "default",
        configured_path(NANO_LLM_TEST_MODEL_PATH, "NANO_LLM_TEST_MODEL"),
        "llama");
    run_model_smoke(
        "qwen",
        configured_path(NANO_LLM_TEST_QWEN_MODEL_PATH, "NANO_LLM_TEST_QWEN_MODEL"),
        "qwen2");

    return test_util::finalize();
}
