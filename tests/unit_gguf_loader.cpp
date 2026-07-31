#include <string>

#include "gguf_loader.h"

#include "gguf_test_util.h"
#include "test_util.h"

using namespace nano_llm_infer;

int main() {
    GGUFLoader loader;

    EXPECT_THROW(loader.load_model("/tmp/this_file_should_not_exist.gguf"));
    EXPECT_TRUE(!loader.has_metadata("tokenizer.ggml.tokens"));
    EXPECT_THROW(loader.get_metadata<std::string>("missing.key"));

    {
        gguf_test_util::SyntheticGGUFSpec spec;
        const auto file = gguf_test_util::write_synthetic_gguf("llama_metadata", spec);
        Model model = loader.load_model(file.path);
        EXPECT_EQ(model.architecture, std::string("llama"));
        EXPECT_EQ(model.n_layers, static_cast<uint32_t>(1));
        EXPECT_EQ(model.n_heads, static_cast<uint32_t>(2));
        EXPECT_EQ(model.n_kv_heads, static_cast<uint32_t>(1));
        EXPECT_EQ(model.embedding_size, static_cast<uint32_t>(8));
        gguf_test_util::free_model(&model);
    }

    {
        gguf_test_util::SyntheticGGUFSpec spec;
        spec.architecture = "qwen2";
        spec.include_conflicting_llama_metadata = true;
        spec.include_attention_biases = true;
        const auto file = gguf_test_util::write_synthetic_gguf("qwen_metadata", spec);
        Model model = loader.load_model(file.path);
        EXPECT_EQ(model.architecture, std::string("qwen2"));
        EXPECT_EQ(model.embedding_size, static_cast<uint32_t>(8));
        EXPECT_TRUE(find_tensor_by_name(model, "blk.0.attn_q.bias") != nullptr);
        EXPECT_TRUE(find_tensor_by_name(model, "blk.0.attn_k.bias") != nullptr);
        EXPECT_TRUE(find_tensor_by_name(model, "blk.0.attn_v.bias") != nullptr);
        gguf_test_util::free_model(&model);
    }

    {
        gguf_test_util::SyntheticGGUFSpec spec;
        spec.architecture = "qwen2";
        const auto file = gguf_test_util::write_synthetic_gguf("qwen_no_bias", spec);
        Model model = loader.load_model(file.path);
        EXPECT_TRUE(find_tensor_by_name(model, "blk.0.attn_q.bias") == nullptr);
        EXPECT_TRUE(find_tensor_by_name(model, "blk.0.attn_k.bias") == nullptr);
        EXPECT_TRUE(find_tensor_by_name(model, "blk.0.attn_v.bias") == nullptr);
        gguf_test_util::free_model(&model);
    }

    {
        gguf_test_util::SyntheticGGUFSpec spec;
        spec.architecture = "qwen2";
        spec.include_embedding_length = false;
        const auto file = gguf_test_util::write_synthetic_gguf("missing_metadata", spec);
        EXPECT_THROW(loader.load_model(file.path));
    }

    {
        gguf_test_util::SyntheticGGUFSpec spec;
        spec.architecture = "gptneox";
        const auto file = gguf_test_util::write_synthetic_gguf("unsupported_arch", spec);
        EXPECT_THROW(loader.load_model(file.path));
    }

    {
        gguf_test_util::SyntheticGGUFSpec spec;
        const auto file = gguf_test_util::write_synthetic_gguf("truncated_tensor", spec);
        file.truncate_tensor_data();
        EXPECT_THROW(loader.load_model(file.path));
    }

    return test_util::finalize();
}
