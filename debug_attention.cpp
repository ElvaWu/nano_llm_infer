#include <iostream>
#include <vector>
#include <cmath>
#include "gguf_loader.h"
#include "tokenizer.h"
#include "transformer.h"
#include "ggml-cpu.h"

void print_tensor_sample(const char* name, ggml_tensor* t, int max_vals = 5) {
    std::cerr << name << " shape=[";
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (i > 0) std::cerr << ", ";
        std::cerr << t->ne[i];
    }
    std::cerr << "] type=" << t->type << " data_sample=[";

    if (t->type == GGML_TYPE_F32) {
        float* data = (float*)t->data;
        for (int i = 0; i < std::min(max_vals, (int)ggml_nelements(t)); i++) {
            if (i > 0) std::cerr << ", ";
            std::cerr << data[i];
        }
    } else if (t->type == GGML_TYPE_I32) {
        int32_t* data = (int32_t*)t->data;
        for (int i = 0; i < std::min(max_vals, (int)ggml_nelements(t)); i++) {
            if (i > 0) std::cerr << ", ";
            std::cerr << data[i];
        }
    }
    std::cerr << "]\n";
}

int main() {
    nano_llm_infer::GGUFLoader loader;
    nano_llm_infer::Model model = loader.load_model("model_file/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");

    std::cerr << "Model: n_layers=" << model.n_layers
              << " n_heads=" << model.n_heads
              << " n_kv_heads=" << model.n_kv_heads << "\n";

    // Check if RoPE freq_base exists
    if (loader.has_metadata("llama.rope.freq_base")) {
        auto freq_base = loader.get_metadata<float>("llama.rope.freq_base");
        std::cerr << "RoPE freq_base=" << freq_base << "\n";
    }

    // Simple test: single token
    std::vector<int> tokens = {1}; // BOS

    ggml_context* compute_ctx = nullptr;
    const ggml_init_params compute_params = {
        512ull * 1024 * 1024, NULL, false
    };
    compute_ctx = ggml_init(compute_params);

    // Embed
    const int64_t n_tokens = tokens.size();
    ggml_tensor* token_tensor = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_I32, n_tokens);
    ((int32_t*)token_tensor->data)[0] = tokens[0];

    auto token_embd_weight = nano_llm_infer::get_tensor_by_name(model, "token_embd.weight");
    auto embeddings = ggml_get_rows(compute_ctx, token_embd_weight, token_tensor);

    std::cerr << "\n=== Before forward ===\n";
    print_tensor_sample("embeddings", embeddings);

    // Run one layer
    nano_llm_infer::TransformerBlock block(model, 0);
    block.set_compute_ctx(compute_ctx);
    auto output = block.forward(embeddings, 0);

    struct ggml_cgraph* gf = ggml_new_graph(compute_ctx);
    ggml_build_forward_expand(gf, output);
    ggml_graph_compute_with_ctx(compute_ctx, gf, 1);

    std::cerr << "\n=== After forward ===\n";
    print_tensor_sample("output", output);

    ggml_free(compute_ctx);
    return 0;
}
