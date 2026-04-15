#include <iostream>
#include <vector>
#include <algorithm>
#include "gguf_loader.h"
#include "tokenizer.h"
#include "transformer.h"
#include "ggml-cpu.h"

ggml_tensor* embedding_lookup(ggml_context* compute_ctx, nano_llm_infer::Model* model, const std::vector<int>& tokens){
    const int64_t n_tokens = static_cast<int64_t>(tokens.size());
    ggml_tensor * token_tensor = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_I32, n_tokens);
    int32_t* data = static_cast<int32_t*>(token_tensor->data);
    for (int64_t i = 0; i < n_tokens; ++i) {
        data[i] = tokens[static_cast<size_t>(i)];
    }
    auto token_embeddings = nano_llm_infer::get_tensor_by_name(*model, "token_embd.weight");
    return ggml_get_rows(compute_ctx, token_embeddings, token_tensor);
}

// Get logits without argmax
ggml_tensor* get_logits(ggml_context* ctx, ggml_tensor* hidden,
                        ggml_tensor* norm_weight, ggml_tensor* lm_head, float eps) {
    auto norm = ggml_rms_norm(ctx, hidden, eps);
    auto normed = ggml_mul(ctx, norm, norm_weight);
    return ggml_mul_mat(ctx, lm_head, normed);
}

int main() {
    nano_llm_infer::GGUFLoader loader;
    nano_llm_infer::Model model = loader.load_model("model_file/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");

    nano_llm_infer::Tokenizer tokenizer;
    tokenizer.load_from_gguf(loader);

    std::vector<int> tokens = {1, 379, 340, 263, 276, 366, 29973};

    std::cerr << "Input: ";
    for (int t : tokens) std::cerr << t << "(" << tokenizer.decode({t}) << ") ";
    std::cerr << "\n";

    ggml_context* compute_ctx = nullptr;
    const ggml_init_params compute_params = {
        512ull * 1024 * 1024, NULL, false
    };

    compute_ctx = ggml_init(compute_params);

    auto token_embeddings = embedding_lookup(compute_ctx, &model, tokens);

    // Run through transformer blocks manually
    ggml_tensor* x = token_embeddings;
    nano_llm_infer::TransformerBlock block0(model, 0);
