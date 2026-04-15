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
ggml_tensor* get_logits_only(ggml_context* ctx, ggml_tensor* hidden,
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
    for (int t : tokens) std::cerr << tokenizer.decode({t});
    std::cerr << "\n";

    nano_llm_infer::Transformer transformer(model);

    ggml_context* compute_ctx = nullptr;
    const ggml_init_params compute_params = {
        512ull * 1024 * 1024, NULL, false
    };

    compute_ctx = ggml_init(compute_params);
    transformer.set_compute_ctx(compute_ctx);

    auto token_embeddings = embedding_lookup(compute_ctx, &model, tokens);

    // Run through transformer blocks to get hidden states
    ggml_tensor* x = token_embeddings;
    for (uint32_t i = 0; i < model.n_layers; ++i) {
        nano_llm_infer::TransformerBlock block(model, i);
        block.set_compute_ctx(compute_ctx);
        x = block.forward(x, 0);
    }

    // Get logits without argmax
    auto norm_weight = nano_llm_infer::get_tensor_by_name(model, "output_norm.weight");
    auto lm_head = nano_llm_infer::get_tensor_by_name(model, "output.weight");
    auto logits = get_logits_only(compute_ctx, x, norm_weight, lm_head, model.rms_norm_eps);

    // Build and execute graph
    struct ggml_cgraph* gf = ggml_new_graph(compute_ctx);
    ggml_build_forward_expand(gf, logits);
    ggml_graph_compute_with_ctx(compute_ctx, gf, 1);

    // Print top-10 logits for the last position
    int64_t vocab_size = logits->ne[0];
    int64_t seq_len = logits->ne[1];
    float* logits_data = (float*)logits->data;

    std::cerr << "\nLogits shape: [" << vocab_size << ", " << seq_len << "]\n";
    std::cerr << "\nTop 10 predictions for last position:\n";

    std::vector<std::pair<float, int>> scores;
    for (int i = 0; i < vocab_size; ++i) {
        float score = logits_data[i + (seq_len - 1) * vocab_size];
        scores.push_back({score, i});
    }

    std::sort(scores.begin(), scores.end(), std::greater<std::pair<float, int>>());

    for (int i = 0; i < 10; ++i) {
        int token_id = scores[i].second;
        float score = scores[i].first;
        std::string token_str = tokenizer.decode({token_id});
        std::cerr << i+1 << ". token=" << token_id << " score=" << score
                  << " decoded='" << token_str << "'\n";
    }

    ggml_free(compute_ctx);
    return 0;
}
