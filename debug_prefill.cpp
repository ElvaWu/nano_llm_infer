#include <algorithm>
#include <iostream>
#include <vector>
#include "gguf_loader.h"
#include "kv_cache.h"
#include "tokenizer.h"
#include "transformer.h"

int main() {
    constexpr const char* kModelPath = "model_file/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";
    constexpr int kMaxSeqLen = 512;

    nano_llm_infer::GGUFLoader loader;
    nano_llm_infer::Model model = loader.load_model(kModelPath);

    nano_llm_infer::Tokenizer tokenizer;
    tokenizer.load_from_gguf(loader);

    const std::vector<nano_llm_infer::ChatMessage> messages = {
        {"user", "Who are you?"},
    };
    const std::string prompt = tokenizer.apply_chat_template(messages, true);
    std::vector<int> prompt_tokens = tokenizer.encode(prompt);
    prompt_tokens.insert(prompt_tokens.begin(), tokenizer.bos());

    std::cerr << "Prompt: " << prompt << '\n';
    std::cerr << "Prompt tokens (" << prompt_tokens.size() << "): ";
    for (int tok : prompt_tokens) {
        std::cerr << tok << " ";
    }
    std::cerr << '\n';

    nano_llm_infer::Transformer transformer(model);
    nano_llm_infer::KVCache kv = nano_llm_infer::make_kv_cache(model, kMaxSeqLen);

    // Prefill
    const ggml_init_params params = {
        512ull * 1024 * 1024, nullptr, false,
    };
    ggml_context* ctx = ggml_init(params);
    transformer.set_compute_ctx(ctx);

    // Embedding lookup
    const int64_t n_tokens = static_cast<int64_t>(prompt_tokens.size());
    ggml_tensor* token_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    auto* data = static_cast<int32_t*>(token_tensor->data);
    for (int64_t i = 0; i < n_tokens; ++i) {
        data[i] = prompt_tokens[static_cast<size_t>(i)];
    }
    ggml_tensor* token_embeddings = nano_llm_infer::get_tensor_by_name(model, "token_embd.weight");
    ggml_tensor* embeds = ggml_get_rows(ctx, token_embeddings, token_tensor);

    std::cerr << "Embeds shape: [" << embeds->ne[0] << ", " << embeds->ne[1] << "]\n";

    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_tensor* logits = transformer.forward_logits(embeds, /*pos=*/0, gf, &kv);

    std::cerr << "Logits shape before compute: [" << logits->ne[0] << ", " << logits->ne[1] << "]\n";

    ggml_build_forward_expand(gf, logits);
    ggml_graph_compute_with_ctx(ctx, gf, 4);

    std::cerr << "Logits shape after compute: [" << logits->ne[0] << ", " << logits->ne[1] << "]\n";

    // Check logits for last position
    auto* logit_data = static_cast<float*>(logits->data);
    const int64_t vocab_size = logits->ne[0];
    const int64_t seq_len = logits->ne[1];

    std::cerr << "Vocab size: " << vocab_size << ", Seq len: " << seq_len << '\n';

    // Print top 10 logits for the last position
    const float* last = logit_data + (seq_len - 1) * vocab_size;
    std::vector<std::pair<int, float>> top_logits;
    for (int i = 0; i < vocab_size; ++i) {
        top_logits.push_back({i, last[i]});
    }
    std::sort(top_logits.begin(), top_logits.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    std::cerr << "\nTop 10 logits for last position:\n";
    for (int i = 0; i < 10; ++i) {
        std::cerr << "  tok=" << top_logits[i].first
                  << " logit=" << top_logits[i].second
                  << " dec=\"" << tokenizer.decode({top_logits[i].first}) << "\"\n";
    }

    ggml_free(ctx);
    nano_llm_infer::free_kv_cache(kv);
    return 0;
}
