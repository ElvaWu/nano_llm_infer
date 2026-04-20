#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "ggml-cpu.h"
#include "gguf_loader.h"
#include "kv_cache.h"
#include "tokenizer.h"
#include "transformer.h"

namespace {

// constexpr const char* kModelPath = "model_file/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";
constexpr const char* kModelPath = "model_file/tinyllama-1.1b-chat-v1.0.Q6_K.gguf";
constexpr int   kComputeThreads = 4;
constexpr int   kMaxNewTokens   = 50;  // Reduced for faster testing
constexpr int   kMaxSeqLen      = 512;

ggml_tensor* embedding_lookup(
    ggml_context* compute_ctx,
    const nano_llm_infer::Model& model,
    const std::vector<int>& tokens) {
    const int64_t n_tokens = static_cast<int64_t>(tokens.size());
    ggml_tensor* token_tensor = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_I32, n_tokens);
    auto* data = static_cast<int32_t*>(token_tensor->data);
    for (int64_t i = 0; i < n_tokens; ++i) {
        data[i] = tokens[static_cast<size_t>(i)];
    }
    ggml_tensor* token_embeddings =
        nano_llm_infer::get_tensor_by_name(model, "token_embd.weight");
    return ggml_get_rows(compute_ctx, token_embeddings, token_tensor);
}

void ensure_valid_omp_threads() {
    const char* omp_env = std::getenv("OMP_NUM_THREADS");
    if (omp_env == nullptr || std::string(omp_env).empty() || std::atoi(omp_env) < 1) {
        setenv("OMP_NUM_THREADS", "4", 1);
    }
}

bool trim_stop_sequence(std::string* response) {
    const std::vector<std::string> stop_strings = {
        "<|assistant|>", "<|user|>", "<|system|>",
        "|assistant|",   "|user|",   "|system|",
    };
    for (const auto& stop : stop_strings) {
        if (response->size() >= stop.size() &&
            response->compare(response->size() - stop.size(), stop.size(), stop) == 0) {
            response->resize(response->size() - stop.size());
            return true;
        }
    }
    return false;
}

int select_argmax(const float* logits, int64_t vocab_size) {
    return static_cast<int>(std::max_element(logits, logits + vocab_size) - logits);
}

}  // namespace

int main(int argc, char** argv) {
    ensure_valid_omp_threads();

    nano_llm_infer::GGUFLoader loader;
    nano_llm_infer::Model model = loader.load_model(kModelPath);

    std::string user_prompt = "Who are you?";
    if (argc > 1) {
        user_prompt = argv[1];
    }

    nano_llm_infer::Tokenizer tokenizer;
    tokenizer.load_from_gguf(loader);

    const std::vector<nano_llm_infer::ChatMessage> messages = {
        {"user", user_prompt},
    };
    const std::string prompt = tokenizer.apply_chat_template(messages, true);
    std::cerr << "Prompt: " << prompt << '\n';

    std::vector<int> prompt_tokens = tokenizer.encode(prompt);
    prompt_tokens.insert(prompt_tokens.begin(), tokenizer.bos());
    const int n_prompt = static_cast<int>(prompt_tokens.size());
    std::cerr << "Prompt tokens: " << n_prompt << '\n';

    nano_llm_infer::Transformer transformer(model);
    nano_llm_infer::KVCache kv = nano_llm_infer::make_kv_cache(model, kMaxSeqLen);

    const int eos = tokenizer.eos();

    // -------------------------------------------------------------------------
    // Phase 1: Prefill — process all prompt tokens, populate KV cache
    // -------------------------------------------------------------------------
    int first_token = -1;
    {
        const ggml_init_params params = {
            /* mem_size   */ 512ull * 1024 * 1024,
            /* mem_buffer */ nullptr,
            /* no_alloc   */ false,
        };
        ggml_context* ctx = ggml_init(params);
        transformer.set_compute_ctx(ctx);

        ggml_tensor* embeds = embedding_lookup(ctx, model, prompt_tokens);
        ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_tensor* logits = transformer.forward_logits(embeds, /*pos=*/0, gf, &kv);
        ggml_build_forward_expand(gf, logits);
        ggml_graph_compute_with_ctx(ctx, gf, kComputeThreads);

        // Logits shape: [vocab_size, n_prompt] — pick the last position
        auto* logit_data = static_cast<float*>(logits->data);
        const int64_t vocab_size = logits->ne[0];
        const float* last = logit_data + (static_cast<int64_t>(n_prompt) - 1) * vocab_size;
        first_token = select_argmax(last, vocab_size);

        ggml_free(ctx);
    }
    kv.n_filled = n_prompt;

    // -------------------------------------------------------------------------
    // Phase 2: Decode — generate one token at a time using KV cache
    // -------------------------------------------------------------------------
    std::string response;
    std::cout << "Response: " << std::flush;

    int next_token = first_token;
    for (int step = 0; step < kMaxNewTokens; ++step) {
        if (next_token == eos) break;
        if (kv.n_filled >= kMaxSeqLen) break;

        std::string decoded = tokenizer.decode({next_token});
        response += decoded;
        std::cout << decoded << std::flush;

        if (trim_stop_sequence(&response)) break;

        const ggml_init_params params = {
            /* mem_size   */ 256ull * 1024 * 1024,
            /* mem_buffer */ nullptr,
            /* no_alloc   */ false,
        };
        ggml_context* ctx = ggml_init(params);
        transformer.set_compute_ctx(ctx);

        ggml_tensor* embed = embedding_lookup(ctx, model, {next_token});
        ggml_cgraph* gf = ggml_new_graph(ctx);

        // Write this token at position kv.n_filled, then read [0..kv.n_filled] for attention
        ggml_tensor* logits = transformer.forward_logits(embed, kv.n_filled, gf, &kv);
        ggml_build_forward_expand(gf, logits);
        ggml_graph_compute_with_ctx(ctx, gf, kComputeThreads);

        kv.n_filled++;  // Increment after writing but before next iteration

        auto* logit_data = static_cast<float*>(logits->data);
        const int64_t vocab_size = logits->ne[0];
        next_token = select_argmax(logit_data, vocab_size);

        ggml_free(ctx);
    }

    std::cout << response << '\n';
    nano_llm_infer::free_kv_cache(kv);
    return 0;
}
