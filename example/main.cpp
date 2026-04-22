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

struct RuntimeConfig {
    const char* model_path = "model_file/tinyllama-1.1b-chat-v1.0.Q6_K.gguf";
    int compute_threads = 4;
    int max_seq_len = 512;
    int max_new_tokens = 600;
    size_t prefill_ctx_bytes = 512ull * 1024 * 1024;
    size_t decode_ctx_bytes = 256ull * 1024 * 1024;
};

struct RuntimeState {
    nano_llm_infer::Transformer transformer;
    nano_llm_infer::KVCache kv;
    int eos_token = -1;
};

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

ggml_context* create_context(size_t mem_size) {
    const ggml_init_params params = {
        /* mem_size   */ mem_size,
        /* mem_buffer */ nullptr,
        /* no_alloc   */ false,
    };
    return ggml_init(params);
}

std::string build_prompt(
    nano_llm_infer::Tokenizer* tokenizer,
    const std::string& user_prompt) {
    const std::vector<nano_llm_infer::ChatMessage> messages = {
        {"user", user_prompt},
    };
    return tokenizer->apply_chat_template(messages, true);
}

std::vector<int> encode_prompt(
    nano_llm_infer::Tokenizer* tokenizer,
    const std::string& prompt) {
    std::vector<int> prompt_tokens = tokenizer->encode(prompt);
    prompt_tokens.insert(prompt_tokens.begin(), tokenizer->bos());
    return prompt_tokens;
}

int run_prefill(
    const RuntimeConfig& config,
    const nano_llm_infer::Model& model,
    RuntimeState* state,
    const std::vector<int>& prompt_tokens) {
    ggml_context* ctx = create_context(config.prefill_ctx_bytes);
    state->transformer.set_compute_ctx(ctx);

    ggml_tensor* embeds = embedding_lookup(ctx, model, prompt_tokens);
    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_tensor* logits = state->transformer.forward_logits(embeds, 0, &state->kv);
    ggml_build_forward_expand(graph, logits);
    ggml_graph_compute_with_ctx(ctx, graph, config.compute_threads);

    auto* logit_data = static_cast<float*>(logits->data);
    const int64_t vocab_size = logits->ne[0];
    const int64_t prompt_len = static_cast<int64_t>(prompt_tokens.size());
    const float* last_logits = logit_data + (prompt_len - 1) * vocab_size;
    const int next_token = select_argmax(last_logits, vocab_size);

    ggml_free(ctx);
    state->kv.n_filled = static_cast<int>(prompt_tokens.size());
    return next_token;
}

int run_decode_step(
    const RuntimeConfig& config,
    const nano_llm_infer::Model& model,
    RuntimeState* state,
    int token) {
    ggml_context* ctx = create_context(config.decode_ctx_bytes);
    state->transformer.set_compute_ctx(ctx);

    ggml_tensor* embed = embedding_lookup(ctx, model, {token});
    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_tensor* logits = state->transformer.forward_logits(embed, state->kv.n_filled, &state->kv);
    ggml_build_forward_expand(graph, logits);
    ggml_graph_compute_with_ctx(ctx, graph, config.compute_threads);

    auto* logit_data = static_cast<float*>(logits->data);
    const int64_t vocab_size = logits->ne[0];
    const int next_token = select_argmax(logit_data, vocab_size);

    ggml_free(ctx);
    state->kv.n_filled++;
    return next_token;
}

bool should_stop(const RuntimeConfig& config, const RuntimeState& state, int token) {
    return token == state.eos_token || state.kv.n_filled >= config.max_seq_len;
}

RuntimeState create_runtime_state(
    const nano_llm_infer::Model& model,
    const RuntimeConfig& config,
    int eos_token) {
    return RuntimeState{
        nano_llm_infer::Transformer(model),
        nano_llm_infer::make_kv_cache(model, config.max_seq_len),
        eos_token,
    };
}

}  // namespace

int main(int argc, char** argv) {
    const RuntimeConfig config;
    ensure_valid_omp_threads();

    nano_llm_infer::GGUFLoader loader;
    nano_llm_infer::Model model = loader.load_model(config.model_path);

    std::string user_prompt = "Who are you?";
    if (argc > 1) {
        user_prompt = argv[1];
    }

    nano_llm_infer::Tokenizer tokenizer;
    tokenizer.load_from_gguf(loader);

    const std::string prompt = build_prompt(&tokenizer, user_prompt);
    std::cerr << "Prompt: " << prompt << '\n';

    std::vector<int> prompt_tokens = encode_prompt(&tokenizer, prompt);
    const int n_prompt = static_cast<int>(prompt_tokens.size());
    std::cerr << "Prompt tokens: " << n_prompt << '\n';

    RuntimeState state = create_runtime_state(model, config, tokenizer.eos());
    int next_token = run_prefill(config, model, &state, prompt_tokens);

    std::string response;
    std::cout << "Response: " << std::flush;

    for (int step = 0; step < config.max_new_tokens; ++step) {
        if (should_stop(config, state, next_token)) {
            break;
        }

        std::string decoded = tokenizer.decode({next_token});
        response += decoded;

        if (trim_stop_sequence(&response)) {
            break;
        }

        std::cout << decoded << std::flush;
        next_token = run_decode_step(config, model, &state, next_token);
    }

    std::cout << '\n';
    nano_llm_infer::free_kv_cache(state.kv);
    return 0;
}
