#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "ggml-cpu.h"
#include "gguf_loader.h"
#include "tokenizer.h"
#include "transformer.h"

namespace {

constexpr const char* kModelPath = "model_file/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";
constexpr int kComputeThreads = 4;
constexpr int kMaxNewTokens = 32;

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

    ggml_tensor* token_embeddings = nano_llm_infer::get_tensor_by_name(model, "token_embd.weight");
    return ggml_get_rows(compute_ctx, token_embeddings, token_tensor);
}

void ensure_valid_omp_threads() {
    const char* omp_env = std::getenv("OMP_NUM_THREADS");
    if (omp_env == nullptr || std::string(omp_env).empty() || std::atoi(omp_env) < 1) {
        setenv("OMP_NUM_THREADS", "4", 1);
    }
}

}  // namespace

int main(int argc, char** argv) {
    ensure_valid_omp_threads();

    nano_llm_infer::GGUFLoader loader;
    nano_llm_infer::Model model = loader.load_model(kModelPath);

    std::string user_prompt = "Hello, how are you?";
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

    std::vector<int> tokens = tokenizer.encode(prompt);
    tokens.insert(tokens.begin(), tokenizer.bos());

    nano_llm_infer::Transformer transformer(model);
    const ggml_init_params compute_params = {
        /* mem_size   */ 384ull * 1024 * 1024,
        /* mem_buffer */ nullptr,
        /* no_alloc   */ false,
    };

    const int eos = tokenizer.eos();
    const std::vector<std::string> stop_strings = {
        "<|assistant|>",
        "<|user|>",
        "<|system|>",
    };

    std::string response;
    std::cout << "Response: " << std::flush;
    for (int step = 0; step < kMaxNewTokens; ++step) {
        ggml_context* compute_ctx = ggml_init(compute_params);
        transformer.set_compute_ctx(compute_ctx);

        ggml_tensor* embeds = embedding_lookup(compute_ctx, model, tokens);
        ggml_tensor* logits_tensor = transformer.forward_logits(embeds, 0);

        ggml_cgraph* graph = ggml_new_graph(compute_ctx);
        ggml_build_forward_expand(graph, logits_tensor);
        ggml_graph_compute_with_ctx(compute_ctx, graph, kComputeThreads);

        auto* logits = static_cast<float*>(logits_tensor->data);
        const int64_t vocab_size = logits_tensor->ne[0];
        const int64_t seq_len = logits_tensor->ne[1];
        float* last_logits = logits + (seq_len - 1) * vocab_size;

        const int next_token = static_cast<int>(
            std::max_element(last_logits, last_logits + vocab_size) - last_logits);

        ggml_free(compute_ctx);

        if (next_token == eos) {
            break;
        }

        tokens.push_back(next_token);
        response += tokenizer.decode({next_token});

        bool hit_stop = false;
        for (const auto& stop : stop_strings) {
            if (response.size() >= stop.size() &&
                response.compare(response.size() - stop.size(), stop.size(), stop) == 0) {
                response.resize(response.size() - stop.size());
                hit_stop = true;
                break;
            }
        }
        if (hit_stop) {
            break;
        }
    }

    std::cout << response << '\n';
    return 0;
}
