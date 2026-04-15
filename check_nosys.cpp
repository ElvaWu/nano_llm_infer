// Quick test: no system message, print top 10 logits
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include "gguf_loader.h"
#include "tokenizer.h"
#include "transformer.h"
#include "ggml-cpu.h"
int main() {
    nano_llm_infer::GGUFLoader loader;
    nano_llm_infer::Model model = loader.load_model("model_file/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");
    nano_llm_infer::Tokenizer tokenizer;
    tokenizer.load_from_gguf(loader);
    // No system message
    std::vector<nano_llm_infer::ChatMessage> messages = {{"user", "How are you?"}};
    std::string prompt = tokenizer.apply_chat_template(messages, true);
    std::cerr << "Prompt: [" << prompt << "]\n";
    std::vector<int> tokens = tokenizer.encode(prompt);
    tokens.insert(tokens.begin(), tokenizer.bos());
    std::cerr << "Tokens (" << tokens.size() << "): ";
    for (int t : tokens) std::cerr << t << " ";
    std::cerr << "\n";
    nano_llm_infer::Transformer transformer(model);
    ggml_context* ctx = ggml_init({256ull*1024*1024, NULL, false});
    transformer.set_compute_ctx(ctx);
    ggml_tensor* ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t)tokens.size());
    for (int64_t i = 0; i < (int64_t)tokens.size(); i++) ((int32_t*)ids->data)[i] = tokens[i];
    auto emb = ggml_get_rows(ctx, nano_llm_infer::get_tensor_by_name(model, "token_embd.weight"), ids);
    auto logits = transformer.forward_logits(emb, 0);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, logits);
    ggml_graph_compute_with_ctx(ctx, gf, 4);
    float* lg = (float*)logits->data + (logits->ne[1]-1)*logits->ne[0];
    std::vector<int> idx(logits->ne[0]);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin()+10, idx.end(), [&](int a,int b){return lg[a]>lg[b];});
    std::cout << "Top-10:\n";
    for (int k=0;k<10;k++) std::cout<<"  "<<idx[k]<<" '"<<tokenizer.decode({idx[k]})<<"' "<<lg[idx[k]]<<"\n";
    ggml_free(ctx);
}
