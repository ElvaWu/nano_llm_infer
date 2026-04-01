#include <iostream>
#include <vector>
#include "gguf_loader.h"
#include "tokenizer.h"
#include "transformer.h"
#include "ggml-cpu.h"

// 定义EOS（假设为0，实际值请根据模型设置）
// const int EOS = 0;

ggml_tensor* embedding_lookup(nano_llm_infer::Model* model, const std::vector<int>& tokens){
    const int64_t n_tokens = static_cast<int64_t>(tokens.size());
    ggml_tensor * token_tensor = ggml_new_tensor_1d(model->ctx, GGML_TYPE_I32, n_tokens);
    // write actual token ids so the embedding lookup reads the right rows
    int32_t* data = static_cast<int32_t*>(token_tensor->data);
    for (int64_t i = 0; i < n_tokens; ++i) {
        data[i] = tokens[static_cast<size_t>(i)];
    }
    auto token_embeddings = nano_llm_infer::get_tensor_by_name(*model, "token_embd.weight");
    return ggml_get_rows(model->ctx, token_embeddings, token_tensor);
} 


int main() {
    // load model
    nano_llm_infer::GGUFLoader loader;
    nano_llm_infer::Model model = loader.load_model("model_file/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");
    
    std::string prompt = "How are you"; // prompt

    // 1. tokenize
    nano_llm_infer::Tokenizer tokenizer;
    tokenizer.load_from_gguf(loader);
    std::vector<int> tokens = tokenizer.encode(prompt); 
    // std::string text = tokenizer.decode(tokens); // decode回去看看
    // std::cout << "Decoded text: " << text << std::endl;

    // for(auto t : tokens) {
    //     std::cout << t << " ";
    // }
    // std::cout << tokens.size() << std::endl;
    auto token_embeddings = embedding_lookup(&model, tokens);
    // std::cout << "Token embeddings shape: [" << token_embeddings->ne[0] << ", " << token_embeddings->ne[1] << "]" << std::endl;

    // while (true) {
        // 1. forward  just 1 layer
        nano_llm_infer::TransformerBlock* transformer_block = new nano_llm_infer::TransformerBlock(model);
        auto tb_output = transformer_block->forward(token_embeddings);

        nano_llm_infer::OutputHead* output_head = new nano_llm_infer::OutputHead(model);
        auto next_token_tensor = output_head->forward(tb_output);

        // Build and execute the compute graph.
        // ggml ops (ggml_mul_mat, ggml_add, etc.) only record nodes in a DAG —
        // no arithmetic runs until ggml_graph_compute is called.
        struct ggml_cgraph* gf = ggml_new_graph(model.ctx);
        ggml_build_forward_expand(gf, next_token_tensor);
        ggml_graph_compute_with_ctx(model.ctx, gf, /*n_threads=*/1);

        // next_token_tensor->data now holds the computed argmax result
        int32_t next_token = *static_cast<int32_t*>(next_token_tensor->data);
        std::cout << "next token id: " << next_token << std::endl;
        auto next_token_text = tokenizer.decode(next_token);
        std::cout << "next token text: " << next_token_text << std::endl;
        
        //  std::vector<float> logits = 

        // // 2. sample
        // int next_token = sample(logits);

        // // 3. append
        // tokens.push_back(next_token);

        // std::cout << next_token;

        // if (next_token == EOS) break;
    // }
    return 0;
}