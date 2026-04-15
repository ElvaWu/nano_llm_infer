#include <iostream>
#include <vector>
#include "gguf_loader.h"
#include "tokenizer.h"
#include "ggml-cpu.h"

void print_shape(const char* name, ggml_tensor* t) {
    std::cerr << name << " shape: [";
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (i > 0) std::cerr << ", ";
        std::cerr << t->ne[i];
    }
    std::cerr << "]\n";
}

int main() {
    nano_llm_infer::GGUFLoader loader;
    nano_llm_infer::Model model = loader.load_model("model_file/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");

    std::cerr << "Model config:\n";
    std::cerr << "  n_layers=" << model.n_layers << "\n";
    std::cerr << "  n_heads=" << model.n_heads << "\n";
    std::cerr << "  n_kv_heads=" << model.n_kv_heads << "\n";
    std::cerr << "  rms_norm_eps=" << model.rms_norm_eps << "\n";

    // Check weight shapes
    auto token_embd = nano_llm_infer::get_tensor_by_name(model, "token_embd.weight");
    print_shape("token_embd.weight", token_embd);

    auto wq = nano_llm_infer::get_tensor_by_name(model, "blk.0.attn_q.weight");
    auto wk = nano_llm_infer::get_tensor_by_name(model, "blk.0.attn_k.weight");
    auto wv = nano_llm_infer::get_tensor_by_name(model, "blk.0.attn_v.weight");
    auto wo = nano_llm_infer::get_tensor_by_name(model, "blk.0.attn_output.weight");

    print_shape("blk.0.attn_q.weight", wq);
    print_shape("blk.0.attn_k.weight", wk);
    print_shape("blk.0.attn_v.weight", wv);
    print_shape("blk.0.attn_output.weight", wo);

    auto output_weight = nano_llm_infer::get_tensor_by_name(model, "output.weight");
    print_shape("output.weight", output_weight);

    return 0;
}
