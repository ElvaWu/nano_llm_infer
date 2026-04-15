#include <iostream>
#include "gguf_loader.h"

int main() {
    nano_llm_infer::GGUFLoader loader;
    auto model = loader.load_model("model_file/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");
    
    std::cout << "n_heads: " << model.n_heads << "\n";
    std::cout << "n_kv_heads: " << model.n_kv_heads << "\n";
    std::cout << "n_embd: " << model.n_embd << "\n";
    std::cout << "head_dim: " << (model.n_embd / model.n_heads) << "\n";
    std::cout << "rope_freq_base: " << model.rope_freq_base << "\n";
    
    return 0;
}
