#include <iostream>
#include "gguf_loader.h"

int main() {
    nano_llm_infer::GGUFLoader loader;
    nano_llm_infer::Model model = loader.load_model("model_file/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");

    std::cout << "Model hyperparameters:\n";
    std::cout << "  n_layers: " << model.n_layers << "\n";
    std::cout << "  n_heads: " << model.n_heads << "\n";
    std::cout << "  n_kv_heads: " << model.n_kv_heads << "\n";
    std::cout << "  rms_norm_eps: " << model.rms_norm_eps << "\n";

    // Check for RoPE parameters
    std::cout << "\nRoPE parameters:\n";
    try {
        float rope_freq_base = loader.get_metadata<float>("llama.rope.freq_base");
        std::cout << "  rope_freq_base: " << rope_freq_base << "\n";
    } catch (...) {
        std::cout << "  rope_freq_base: not found\n";
    }

    try {
        uint32_t rope_dim = loader.get_metadata<uint32_t>("llama.rope.dimension_count");
        std::cout << "  rope_dimension_count: " << rope_dim << "\n";
    } catch (...) {
        std::cout << "  rope_dimension_count: not found\n";
    }

    try {
        uint8_t rope_scaling_type = loader.get_metadata<uint8_t>("llama.rope.scaling.type");
        std::cout << "  rope_scaling_type: " << (int)rope_scaling_type << "\n";
    } catch (...) {
        std::cout << "  rope_scaling_type: not found\n";
    }

    try {
        std::string arch = loader.get_metadata<std::string>("general.architecture");
        std::cout << "  architecture: " << arch << "\n";
    } catch (...) {
        std::cout << "  architecture: not found\n";
    }

    // Check embedding dimension
    auto token_embd = nano_llm_infer::get_tensor_by_name(model, "token_embd.weight");
    std::cout << "\nEmbedding dimensions:\n";
    std::cout << "  vocab_size: " << token_embd->ne[1] << "\n";
    std::cout << "  n_embd: " << token_embd->ne[0] << "\n";

    // Check attention weight dimensions
    auto wq = nano_llm_infer::get_tensor_by_name(model, "blk.0.attn_q.weight");
    std::cout << "\nAttention weight dimensions (layer 0):\n";
    std::cout << "  wq: [" << wq->ne[0] << ", " << wq->ne[1] << "]\n";

    auto wk = nano_llm_infer::get_tensor_by_name(model, "blk.0.attn_k.weight");
    std::cout << "  wk: [" << wk->ne[0] << ", " << wk->ne[1] << "]\n";

    auto wv = nano_llm_infer::get_tensor_by_name(model, "blk.0.attn_v.weight");
    std::cout << "  wv: [" << wv->ne[0] << ", " << wv->ne[1] << "]\n";

    auto wo = nano_llm_infer::get_tensor_by_name(model, "blk.0.attn_output.weight");
    std::cout << "  wo: [" << wo->ne[0] << ", " << wo->ne[1] << "]\n";

    // Calculate expected dimensions
    int64_t n_embd = token_embd->ne[0];
    int64_t head_dim = n_embd / model.n_heads;
    int64_t kv_dim = head_dim * model.n_kv_heads;

    std::cout << "\nExpected dimensions:\n";
    std::cout << "  head_dim: " << head_dim << "\n";
    std::cout << "  kv_dim: " << kv_dim << "\n";
    std::cout << "  Expected wq: [" << n_embd << ", " << n_embd << "]\n";
    std::cout << "  Expected wk/wv: [" << kv_dim << ", " << n_embd << "]\n";
    std::cout << "  Expected wo: [" << n_embd << ", " << n_embd << "]\n";

    return 0;
}
