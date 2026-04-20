#include "internal.h"

#include <string>

namespace nano_llm_infer {

AttentionWeights load_attention_weights(const Model& model, uint32_t layer_idx) {
    const std::string prefix = "blk." + std::to_string(layer_idx) + ".attn_";
    AttentionWeights weights;
    weights.wq = get_tensor_by_name(model, prefix + "q.weight");
    weights.wk = get_tensor_by_name(model, prefix + "k.weight");
    weights.wv = get_tensor_by_name(model, prefix + "v.weight");
    weights.wo = get_tensor_by_name(model, prefix + "output.weight");
    return weights;
}

TransformerBlock::TransformerBlock(const Model& model, uint32_t layer_idx)
    : eps_(model.rms_norm_eps),
      n_heads_(model.n_heads),
      n_kv_heads_(model.n_kv_heads),
      rope_n_ctx_orig_(model.n_ctx_train),
      rope_dimensions_(model.rope_dimension_count),
      rope_mode_(model.rope_mode),
      rope_freq_base_(model.rope_freq_base),
      rope_freq_scale_(model.rope_freq_scale),
      context_(model.ctx),
      attn_weights_(load_attention_weights(model, layer_idx)) {
    const std::string prefix = "blk." + std::to_string(layer_idx) + ".";
    attn_norm_weight_ = get_tensor_by_name(model, prefix + "attn_norm.weight");
    ffn_norm_weight_ = get_tensor_by_name(model, prefix + "ffn_norm.weight");
    mlp_weights_.gate_proj = get_tensor_by_name(model, prefix + "ffn_gate.weight");
    mlp_weights_.up_proj = get_tensor_by_name(model, prefix + "ffn_up.weight");
    mlp_weights_.down_proj = get_tensor_by_name(model, prefix + "ffn_down.weight");
}

ggml_tensor* TransformerBlock::mlp(ggml_tensor* input) {
    ggml_tensor* gate = ggml_mul_mat(context_, mlp_weights_.gate_proj, input);
    ggml_tensor* up = ggml_mul_mat(context_, mlp_weights_.up_proj, input);
    ggml_tensor* hidden = ggml_mul(context_, ggml_silu(context_, gate), up);
    return ggml_mul_mat(context_, mlp_weights_.down_proj, hidden);
}

ggml_tensor* TransformerBlock::forward(ggml_tensor* input, int pos, ggml_cgraph* gf, KVCacheLayer* kv) {
    ggml_tensor* attn_input = rms_norm(context_, input, attn_norm_weight_, eps_);
    ggml_tensor* attn_output = self_attention(attn_input, pos, gf, kv);
    ggml_tensor* residual = ggml_add(context_, input, attn_output);

    ggml_tensor* mlp_input = rms_norm(context_, residual, ffn_norm_weight_, eps_);
    ggml_tensor* mlp_output = mlp(mlp_input);
    return ggml_add(context_, residual, mlp_output);
}

}  // namespace nano_llm_infer
