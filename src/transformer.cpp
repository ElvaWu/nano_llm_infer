#include "transformer.h"
#include <cmath>

namespace nano_llm_infer{

ggml_tensor* rms_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight, float eps) {
    auto norm = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, norm, weight);
}

OutputHead::OutputHead(const Model& model):context_(model.ctx), eps_(model.rms_norm_eps) {
    norm_weight_ = get_tensor_by_name(model, "output_norm.weight");
    lm_head_weight_ = get_tensor_by_name(model, "output.weight");
}

ggml_tensor* OutputHead::forward(ggml_tensor* hidden) {
    auto normed = rms_norm(context_, hidden, norm_weight_, eps_);
    auto logits = ggml_mul_mat(context_, lm_head_weight_, normed);
    return ggml_argmax(context_, logits);
}

// get attention weights for specific layer
AttentionWeights* get_attention_weights(const Model& model, uint32_t layer_idx) {
    std::string prefix = "blk." + std::to_string(layer_idx) + ".attn_";
    AttentionWeights* w = new AttentionWeights;

    w->wq = get_tensor_by_name(model, prefix + "q.weight");
    w->wk = get_tensor_by_name(model, prefix + "k.weight");
    w->wv = get_tensor_by_name(model, prefix + "v.weight");
    w->wo = get_tensor_by_name(model, prefix + "output.weight");

    return w;
}


TransformerBlock::TransformerBlock(const Model& model): context_(model.ctx), eps_(model.rms_norm_eps), n_heads_(model.n_heads), n_kv_heads_(model.n_kv_heads) {
    std::string prefix = "blk.0.";
    attn_norm_weight_ = get_tensor_by_name(model, prefix + "attn_norm.weight");
    ffn_norm_weight_  = get_tensor_by_name(model, prefix + "ffn_norm.weight");
    attn_weights_     = get_attention_weights(model, 0);

    mlp_weights_ = new MLPWeights;
    mlp_weights_->gate_proj = get_tensor_by_name(model, prefix + "ffn_gate.weight");
    mlp_weights_->up_proj   = get_tensor_by_name(model, prefix + "ffn_up.weight");
    mlp_weights_->down_proj = get_tensor_by_name(model, prefix + "ffn_down.weight");
}

TransformerBlock::~TransformerBlock() {
    delete attn_weights_;
    delete mlp_weights_;
}

ggml_tensor* TransformerBlock::self_attention(ggml_tensor* x, AttentionWeights* w) {
    // no kv kache temp
    auto q = ggml_mul_mat(context_, w->wq, x);
    auto k = ggml_mul_mat(context_, w->wk, x);
    auto v = ggml_mul_mat(context_, w->wv, x);

    // split into heads: q -> [head_dim, seq_len, n_heads], k/v -> [head_dim, seq_len, n_kv_heads]
    const int64_t seq_len  = q->ne[1];
    const int64_t head_dim = q->ne[0] / n_heads_;
    q = ggml_reshape_3d(context_, q, head_dim, seq_len, n_heads_);
    k = ggml_reshape_3d(context_, k, head_dim, seq_len, n_kv_heads_);
    v = ggml_reshape_3d(context_, v, head_dim, seq_len, n_kv_heads_);

    auto attn = ggml_mul_mat(context_, k, q);
    attn = ggml_scale(context_, attn, 1.0f / sqrtf((float)head_dim)); // attn / sqrt(head_dim)
    attn = ggml_soft_max(context_, attn);

    auto v_t = ggml_cont(context_, ggml_transpose(context_, v));
    auto out = ggml_mul_mat(context_, v_t, attn);
    // merge heads: [head_dim, seq_len, n_heads] -> [n_embd, seq_len]
    out = ggml_reshape_2d(context_, out, head_dim * n_heads_, seq_len);
    out = ggml_mul_mat(context_, w->wo, out);

    return out;
}

ggml_tensor* TransformerBlock::mlp(ggml_tensor* x) {
    // SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))
    auto gate = ggml_mul_mat(context_, mlp_weights_->gate_proj, x);
    auto up   = ggml_mul_mat(context_, mlp_weights_->up_proj,   x);
    gate = ggml_silu(context_, gate);
    auto hidden = ggml_mul(context_, gate, up);
    return ggml_mul_mat(context_, mlp_weights_->down_proj, hidden);
}

ggml_tensor* TransformerBlock::forward(ggml_tensor* input) {
    // 1. attention pre-norm + self-attention + residual
    auto h    = rms_norm(context_, input, attn_norm_weight_, eps_);
    auto attn = self_attention(h, attn_weights_);
    auto x    = ggml_add(context_, input, attn);

    // 2. FFN pre-norm + MLP + residual  
    auto h2  = rms_norm(context_, x, ffn_norm_weight_, eps_);
    auto ff  = mlp(h2);
    x        = ggml_add(context_, x, ff);
    return x;
}



} // namespace nano_llm_infer