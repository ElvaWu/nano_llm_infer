#include "transformer.h"

#include <algorithm>
#include <cmath>

namespace nano_llm_infer {

namespace {

AttentionWeights* get_attention_weights(const Model& model, uint32_t layer_idx) {
    const std::string prefix = "blk." + std::to_string(layer_idx) + ".attn_";
    auto* weights = new AttentionWeights;
    weights->wq = get_tensor_by_name(model, prefix + "q.weight");
    weights->wk = get_tensor_by_name(model, prefix + "k.weight");
    weights->wv = get_tensor_by_name(model, prefix + "v.weight");
    weights->wo = get_tensor_by_name(model, prefix + "output.weight");
    return weights;
}

}  // namespace

ggml_tensor* rms_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight, float eps) {
    ggml_tensor* norm = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, norm, weight);
}

OutputHead::OutputHead(const Model& model)
    : eps_(model.rms_norm_eps), context_(model.ctx) {
    norm_weight_ = get_tensor_by_name(model, "output_norm.weight");
    lm_head_weight_ = get_tensor_by_name(model, "output.weight");
}

ggml_tensor* OutputHead::forward(ggml_tensor* hidden) {
    ggml_tensor* normed = rms_norm(context_, hidden, norm_weight_, eps_);
    ggml_tensor* logits = ggml_mul_mat(context_, lm_head_weight_, normed);
    return ggml_argmax(context_, logits);
}

ggml_tensor* OutputHead::forward_logits(ggml_tensor* hidden) {
    ggml_tensor* normed = rms_norm(context_, hidden, norm_weight_, eps_);
    return ggml_mul_mat(context_, lm_head_weight_, normed);
}

TransformerBlock::TransformerBlock(const Model& model, uint32_t layer_idx)
    : eps_(model.rms_norm_eps),
      n_heads_(model.n_heads),
      n_kv_heads_(model.n_kv_heads),
      context_(model.ctx) {
    const std::string prefix = "blk." + std::to_string(layer_idx) + ".";
    attn_norm_weight_ = get_tensor_by_name(model, prefix + "attn_norm.weight");
    ffn_norm_weight_ = get_tensor_by_name(model, prefix + "ffn_norm.weight");
    attn_weights_ = get_attention_weights(model, layer_idx);

    mlp_weights_ = new MLPWeights;
    mlp_weights_->gate_proj = get_tensor_by_name(model, prefix + "ffn_gate.weight");
    mlp_weights_->up_proj = get_tensor_by_name(model, prefix + "ffn_up.weight");
    mlp_weights_->down_proj = get_tensor_by_name(model, prefix + "ffn_down.weight");
}

TransformerBlock::~TransformerBlock() {
    delete attn_weights_;
    delete mlp_weights_;
}

ggml_tensor* TransformerBlock::self_attention(ggml_tensor* x, AttentionWeights* w, int pos) {
    ggml_tensor* q = ggml_mul_mat(context_, w->wq, x);
    ggml_tensor* k = ggml_mul_mat(context_, w->wk, x);
    ggml_tensor* v = ggml_mul_mat(context_, w->wv, x);

    const int64_t seq_len = q->ne[1];
    const int64_t head_dim = q->ne[0] / n_heads_;
    q = ggml_reshape_3d(context_, q, head_dim, n_heads_, seq_len);
    k = ggml_reshape_3d(context_, k, head_dim, n_kv_heads_, seq_len);
    v = ggml_reshape_3d(context_, v, head_dim, n_kv_heads_, seq_len);

    ggml_tensor* pos_tensor = ggml_new_tensor_1d(context_, GGML_TYPE_I32, seq_len);
    auto* pos_data = static_cast<int32_t*>(pos_tensor->data);
    for (int64_t i = 0; i < seq_len; ++i) {
        pos_data[i] = pos + static_cast<int>(i);
    }
    k = ggml_rope(context_, k, pos_tensor, head_dim, GGML_ROPE_TYPE_NEOX);
    q = ggml_rope(context_, q, pos_tensor, head_dim, GGML_ROPE_TYPE_NEOX);

    q = ggml_cont(context_, ggml_permute(context_, q, 0, 2, 1, 3));
    k = ggml_cont(context_, ggml_permute(context_, k, 0, 2, 1, 3));
    v = ggml_cont(context_, ggml_permute(context_, v, 0, 2, 1, 3));

    ggml_tensor* attn = ggml_mul_mat(context_, k, q);
    attn = ggml_scale(context_, attn, 1.0f / std::sqrt(static_cast<float>(head_dim)));
    attn = ggml_diag_mask_inf(context_, attn, 0);
    attn = ggml_soft_max(context_, attn);

    ggml_tensor* v_t = ggml_cont(context_, ggml_transpose(context_, v));
    ggml_tensor* out = ggml_mul_mat(context_, v_t, attn);
    out = ggml_cont(context_, ggml_permute(context_, out, 0, 2, 1, 3));
    out = ggml_reshape_2d(context_, out, head_dim * n_heads_, seq_len);
    return ggml_mul_mat(context_, w->wo, out);
}

ggml_tensor* TransformerBlock::mlp(ggml_tensor* x) {
    ggml_tensor* gate = ggml_mul_mat(context_, mlp_weights_->gate_proj, x);
    ggml_tensor* up = ggml_mul_mat(context_, mlp_weights_->up_proj, x);
    gate = ggml_silu(context_, gate);
    ggml_tensor* hidden = ggml_mul(context_, gate, up);
    return ggml_mul_mat(context_, mlp_weights_->down_proj, hidden);
}

ggml_tensor* TransformerBlock::forward(ggml_tensor* input, int pos) {
    ggml_tensor* h = rms_norm(context_, input, attn_norm_weight_, eps_);
    ggml_tensor* attn = self_attention(h, attn_weights_, pos);
    ggml_tensor* x = ggml_add(context_, input, attn);

    ggml_tensor* h2 = rms_norm(context_, x, ffn_norm_weight_, eps_);
    ggml_tensor* ff = mlp(h2);
    return ggml_add(context_, x, ff);
}

Transformer::Transformer(const Model& model)
    : output_head_(model) {
    blocks_.reserve(model.n_layers);
    for (uint32_t i = 0; i < model.n_layers; ++i) {
        blocks_.emplace_back(model, i);
    }
}

void Transformer::set_compute_ctx(ggml_context* ctx) {
    for (auto& block : blocks_) {
        block.set_compute_ctx(ctx);
    }
    output_head_.set_compute_ctx(ctx);
}

ggml_tensor* Transformer::forward(ggml_tensor* input, int pos) {
    ggml_tensor* x = input;
    for (auto& block : blocks_) {
        x = block.forward(x, pos);
    }
    return output_head_.forward(x);
}

ggml_tensor* Transformer::forward_logits(ggml_tensor* input, int pos) {
    ggml_tensor* x = input;
    for (auto& block : blocks_) {
        x = block.forward(x, pos);
    }
    return output_head_.forward_logits(x);
}

ggml_tensor* Transformer::forward_logits_skip_blocks(ggml_tensor* input) {
    return output_head_.forward_logits(input);
}

ggml_tensor* Transformer::forward_n_blocks(ggml_tensor* input, int pos, int n) {
    ggml_tensor* x = input;
    const int limit = std::min(n, static_cast<int>(blocks_.size()));
    for (int i = 0; i < limit; ++i) {
        x = blocks_[static_cast<size_t>(i)].forward(x, pos);
    }
    return x;
}

}  // namespace nano_llm_infer
