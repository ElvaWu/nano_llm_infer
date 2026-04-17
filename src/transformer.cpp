#include "transformer.h"

#include <algorithm>
#include <cmath>
#include "ggml-cpu.h"

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

ggml_tensor* OutputHead::forward_logits(ggml_tensor* hidden) {
    ggml_tensor* normed = rms_norm(context_, hidden, norm_weight_, eps_);
    return ggml_mul_mat(context_, lm_head_weight_, normed);
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

ggml_tensor* TransformerBlock::self_attention(
    ggml_tensor* x, AttentionWeights* w, int pos, ggml_cgraph* gf, KVCacheLayer* kv) {

    ggml_context* ctx = context_;
    const int64_t seq_len = x->ne[1];
    const int64_t head_dim = static_cast<int64_t>(kv->k->ne[0]);
    const int rope_dims = rope_dimensions_ == 0
        ? static_cast<int>(head_dim)
        : static_cast<int>(std::min<int64_t>(rope_dimensions_, head_dim));

    ggml_tensor* q = ggml_mul_mat(ctx, w->wq, x);
    ggml_tensor* k = ggml_mul_mat(ctx, w->wk, x);
    ggml_tensor* v = ggml_mul_mat(ctx, w->wv, x);

    // Reshape to split heads: [n_embd, seq_len] -> [head_dim, n_heads, seq_len]
    q = ggml_reshape_3d(ctx, q, head_dim, n_heads_, seq_len);
    k = ggml_reshape_3d(ctx, k, head_dim, n_kv_heads_, seq_len);
    v = ggml_reshape_3d(ctx, v, head_dim, n_kv_heads_, seq_len);

    // Apply RoPE using model metadata instead of legacy hard-coded defaults.
    ggml_tensor* pos_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len);
    auto* pos_data = static_cast<int32_t*>(pos_tensor->data);
    for (int64_t i = 0; i < seq_len; ++i) {
        pos_data[i] = pos + static_cast<int>(i);
    }

    q = ggml_rope_ext(
        ctx,
        q,
        pos_tensor,
        nullptr,
        rope_dims,
        rope_mode_,
        static_cast<int>(rope_n_ctx_orig_),
        rope_freq_base_,
        rope_freq_scale_,
        0.0f,
        1.0f,
        0.0f,
        0.0f);
    k = ggml_rope_ext(
        ctx,
        k,
        pos_tensor,
        nullptr,
        rope_dims,
        rope_mode_,
        static_cast<int>(rope_n_ctx_orig_),
        rope_freq_base_,
        rope_freq_scale_,
        0.0f,
        1.0f,
        0.0f,
        0.0f);

    // Create views into cache for writing current K/V
    ggml_tensor* k_cache_view = ggml_view_3d(ctx, kv->k,
        head_dim, static_cast<int64_t>(n_kv_heads_), seq_len,
        head_dim * sizeof(float),
        head_dim * static_cast<size_t>(n_kv_heads_) * sizeof(float),
        pos * head_dim * static_cast<size_t>(n_kv_heads_) * sizeof(float));

    ggml_tensor* v_cache_view = ggml_view_3d(ctx, kv->v,
        head_dim, static_cast<int64_t>(n_kv_heads_), seq_len,
        head_dim * sizeof(float),
        head_dim * static_cast<size_t>(n_kv_heads_) * sizeof(float),
        pos * head_dim * static_cast<size_t>(n_kv_heads_) * sizeof(float));

    ggml_tensor* k_cpy = ggml_cpy(ctx, k, k_cache_view);
    ggml_tensor* v_cpy = ggml_cpy(ctx, v, v_cache_view);

    // Match llama.cpp ordering: expand q/v/k then cache writes in the same graph
    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, v);
    ggml_build_forward_expand(gf, k);
    ggml_build_forward_expand(gf, k_cpy);
    ggml_build_forward_expand(gf, v_cpy);

    // Always read attention K/V from cache after writes
    const int64_t n_attn = static_cast<int64_t>(pos) + seq_len;

    ggml_tensor* k_attn = ggml_view_3d(ctx, kv->k,
        head_dim, static_cast<int64_t>(n_kv_heads_), n_attn,
        head_dim * sizeof(float),
        head_dim * static_cast<size_t>(n_kv_heads_) * sizeof(float),
        0);

    ggml_tensor* v_attn = ggml_view_3d(ctx, kv->v,
        head_dim, static_cast<int64_t>(n_kv_heads_), n_attn,
        head_dim * sizeof(float),
        head_dim * static_cast<size_t>(n_kv_heads_) * sizeof(float),
        0);

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    k_attn = ggml_cont(ctx, ggml_permute(ctx, k_attn, 0, 2, 1, 3));
    v_attn = ggml_cont(ctx, ggml_permute(ctx, v_attn, 0, 2, 1, 3));

    ggml_tensor* kq = ggml_mul_mat(ctx, k_attn, q);
    ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
    kq = ggml_scale(ctx, kq, scale);
    kq = ggml_diag_mask_inf(ctx, kq, n_attn - seq_len);
    kq = ggml_soft_max(ctx, kq);

    ggml_tensor* v_t = ggml_cont(ctx, ggml_transpose(ctx, v_attn));
    ggml_tensor* kqv = ggml_mul_mat(ctx, v_t, kq);

    kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));
    kqv = ggml_reshape_2d(ctx, kqv, head_dim * n_heads_, seq_len);

    return ggml_mul_mat(ctx, w->wo, kqv);
}

ggml_tensor* TransformerBlock::mlp(ggml_tensor* x) {
    ggml_tensor* gate = ggml_mul_mat(context_, mlp_weights_->gate_proj, x);
    ggml_tensor* up = ggml_mul_mat(context_, mlp_weights_->up_proj, x);
    gate = ggml_silu(context_, gate);
    ggml_tensor* hidden = ggml_mul(context_, gate, up);
    return ggml_mul_mat(context_, mlp_weights_->down_proj, hidden);
}

ggml_tensor* TransformerBlock::forward(ggml_tensor* input, int pos, ggml_cgraph* gf, KVCacheLayer* kv) {
    ggml_tensor* h = rms_norm(context_, input, attn_norm_weight_, eps_);
    ggml_tensor* attn = self_attention(h, attn_weights_, pos, gf, kv);
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

ggml_tensor* Transformer::forward_logits(ggml_tensor* input, int pos, ggml_cgraph* gf, KVCache* kv) {
    ggml_tensor* x = input;
    for (size_t i = 0; i < blocks_.size(); ++i) {
        x = blocks_[i].forward(x, pos, gf, &kv->layers[i]);
    }
    return output_head_.forward_logits(x);
}

}  // namespace nano_llm_infer
