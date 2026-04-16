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
    const bool is_decode = (seq_len == 1 && pos > 0);

    ggml_tensor* q = ggml_mul_mat(ctx, w->wq, x);
    ggml_tensor* k = ggml_mul_mat(ctx, w->wk, x);
    ggml_tensor* v = ggml_mul_mat(ctx, w->wv, x);

    // Reshape to split heads: [n_embd, seq_len] -> [head_dim, n_heads, seq_len]
    q = ggml_reshape_3d(ctx, q, head_dim, n_heads_, seq_len);
    k = ggml_reshape_3d(ctx, k, head_dim, n_kv_heads_, seq_len);
    v = ggml_reshape_3d(ctx, v, head_dim, n_kv_heads_, seq_len);

    // Apply RoPE
    ggml_tensor* pos_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len);
    auto* pos_data = static_cast<int32_t*>(pos_tensor->data);
    for (int64_t i = 0; i < seq_len; ++i) {
        pos_data[i] = pos + static_cast<int>(i);
    }

    q = ggml_rope(ctx, q, pos_tensor, head_dim, GGML_ROPE_TYPE_NEOX);
    k = ggml_rope(ctx, k, pos_tensor, head_dim, GGML_ROPE_TYPE_NEOX);

    // Cache layout: [head_dim, n_kv_heads, max_seq_len]
    // Write K and V into cache using ggml_cpy + ggml_view_3d

    // Create views into the cache at the correct position
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

    // Copy current k/v into the cache views
    ggml_tensor* k_cpy = ggml_cpy(ctx, k, k_cache_view);
    ggml_tensor* v_cpy = ggml_cpy(ctx, v, v_cache_view);

    // CRITICAL FIX: Execute the cache write immediately in a separate graph
    // This ensures the cache is updated before we try to read from it
    {
        ggml_cgraph* write_graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(write_graph, k_cpy);
        ggml_build_forward_expand(write_graph, v_cpy);
        ggml_graph_compute_with_ctx(ctx, write_graph, 4);
    }

    // Select K and V tensors for attention
    ggml_tensor* k_attn;
    ggml_tensor* v_attn;
    int64_t n_attn;

    if (!is_decode) {
        // Prefill: use current k/v directly (not from cache)
        n_attn = seq_len;
        k_attn = k;
        v_attn = v;
    } else {
        // Decode: concatenate cached K/V with current k/v
        // Read previous cache [0..pos-1]
        if (pos > 0) {
            n_attn = static_cast<int64_t>(pos) + 1;

            ggml_tensor* k_prev = ggml_view_3d(ctx, kv->k,
                head_dim, static_cast<int64_t>(n_kv_heads_), pos,
                head_dim * sizeof(float),
                head_dim * static_cast<size_t>(n_kv_heads_) * sizeof(float),
                0);
            ggml_tensor* v_prev = ggml_view_3d(ctx, kv->v,
                head_dim, static_cast<int64_t>(n_kv_heads_), pos,
                head_dim * sizeof(float),
                head_dim * static_cast<size_t>(n_kv_heads_) * sizeof(float),
                0);

            // Concatenate: [head_dim, n_kv_heads, pos] + [head_dim, n_kv_heads, 1]
            k_attn = ggml_concat(ctx, k_prev, k, 2);
            v_attn = ggml_concat(ctx, v_prev, v, 2);
        } else {
            // First decode step, no previous cache
            n_attn = 1;
            k_attn = k;
            v_attn = v;
        }
    }

    // Manual attention with GQA support
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Permute: [head_dim, n_heads, seq_len] -> [head_dim, seq_len, n_heads]
    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    // Permute: [head_dim, n_kv_heads, n_attn] -> [head_dim, n_attn, n_kv_heads]
    k_attn = ggml_permute(ctx, k_attn, 0, 2, 1, 3);
    v_attn = ggml_permute(ctx, v_attn, 0, 2, 1, 3);

    // Compute attention scores: K @ Q
    ggml_tensor* kq = ggml_mul_mat(ctx, k_attn, q);
    kq = ggml_scale(ctx, kq, scale);

    // Apply causal mask for prefill
    if (!is_decode) {
        kq = ggml_diag_mask_inf(ctx, kq, 0);
    }

    kq = ggml_soft_max(ctx, kq);

    // Transpose V: [head_dim, n_attn, n_kv_heads] -> [n_attn, head_dim, n_kv_heads]
    v_attn = ggml_cont(ctx, ggml_transpose(ctx, v_attn));

    // Apply attention to values: V @ kq
    ggml_tensor* kqv = ggml_mul_mat(ctx, v_attn, kq);

    // Permute back: [head_dim, seq_len, n_heads] -> [head_dim, n_heads, seq_len]
    kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));

    // Merge heads: [head_dim, n_heads, seq_len] -> [n_embd, seq_len]
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
