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

    if (pos < 22) {  // Debug first few steps
        fprintf(stderr, "\n=== Attention Debug (pos=%d, seq_len=%ld, is_decode=%d) ===\n",
                pos, seq_len, is_decode);
        fprintf(stderr, "Input x: [%ld, %ld, %ld, %ld]\n", x->ne[0], x->ne[1], x->ne[2], x->ne[3]);
    }

    ggml_tensor* q = ggml_mul_mat(ctx, w->wq, x);
    ggml_tensor* k = ggml_mul_mat(ctx, w->wk, x);
    ggml_tensor* v = ggml_mul_mat(ctx, w->wv, x);

    // Reshape to split heads: [n_embd, seq_len] -> [head_dim, n_heads, seq_len]
    // RoPE expects seq_len in ne2 so it can index positions correctly
    q = ggml_reshape_3d(ctx, q, head_dim, n_heads_, seq_len);
    k = ggml_reshape_3d(ctx, k, head_dim, n_kv_heads_, seq_len);
    v = ggml_reshape_3d(ctx, v, head_dim, n_kv_heads_, seq_len);

    if (pos < 22) {
        fprintf(stderr, "After reshape - Q: [%ld, %ld, %ld], K: [%ld, %ld, %ld], V: [%ld, %ld, %ld]\n",
                q->ne[0], q->ne[1], q->ne[2], k->ne[0], k->ne[1], k->ne[2], v->ne[0], v->ne[1], v->ne[2]);
    }

    // Apply RoPE
    // Position tensor size must match ne2 (seq_len)
    ggml_tensor* pos_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len);
    auto* pos_data = static_cast<int32_t*>(pos_tensor->data);

    // Each token gets its position: [pos, pos+1, ..., pos+seq_len-1]
    for (int64_t i = 0; i < seq_len; ++i) {
        pos_data[i] = pos + static_cast<int>(i);
    }

    q = ggml_rope(ctx, q, pos_tensor, head_dim, GGML_ROPE_TYPE_NEOX);
    k = ggml_rope(ctx, k, pos_tensor, head_dim, GGML_ROPE_TYPE_NEOX);

    if (pos < 22) {
        fprintf(stderr, "After RoPE - Q: [%ld, %ld, %ld], K: [%ld, %ld, %ld]\n",
                q->ne[0], q->ne[1], q->ne[2], k->ne[0], k->ne[1], k->ne[2]);
    }

    // Cache layout: [head_dim, n_kv_heads, max_seq_len]
    // Strides: nb[0]=sizeof(float), nb[1]=head_dim*sizeof(float), nb[2]=head_dim*n_kv_heads*sizeof(float)

    // k, v are now [head_dim, n_kv_heads, seq_len] - already matches cache layout!
    // No permute needed
    ggml_tensor* k_cache = k;
    ggml_tensor* v_cache = v;

    // Write K and V into the cache
    // For cache [head_dim, n_kv_heads, max_seq_len], to write at position pos:
    // - nb[0] = sizeof(float) (stride for head_dim)
    // - nb[1] = head_dim * sizeof(float) (stride for n_kv_heads)
    // - nb[2] = head_dim * n_kv_heads * sizeof(float) (stride for seq position)
    // - offset = pos * head_dim * n_kv_heads * sizeof(float)
    const size_t stride_seq = static_cast<size_t>(head_dim) * static_cast<size_t>(n_kv_heads_) * sizeof(float);

    ggml_tensor* k_slot = ggml_view_3d(ctx, kv->k,
        head_dim, static_cast<int64_t>(n_kv_heads_), seq_len,
        head_dim * sizeof(float),
        head_dim * static_cast<size_t>(n_kv_heads_) * sizeof(float),
        static_cast<size_t>(pos) * stride_seq);
    ggml_tensor* v_slot = ggml_view_3d(ctx, kv->v,
        head_dim, static_cast<int64_t>(n_kv_heads_), seq_len,
        head_dim * sizeof(float),
        head_dim * static_cast<size_t>(n_kv_heads_) * sizeof(float),
        static_cast<size_t>(pos) * stride_seq);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, k_cache, k_slot));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, v_cache, v_slot));

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
        // Decode: read entire cache [head_dim, n_kv_heads, pos+1]
        n_attn = static_cast<int64_t>(pos) + 1;
        k_attn = ggml_view_3d(ctx, kv->k,
            head_dim, static_cast<int64_t>(n_kv_heads_), n_attn,
            head_dim * sizeof(float),
            head_dim * static_cast<size_t>(n_kv_heads_) * sizeof(float),
            0);
        v_attn = ggml_view_3d(ctx, kv->v,
            head_dim, static_cast<int64_t>(n_kv_heads_), n_attn,
            head_dim * sizeof(float),
            head_dim * static_cast<size_t>(n_kv_heads_) * sizeof(float),
            0);

        if (pos < 22) {
            fprintf(stderr, "Cache view strides - K: nb[0]=%zu, nb[1]=%zu, nb[2]=%zu\n",
                    k_attn->nb[0], k_attn->nb[1], k_attn->nb[2]);
            fprintf(stderr, "Cache K tensor: ne[0]=%ld, ne[1]=%ld, ne[2]=%ld\n",
                    kv->k->ne[0], kv->k->ne[1], kv->k->ne[2]);
        }
    }

    // Manual attention with GQA support (following llama.cpp pattern)
    // Current: q=[head_dim, n_heads, seq_len], k_attn=[head_dim, n_kv_heads, n_attn]
    // Permute to: q=[head_dim, seq_len, n_heads], k=[head_dim, n_attn, n_kv_heads]

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Permute: [head_dim, n_heads, seq_len] -> [head_dim, seq_len, n_heads]
    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    // Permute: [head_dim, n_kv_heads, n_attn] -> [head_dim, n_attn, n_kv_heads]
    k_attn = ggml_permute(ctx, k_attn, 0, 2, 1, 3);
    v_attn = ggml_permute(ctx, v_attn, 0, 2, 1, 3);

    if (pos < 22) {
        fprintf(stderr, "After permute - Q: [%ld, %ld, %ld], K: [%ld, %ld, %ld], V: [%ld, %ld, %ld]\n",
                q->ne[0], q->ne[1], q->ne[2], k_attn->ne[0], k_attn->ne[1], k_attn->ne[2],
                v_attn->ne[0], v_attn->ne[1], v_attn->ne[2]);
    }

    // Compute attention scores: K @ Q
    // ggml_mul_mat(K, Q) with K=[head_dim, n_attn, n_kv_heads], Q=[head_dim, seq_len, n_heads]
    // Result: [n_attn, seq_len, n_heads]
    ggml_tensor* kq = ggml_mul_mat(ctx, k_attn, q);
    kq = ggml_scale(ctx, kq, scale);

    if (pos < 22) {
        fprintf(stderr, "After K@Q - kq: [%ld, %ld, %ld], scale=%.6f\n",
                kq->ne[0], kq->ne[1], kq->ne[2], scale);
    }

    // Apply causal mask for prefill
    if (!is_decode) {
        kq = ggml_diag_mask_inf(ctx, kq, 0);
    }

    kq = ggml_soft_max(ctx, kq);

    // Transpose V: [head_dim, n_attn, n_kv_heads] -> [n_attn, head_dim, n_kv_heads]
    v_attn = ggml_cont(ctx, ggml_transpose(ctx, v_attn));

    if (pos < 22) {
        fprintf(stderr, "After V transpose - V: [%ld, %ld, %ld]\n",
                v_attn->ne[0], v_attn->ne[1], v_attn->ne[2]);
    }

    // Apply attention to values: V @ kq
    // ggml_mul_mat(V, kq) with V=[n_attn, head_dim, n_kv_heads], kq=[n_attn, seq_len, n_heads]
    // Result: [head_dim, seq_len, n_heads]
    ggml_tensor* kqv = ggml_mul_mat(ctx, v_attn, kq);

    if (pos < 22) {
        fprintf(stderr, "After V@kq - kqv: [%ld, %ld, %ld]\n",
                kqv->ne[0], kqv->ne[1], kqv->ne[2]);
    }

    // Permute back: [head_dim, seq_len, n_heads] -> [head_dim, n_heads, seq_len]
    kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));

    if (pos < 22) {
        fprintf(stderr, "After permute back - kqv: [%ld, %ld, %ld]\n",
                kqv->ne[0], kqv->ne[1], kqv->ne[2]);
    }

    // Merge heads: [head_dim, n_heads, seq_len] -> [n_embd, seq_len]
    kqv = ggml_reshape_2d(ctx, kqv, head_dim * n_heads_, seq_len);

    if (pos < 22) {
        fprintf(stderr, "After reshape - kqv: [%ld, %ld]\n", kqv->ne[0], kqv->ne[1]);
    }

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
