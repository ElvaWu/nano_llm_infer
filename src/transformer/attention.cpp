#include "internal.h"

#include <algorithm>
#include <cmath>

namespace nano_llm_infer {

namespace {

ggml_tensor* make_position_tensor(ggml_context* ctx, int pos, int64_t seq_len) {
    ggml_tensor* positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len);
    auto* pos_data = static_cast<int32_t*>(positions->data);
    for (int64_t i = 0; i < seq_len; ++i) {
        pos_data[i] = pos + static_cast<int>(i);
    }
    return positions;
}

ggml_tensor* apply_rope(
    ggml_context* ctx,
    ggml_tensor* tensor,
    ggml_tensor* positions,
    int rope_dims,
    int rope_mode,
    uint32_t rope_n_ctx_orig,
    float rope_freq_base,
    float rope_freq_scale) {
    return ggml_rope_ext(
        ctx,
        tensor,
        positions,
        nullptr,
        rope_dims,
        rope_mode,
        static_cast<int>(rope_n_ctx_orig),
        rope_freq_base,
        rope_freq_scale,
        0.0f,
        1.0f,
        0.0f,
        0.0f);
}

ggml_tensor* make_cache_view(
    ggml_context* ctx,
    ggml_tensor* cache,
    int64_t head_dim,
    uint32_t n_kv_heads,
    int64_t seq_len,
    size_t offset_bytes) {
    return ggml_view_3d(
        ctx,
        cache,
        head_dim,
        static_cast<int64_t>(n_kv_heads),
        seq_len,
        head_dim * sizeof(float),
        head_dim * static_cast<size_t>(n_kv_heads) * sizeof(float),
        offset_bytes);
}

void write_kv_cache(
    ggml_context* ctx,
    ggml_tensor* k,
    ggml_tensor* v,
    KVCacheLayer* kv,
    int pos,
    int64_t head_dim,
    uint32_t n_kv_heads,
    int64_t seq_len) {
    const size_t write_offset =
        static_cast<size_t>(pos) * head_dim * static_cast<size_t>(n_kv_heads) * sizeof(float);

    ggml_tensor* k_cache_view = make_cache_view(ctx, kv->k, head_dim, n_kv_heads, seq_len, write_offset);
    ggml_tensor* v_cache_view = make_cache_view(ctx, kv->v, head_dim, n_kv_heads, seq_len, write_offset);
    ggml_tensor* k_copy = ggml_cpy(ctx, k, k_cache_view);
    ggml_tensor* v_copy = ggml_cpy(ctx, v, v_cache_view);

    ggml_cgraph* write_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(write_graph, k_copy);
    ggml_build_forward_expand(write_graph, v_copy);
    ggml_graph_compute_with_ctx(ctx, write_graph, 1);
}

ggml_tensor* permute_attention_tensor(ggml_context* ctx, ggml_tensor* tensor) {
    return ggml_cont(ctx, ggml_permute(ctx, tensor, 0, 2, 1, 3));
}

ggml_tensor* compute_attention_output(
    ggml_context* ctx,
    ggml_tensor* q,
    ggml_tensor* k_attn,
    ggml_tensor* v_attn,
    int64_t head_dim,
    uint32_t n_heads,
    int64_t seq_len,
    int64_t n_attn) {
    ggml_tensor* q_heads = permute_attention_tensor(ctx, q);
    ggml_tensor* k_heads = permute_attention_tensor(ctx, k_attn);
    ggml_tensor* v_heads = permute_attention_tensor(ctx, v_attn);

    ggml_tensor* scores = ggml_mul_mat(ctx, k_heads, q_heads);
    ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
    scores = ggml_scale(ctx, scores, 1.0f / std::sqrt(static_cast<float>(head_dim)));
    scores = ggml_diag_mask_inf(ctx, scores, n_attn - seq_len);
    scores = ggml_soft_max(ctx, scores);

    ggml_tensor* values_t = ggml_cont(ctx, ggml_transpose(ctx, v_heads));
    ggml_tensor* merged = ggml_mul_mat(ctx, values_t, scores);
    merged = ggml_cont(ctx, ggml_permute(ctx, merged, 0, 2, 1, 3));
    return ggml_reshape_2d(ctx, merged, head_dim * n_heads, seq_len);
}

}  // namespace

ggml_tensor* TransformerBlock::self_attention(ggml_tensor* x, int pos, KVCacheLayer* kv) {
    ggml_context* ctx = context_;
    const int64_t seq_len = x->ne[1];
    const int64_t head_dim = static_cast<int64_t>(kv->k->ne[0]);
    const int rope_dims = rope_dimensions_ == 0
        ? static_cast<int>(head_dim)
        : static_cast<int>(std::min<int64_t>(rope_dimensions_, head_dim));

    ggml_tensor* q = ggml_mul_mat(ctx, attn_weights_.wq, x);
    ggml_tensor* k = ggml_mul_mat(ctx, attn_weights_.wk, x);
    ggml_tensor* v = ggml_mul_mat(ctx, attn_weights_.wv, x);

    q = ggml_reshape_3d(ctx, q, head_dim, n_heads_, seq_len);
    k = ggml_reshape_3d(ctx, k, head_dim, n_kv_heads_, seq_len);
    v = ggml_reshape_3d(ctx, v, head_dim, n_kv_heads_, seq_len);

    ggml_tensor* positions = make_position_tensor(ctx, pos, seq_len);
    q = apply_rope(
        ctx, q, positions, rope_dims, rope_mode_, rope_n_ctx_orig_, rope_freq_base_, rope_freq_scale_);
    k = apply_rope(
        ctx, k, positions, rope_dims, rope_mode_, rope_n_ctx_orig_, rope_freq_base_, rope_freq_scale_);

    write_kv_cache(ctx, k, v, kv, pos, head_dim, n_kv_heads_, seq_len);

    const int64_t n_attn = static_cast<int64_t>(pos) + seq_len;
    ggml_tensor* k_attn = make_cache_view(ctx, kv->k, head_dim, n_kv_heads_, n_attn, 0);
    ggml_tensor* v_attn = make_cache_view(ctx, kv->v, head_dim, n_kv_heads_, n_attn, 0);
    ggml_tensor* attn_output =
        compute_attention_output(ctx, q, k_attn, v_attn, head_dim, n_heads_, seq_len, n_attn);
    return ggml_mul_mat(ctx, attn_weights_.wo, attn_output);
}

}  // namespace nano_llm_infer
