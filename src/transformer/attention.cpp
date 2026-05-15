#include "internal.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "graph_inputs.h"

namespace nano_llm_infer {

namespace {

ggml_tensor* make_position_tensor(ggml_context* ctx, int pos, int64_t seq_len) {
    ggml_tensor* positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len);
    std::vector<int32_t> pos_data(static_cast<size_t>(seq_len));
    for (int64_t i = 0; i < seq_len; ++i) {
        pos_data[static_cast<size_t>(i)] = pos + static_cast<int>(i);
    }
    set_graph_input_data(positions, pos_data.data(), pos_data.size() * sizeof(int32_t));
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

size_t cache_slot_stride_bytes(const ggml_tensor* cache) {
    return static_cast<size_t>(cache->ne[0] * cache->ne[1] * cache->ne[2]) * sizeof(float);
}

size_t cache_pos_stride_bytes(int64_t head_dim, uint32_t n_kv_heads) {
    return static_cast<size_t>(head_dim) * n_kv_heads * sizeof(float);
}

ggml_tensor* make_cache_view(
    ggml_context* ctx,
    ggml_tensor* cache,
    int64_t head_dim,
    uint32_t n_kv_heads,
    int64_t seq_len,
    int slot,
    size_t offset_bytes) {
    const size_t slot_offset = static_cast<size_t>(slot) * cache_slot_stride_bytes(cache);
    return ggml_view_3d(
        ctx,
        cache,
        head_dim,
        static_cast<int64_t>(n_kv_heads),
        seq_len,
        head_dim * sizeof(float),
        head_dim * static_cast<size_t>(n_kv_heads) * sizeof(float),
        slot_offset + offset_bytes);
}

ggml_tensor* make_cache_batch_view(
    ggml_context* ctx,
    ggml_tensor* cache,
    int64_t head_dim,
    uint32_t n_kv_heads,
    int64_t seq_len,
    int batch_size) {
    // Expose the 4-D cache layout as [head_dim, seq, kv_heads, batch] for
    // batched attention kernels without copying the full cache tensor.
    return ggml_view_4d(
        ctx,
        cache,
        head_dim,
        seq_len,
        static_cast<int64_t>(n_kv_heads),
        batch_size,
        cache_pos_stride_bytes(head_dim, n_kv_heads),
        head_dim * sizeof(float),
        cache_slot_stride_bytes(cache),
        0);
}

ggml_tensor* make_batch_slot_view(
    ggml_context* ctx,
    ggml_tensor* tensor,
    int64_t ne0,
    int64_t ne1,
    int64_t ne2,
    int slot) {
    const size_t offset =
        static_cast<size_t>(slot) * ne0 * ne1 * ne2 * static_cast<int64_t>(sizeof(float));
    return ggml_view_3d(
        ctx,
        tensor,
        ne0,
        ne1,
        ne2,
        ne0 * sizeof(float),
        ne0 * static_cast<size_t>(ne1) * sizeof(float),
        offset);
}

ggml_tensor* make_batch_mask(ggml_context* ctx, const std::vector<int>& lengths, int64_t max_seq_len) {
    ggml_tensor* mask = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F16, max_seq_len, 1, 1, static_cast<int64_t>(lengths.size()));
    std::vector<ggml_fp16_t> data(static_cast<size_t>(max_seq_len) * lengths.size());
    for (size_t batch = 0; batch < lengths.size(); ++batch) {
        const int valid = lengths[batch];
        for (int64_t pos = 0; pos < max_seq_len; ++pos) {
            // Each batch element can have a different prefix length, so pad the
            // extra columns with -inf before the shared attention kernel runs.
            data[batch * static_cast<size_t>(max_seq_len) + static_cast<size_t>(pos)] =
                ggml_fp32_to_fp16(pos < valid ? 0.0f : -INFINITY);
        }
    }
    set_graph_input_data(mask, data.data(), data.size() * sizeof(ggml_fp16_t));
    return mask;
}

ggml_tensor* flatten_attention_output(
    ggml_context* ctx,
    ggml_tensor* attn_output,
    int64_t head_dim,
    uint32_t n_heads,
    int64_t batch_size) {
    ggml_tensor* contiguous = ggml_cont(ctx, attn_output);
    return ggml_reshape_2d(ctx, contiguous, head_dim * n_heads, batch_size);
}

ggml_tensor* compute_attention_output_batched(
    ggml_context* ctx,
    ggml_tensor* q,
    ggml_tensor* k_attn,
    ggml_tensor* v_attn,
    ggml_tensor* mask,
    int64_t head_dim,
    uint32_t n_heads,
    int64_t batch_size) {
    ggml_tensor* attn_output = ggml_flash_attn_ext(
        ctx,
        q,
        k_attn,
        v_attn,
        mask,
        1.0f / std::sqrt(static_cast<float>(head_dim)),
        0.0f,
        0.0f);
    ggml_flash_attn_ext_set_prec(attn_output, GGML_PREC_F32);
    return flatten_attention_output(ctx, attn_output, head_dim, n_heads, batch_size);
}

void write_kv_cache(
    ggml_context* ctx,
    ggml_tensor* k,
    ggml_tensor* v,
    KVCacheLayer* kv,
    int pos,
    int64_t head_dim,
    uint32_t n_kv_heads,
    int64_t seq_len,
    ggml_tensor** k_updated,
    ggml_tensor** v_updated) {
    const size_t write_offset = static_cast<size_t>(pos) * cache_pos_stride_bytes(head_dim, n_kv_heads);
    *k_updated = ggml_set_inplace(
        ctx,
        kv->k,
        k,
        head_dim * sizeof(float),
        head_dim * static_cast<size_t>(n_kv_heads) * sizeof(float),
        cache_slot_stride_bytes(kv->k),
        write_offset);
    *v_updated = ggml_set_inplace(
        ctx,
        kv->v,
        v,
        head_dim * sizeof(float),
        head_dim * static_cast<size_t>(n_kv_heads) * sizeof(float),
        cache_slot_stride_bytes(kv->v),
        write_offset);
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
    if (ctx == nullptr) {
        throw std::runtime_error("self_attention called without compute context");
    }
    if (x == nullptr || kv == nullptr || kv->k == nullptr || kv->v == nullptr) {
        throw std::invalid_argument("self_attention received null tensor or KV cache");
    }
    if (pos < 0) {
        throw std::invalid_argument("self_attention received negative position");
    }

    const int64_t seq_len = x->ne[1];
    if (seq_len <= 0) {
        throw std::invalid_argument("self_attention requires positive sequence length");
    }

    const int64_t cache_capacity = kv->k->ne[2];
    if (kv->v->ne[2] != cache_capacity) {
        throw std::runtime_error("KV cache K/V sequence dimensions do not match");
    }
    if (kv->k->ne[0] != kv->v->ne[0] || kv->k->ne[1] != kv->v->ne[1]) {
        throw std::runtime_error("KV cache K/V head dimensions do not match");
    }
    if (static_cast<int64_t>(pos) + seq_len > cache_capacity) {
        throw std::runtime_error("self_attention position exceeds KV cache capacity");
    }

    const int64_t head_dim = static_cast<int64_t>(kv->k->ne[0]);
    if (head_dim <= 0 || n_heads_ == 0 || n_kv_heads_ == 0) {
        throw std::runtime_error("self_attention received invalid attention dimensions");
    }
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

    ggml_tensor* k_updated = nullptr;
    ggml_tensor* v_updated = nullptr;
    write_kv_cache(
        ctx,
        k,
        v,
        kv,
        pos,
        head_dim,
        n_kv_heads_,
        seq_len,
        &k_updated,
        &v_updated);

    const int64_t n_attn = static_cast<int64_t>(pos) + seq_len;
    ggml_tensor* k_attn = make_cache_view(ctx, k_updated, head_dim, n_kv_heads_, n_attn, 0, 0);
    ggml_tensor* v_attn = make_cache_view(ctx, v_updated, head_dim, n_kv_heads_, n_attn, 0, 0);
    ggml_tensor* attn_output =
        compute_attention_output(ctx, q, k_attn, v_attn, head_dim, n_heads_, seq_len, n_attn);
    return ggml_mul_mat(ctx, attn_weights_.wo, attn_output);
}

ggml_tensor* TransformerBlock::self_attention_batched(
    ggml_tensor* x,
    const std::vector<int>& positions,
    KVCacheLayer* kv) {
    ggml_context* ctx = context_;
    if (ctx == nullptr) {
        throw std::runtime_error("self_attention_batched called without compute context");
    }
    if (x == nullptr || kv == nullptr || kv->k == nullptr || kv->v == nullptr) {
        throw std::invalid_argument("self_attention_batched received null tensor or KV cache");
    }
    if (positions.empty()) {
        throw std::invalid_argument("self_attention_batched requires at least one position");
    }

    const int64_t batch_size = x->ne[1];
    if (batch_size != static_cast<int64_t>(positions.size())) {
        throw std::invalid_argument("self_attention_batched batch size does not match positions size");
    }

    const int64_t cache_capacity = kv->k->ne[2];
    if (kv->v->ne[2] != cache_capacity) {
        throw std::runtime_error("batched KV cache K/V sequence dimensions do not match");
    }
    if (kv->k->ne[0] != kv->v->ne[0] || kv->k->ne[1] != kv->v->ne[1]) {
        throw std::runtime_error("batched KV cache K/V head dimensions do not match");
    }

    const int64_t head_dim = static_cast<int64_t>(kv->k->ne[0]);
    if (head_dim <= 0 || n_heads_ == 0 || n_kv_heads_ == 0) {
        throw std::runtime_error("self_attention_batched received invalid attention dimensions");
    }
    const int rope_dims = rope_dimensions_ == 0
        ? static_cast<int>(head_dim)
        : static_cast<int>(std::min<int64_t>(rope_dimensions_, head_dim));

    ggml_tensor* q = ggml_mul_mat(ctx, attn_weights_.wq, x);
    ggml_tensor* k = ggml_mul_mat(ctx, attn_weights_.wk, x);
    ggml_tensor* v = ggml_mul_mat(ctx, attn_weights_.wv, x);

    q = ggml_reshape_4d(ctx, q, head_dim, n_heads_, 1, batch_size);
    k = ggml_reshape_4d(ctx, k, head_dim, n_kv_heads_, 1, batch_size);
    v = ggml_reshape_4d(ctx, v, head_dim, n_kv_heads_, 1, batch_size);

    ggml_tensor* q_roped = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_heads_, 1, batch_size);
    ggml_tensor* k_updated = kv->k;
    ggml_tensor* v_updated = kv->v;

    int max_attn = 0;
    std::vector<int> lengths;
    lengths.reserve(positions.size());

    for (size_t i = 0; i < positions.size(); ++i) {
        const int pos = positions[i];
        if (pos < 0) {
            throw std::invalid_argument("self_attention_batched received negative position");
        }
        if (static_cast<int64_t>(pos) + 1 > cache_capacity) {
            throw std::runtime_error("self_attention_batched position exceeds KV cache capacity");
        }

        const int n_attn = pos + 1;
        max_attn = std::max(max_attn, n_attn);
        lengths.push_back(n_attn);

        ggml_tensor* q_slot = make_batch_slot_view(ctx, q, head_dim, n_heads_, 1, static_cast<int>(i));
        ggml_tensor* k_slot = make_batch_slot_view(ctx, k, head_dim, n_kv_heads_, 1, static_cast<int>(i));
        ggml_tensor* v_slot = make_batch_slot_view(ctx, v, head_dim, n_kv_heads_, 1, static_cast<int>(i));

        ggml_tensor* position = make_position_tensor(ctx, pos, 1);
        ggml_tensor* q_rope = apply_rope(
            ctx, q_slot, position, rope_dims, rope_mode_, rope_n_ctx_orig_, rope_freq_base_, rope_freq_scale_);
        ggml_tensor* k_rope = apply_rope(
            ctx, k_slot, position, rope_dims, rope_mode_, rope_n_ctx_orig_, rope_freq_base_, rope_freq_scale_);

        q_roped = ggml_set_inplace(
            ctx,
            q_roped,
            q_rope,
            head_dim * sizeof(float),
            head_dim * static_cast<size_t>(n_heads_) * sizeof(float),
            head_dim * static_cast<size_t>(n_heads_) * sizeof(float),
            static_cast<size_t>(i) * head_dim * n_heads_ * sizeof(float));

        const size_t write_offset = static_cast<size_t>(pos) * cache_pos_stride_bytes(head_dim, n_kv_heads_);
        k_updated = ggml_set_inplace(
            ctx,
            k_updated,
            k_rope,
            head_dim * sizeof(float),
            head_dim * static_cast<size_t>(n_kv_heads_) * sizeof(float),
            cache_slot_stride_bytes(kv->k),
            static_cast<size_t>(i) * cache_slot_stride_bytes(kv->k) + write_offset);
        v_updated = ggml_set_inplace(
            ctx,
            v_updated,
            v_slot,
            head_dim * sizeof(float),
            head_dim * static_cast<size_t>(n_kv_heads_) * sizeof(float),
            cache_slot_stride_bytes(kv->v),
            static_cast<size_t>(i) * cache_slot_stride_bytes(kv->v) + write_offset);
    }

    ggml_tensor* q_attn = ggml_cont(ctx, ggml_permute(ctx, q_roped, 0, 2, 1, 3));
    ggml_tensor* k_attn = make_cache_batch_view(ctx, k_updated, head_dim, n_kv_heads_, max_attn, static_cast<int>(batch_size));
    ggml_tensor* v_attn = make_cache_batch_view(ctx, v_updated, head_dim, n_kv_heads_, max_attn, static_cast<int>(batch_size));
    ggml_tensor* mask = make_batch_mask(ctx, lengths, max_attn);

    ggml_tensor* attn_output = compute_attention_output_batched(
        ctx, q_attn, k_attn, v_attn, mask, head_dim, n_heads_, batch_size);
    return ggml_mul_mat(ctx, attn_weights_.wo, attn_output);
}

}  // namespace nano_llm_infer
