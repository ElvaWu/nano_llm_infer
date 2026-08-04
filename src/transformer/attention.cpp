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

ggml_tensor* TransformerBlock::self_attention(ggml_tensor* x, int pos, int kv_slot, KVCacheLayer* kv) {
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
    if (kv_slot < 0 || kv_slot >= kv->k->ne[3]) {
        throw std::invalid_argument("self_attention received invalid KV slot");
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
    // Qwen2 adds per-output Q/K/V biases; Llama leaves these optional tensors null.
    if (attn_weights_.bq != nullptr) {
        q = ggml_add(ctx, q, attn_weights_.bq);
    }
    if (attn_weights_.bk != nullptr) {
        k = ggml_add(ctx, k, attn_weights_.bk);
    }
    if (attn_weights_.bv != nullptr) {
        v = ggml_add(ctx, v, attn_weights_.bv);
    }

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
    const size_t write_offset =
        static_cast<size_t>(kv_slot) * cache_slot_stride_bytes(kv->k) +
        static_cast<size_t>(pos) * cache_pos_stride_bytes(head_dim, n_kv_heads_);
    k_updated = ggml_set_inplace(
        ctx,
        kv->k,
        k,
        head_dim * sizeof(float),
        head_dim * static_cast<size_t>(n_kv_heads_) * sizeof(float),
        cache_slot_stride_bytes(kv->k),
        write_offset);
    v_updated = ggml_set_inplace(
        ctx,
        kv->v,
        v,
        head_dim * sizeof(float),
        head_dim * static_cast<size_t>(n_kv_heads_) * sizeof(float),
        cache_slot_stride_bytes(kv->v),
        write_offset);

    const int64_t n_attn = static_cast<int64_t>(pos) + seq_len;
    ggml_tensor* k_attn = make_cache_view(ctx, k_updated, head_dim, n_kv_heads_, n_attn, kv_slot, 0);
    ggml_tensor* v_attn = make_cache_view(ctx, v_updated, head_dim, n_kv_heads_, n_attn, kv_slot, 0);
    ggml_tensor* attn_output =
        compute_attention_output(ctx, q, k_attn, v_attn, head_dim, n_heads_, seq_len, n_attn);
    return ggml_mul_mat(ctx, attn_weights_.wo, attn_output);
}

ggml_tensor* TransformerBlock::self_attention_batched(
    ggml_tensor* x, const TransformerBatchInputs& inputs, KVCacheLayer* kv) {
    ggml_context* ctx = context_;
    if (ctx == nullptr) {
        throw std::runtime_error("self_attention_batched called without compute context");
    }
    if (x == nullptr || kv == nullptr || kv->k == nullptr || kv->v == nullptr) {
        throw std::invalid_argument("self_attention_batched received null tensor or KV cache");
    }

    const int64_t batch_size = x->ne[1];
    const int64_t attention_length = inputs.attention_length;
    const int64_t cache_capacity = kv->k->ne[2];
    const int64_t cache_slots = kv->k->ne[3];
    if (batch_size < 1 || attention_length < 1 || attention_length > cache_capacity) {
        throw std::invalid_argument("self_attention_batched received invalid batch or attention length");
    }
    if (kv->v->ne[2] != cache_capacity || kv->v->ne[3] != cache_slots ||
        kv->k->ne[0] != kv->v->ne[0] || kv->k->ne[1] != kv->v->ne[1]) {
        throw std::runtime_error("batched KV cache dimensions do not match");
    }

    const int64_t head_dim = kv->k->ne[0];
    if (head_dim <= 0 || n_heads_ == 0 || n_kv_heads_ == 0) {
        throw std::runtime_error("self_attention_batched received invalid attention dimensions");
    }
    if (inputs.positions->type != GGML_TYPE_I32 || inputs.positions->ne[0] != batch_size ||
        inputs.kv_write_rows->type != GGML_TYPE_I32 ||
        inputs.kv_write_rows->ne[0] != batch_size * n_kv_heads_ ||
        inputs.kv_read_rows->type != GGML_TYPE_I32 ||
        inputs.kv_read_rows->ne[0] != batch_size * n_kv_heads_ * attention_length ||
        inputs.attention_mask->type != GGML_TYPE_F16 ||
        inputs.attention_mask->ne[0] != attention_length ||
        inputs.attention_mask->ne[3] != batch_size) {
        throw std::invalid_argument("self_attention_batched input shapes do not match");
    }
    const int rope_dims = rope_dimensions_ == 0
        ? static_cast<int>(head_dim)
        : static_cast<int>(std::min<int64_t>(rope_dimensions_, head_dim));

    ggml_tensor* q = ggml_mul_mat(ctx, attn_weights_.wq, x);
    ggml_tensor* k = ggml_mul_mat(ctx, attn_weights_.wk, x);
    ggml_tensor* v = ggml_mul_mat(ctx, attn_weights_.wv, x);
    if (attn_weights_.bq != nullptr) {
        q = ggml_add(ctx, q, attn_weights_.bq);
    }
    if (attn_weights_.bk != nullptr) {
        k = ggml_add(ctx, k, attn_weights_.bk);
    }
    if (attn_weights_.bv != nullptr) {
        v = ggml_add(ctx, v, attn_weights_.bv);
    }

    q = ggml_reshape_3d(ctx, q, head_dim, n_heads_, batch_size);
    k = ggml_reshape_3d(ctx, k, head_dim, n_kv_heads_, batch_size);
    v = ggml_reshape_3d(ctx, v, head_dim, n_kv_heads_, batch_size);
    q = apply_rope(
        ctx, q, inputs.positions, rope_dims, rope_mode_, rope_n_ctx_orig_, rope_freq_base_, rope_freq_scale_);
    k = apply_rope(
        ctx, k, inputs.positions, rope_dims, rope_mode_, rope_n_ctx_orig_, rope_freq_base_, rope_freq_scale_);

    const int64_t cache_rows = n_kv_heads_ * cache_capacity * cache_slots;
    ggml_tensor* k_cache = ggml_reshape_2d(ctx, kv->k, head_dim, cache_rows);
    ggml_tensor* v_cache = ggml_reshape_2d(ctx, kv->v, head_dim, cache_rows);
    ggml_tensor* k_rows = ggml_reshape_2d(ctx, k, head_dim, n_kv_heads_ * batch_size);
    ggml_tensor* v_rows = ggml_reshape_2d(ctx, v, head_dim, n_kv_heads_ * batch_size);
    ggml_tensor* k_updated = ggml_set_rows(ctx, k_cache, k_rows, inputs.kv_write_rows);
    ggml_tensor* v_updated = ggml_set_rows(ctx, v_cache, v_rows, inputs.kv_write_rows);

    ggml_tensor* k_gathered = ggml_get_rows(ctx, k_updated, inputs.kv_read_rows);
    ggml_tensor* v_gathered = ggml_get_rows(ctx, v_updated, inputs.kv_read_rows);
    ggml_tensor* k_attn = ggml_reshape_4d(
        ctx, k_gathered, head_dim, attention_length, n_kv_heads_, batch_size);
    ggml_tensor* v_attn = ggml_reshape_4d(
        ctx, v_gathered, head_dim, attention_length, n_kv_heads_, batch_size);
    ggml_tensor* q_attn = ggml_cont(
        ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, q, head_dim, n_heads_, 1, batch_size), 0, 2, 1, 3));

    ggml_tensor* attn_output = compute_attention_output_batched(
        ctx,
        q_attn,
        k_attn,
        v_attn,
        inputs.attention_mask,
        head_dim,
        n_heads_,
        batch_size);
    return ggml_mul_mat(ctx, attn_weights_.wo, attn_output);
}

}  // namespace nano_llm_infer
