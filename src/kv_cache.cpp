#include "kv_cache.h"

#include <cstddef>
#include <stdexcept>

#include "ggml-alloc.h"

namespace nano_llm_infer {

KVCache make_kv_cache(
    const Model& model,
    int max_seq_len,
    int batch_size,
    ggml_backend_buffer_type_t buffer_type) {
    if (model.n_heads == 0 || model.n_kv_heads == 0) {
        throw std::runtime_error("Model not loaded or missing head counts");
    }
    if (batch_size < 1) {
        throw std::runtime_error("KV cache batch size must be positive");
    }

    const int head_dim = static_cast<int>(
        get_tensor_by_name(model, "token_embd.weight")->ne[0] / model.n_heads);
    const int n_kv_heads = static_cast<int>(model.n_kv_heads);
    const int n_layers = static_cast<int>(model.n_layers);

    // Each layer: 2 tensors of [head_dim, n_kv_heads, max_seq_len, batch_size] f32
    const size_t layer_bytes =
        2ull * static_cast<size_t>(head_dim) * n_kv_heads * max_seq_len * batch_size * sizeof(float);
    const size_t ctx_bytes = static_cast<size_t>(n_layers) * layer_bytes + 2 * 1024 * 1024;

    ggml_init_params params = {
        /* mem_size   */ ctx_bytes,
        /* mem_buffer */ nullptr,
        /* no_alloc   */ buffer_type != nullptr,
    };

    KVCache kv;
    kv.ctx = ggml_init(params);
    kv.n_filled = 0;
    kv.max_seq_len = max_seq_len;
    kv.head_dim = head_dim;
    kv.n_kv_heads = n_kv_heads;
    kv.batch_size = batch_size;
    kv.layers.resize(static_cast<size_t>(n_layers));

    for (int i = 0; i < n_layers; ++i) {
        kv.layers[static_cast<size_t>(i)].k =
            ggml_new_tensor_4d(kv.ctx, GGML_TYPE_F32, head_dim, n_kv_heads, max_seq_len, batch_size);
        kv.layers[static_cast<size_t>(i)].v =
            ggml_new_tensor_4d(kv.ctx, GGML_TYPE_F32, head_dim, n_kv_heads, max_seq_len, batch_size);

    }

    if (buffer_type != nullptr) {
        kv.buffer = ggml_backend_alloc_ctx_tensors_from_buft(kv.ctx, buffer_type);
        if (kv.buffer == nullptr) {
            free_kv_cache(kv);
            throw std::runtime_error("failed to allocate KV cache backend buffer");
        }
    }

    const size_t tensor_bytes =
        static_cast<size_t>(head_dim) * n_kv_heads * max_seq_len * batch_size * sizeof(float);
    for (int i = 0; i < n_layers; ++i) {
        // Zero-initialize the cache to avoid reading garbage.
        ggml_backend_tensor_memset(kv.layers[static_cast<size_t>(i)].k, 0, 0, tensor_bytes);
        ggml_backend_tensor_memset(kv.layers[static_cast<size_t>(i)].v, 0, 0, tensor_bytes);
    }

    return kv;
}

void free_kv_cache(KVCache& kv) {
    if (kv.buffer) {
        ggml_backend_buffer_free(kv.buffer);
        kv.buffer = nullptr;
    }
    if (kv.ctx) {
        ggml_free(kv.ctx);
        kv.ctx = nullptr;
    }
    kv.layers.clear();
    kv.n_filled = 0;
    kv.batch_size = 1;
}

}  // namespace nano_llm_infer
