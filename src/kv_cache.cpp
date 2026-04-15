#include "kv_cache.h"

#include <cstddef>
#include <cstring>
#include <stdexcept>

namespace nano_llm_infer {

KVCache make_kv_cache(const Model& model, int max_seq_len) {
    if (model.n_heads == 0 || model.n_kv_heads == 0) {
        throw std::runtime_error("Model not loaded or missing head counts");
    }

    const int head_dim = static_cast<int>(
        get_tensor_by_name(model, "token_embd.weight")->ne[0] / model.n_heads);
    const int n_kv_heads = static_cast<int>(model.n_kv_heads);
    const int n_layers = static_cast<int>(model.n_layers);

    // Each layer: 2 tensors of [head_dim, n_kv_heads, max_seq_len] f32
    const size_t layer_bytes =
        2ull * static_cast<size_t>(head_dim) * n_kv_heads * max_seq_len * sizeof(float);
    const size_t ctx_bytes = static_cast<size_t>(n_layers) * layer_bytes + 2 * 1024 * 1024;

    ggml_init_params params = {
        /* mem_size   */ ctx_bytes,
        /* mem_buffer */ nullptr,
        /* no_alloc   */ false,
    };

    KVCache kv;
    kv.ctx = ggml_init(params);
    kv.n_filled = 0;
    kv.max_seq_len = max_seq_len;
    kv.head_dim = head_dim;
    kv.n_kv_heads = n_kv_heads;
    kv.layers.resize(static_cast<size_t>(n_layers));

    for (int i = 0; i < n_layers; ++i) {
        kv.layers[static_cast<size_t>(i)].k =
            ggml_new_tensor_3d(kv.ctx, GGML_TYPE_F32, head_dim, n_kv_heads, max_seq_len);
        kv.layers[static_cast<size_t>(i)].v =
            ggml_new_tensor_3d(kv.ctx, GGML_TYPE_F32, head_dim, n_kv_heads, max_seq_len);

        // Zero-initialize the cache to avoid reading garbage
        const size_t tensor_bytes = static_cast<size_t>(head_dim * n_kv_heads * max_seq_len) * sizeof(float);
        std::memset(kv.layers[static_cast<size_t>(i)].k->data, 0, tensor_bytes);
        std::memset(kv.layers[static_cast<size_t>(i)].v->data, 0, tensor_bytes);
    }

    return kv;
}

void free_kv_cache(KVCache& kv) {
    if (kv.ctx) {
        ggml_free(kv.ctx);
        kv.ctx = nullptr;
    }
    kv.layers.clear();
    kv.n_filled = 0;
}

}  // namespace nano_llm_infer
