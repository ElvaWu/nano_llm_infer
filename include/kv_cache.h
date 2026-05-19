#pragma once

#include <vector>

#include "gguf.h"
#include "ggml-backend.h"
#include "gguf_loader.h"

namespace nano_llm_infer {

struct KVCacheLayer {
    ggml_tensor* k = nullptr;  // [head_dim, n_kv_heads, max_seq_len, batch_size] f32
    ggml_tensor* v = nullptr;  // [head_dim, n_kv_heads, max_seq_len, batch_size] f32
};

struct KVCache {
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::vector<KVCacheLayer> layers;
    int n_filled = 0;
    int max_seq_len = 0;
    int head_dim = 0;
    int n_kv_heads = 0;
    int batch_size = 1;
};

KVCache make_kv_cache(
    const Model& model,
    int max_seq_len,
    int batch_size = 1,
    ggml_backend_buffer_type_t buffer_type = nullptr);
void free_kv_cache(KVCache& kv);

}  // namespace nano_llm_infer
