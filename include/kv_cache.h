#pragma once

#include <vector>

#include "gguf.h"
#include "gguf_loader.h"

namespace nano_llm_infer {

struct KVCacheLayer {
    ggml_tensor* k = nullptr;  // [head_dim, n_kv_heads, max_seq_len] f32
    ggml_tensor* v = nullptr;  // [head_dim, n_kv_heads, max_seq_len] f32
};

struct KVCache {
    ggml_context* ctx = nullptr;
    std::vector<KVCacheLayer> layers;
    int n_filled = 0;
    int max_seq_len = 0;
    int head_dim = 0;
    int n_kv_heads = 0;
};

KVCache make_kv_cache(const Model& model, int max_seq_len);
void free_kv_cache(KVCache& kv);

}  // namespace nano_llm_infer
