#pragma once

#include <vector>

#include "gguf.h"
#include "gguf_loader.h"
#include "kv_cache.h"

namespace nano_llm_infer {

struct AttentionWeights {
    ggml_tensor* wq = nullptr;
    ggml_tensor* wk = nullptr;
    ggml_tensor* wv = nullptr;
    ggml_tensor* wo = nullptr;
};

struct MLPWeights {
    ggml_tensor* gate_proj = nullptr;
    ggml_tensor* up_proj = nullptr;
    ggml_tensor* down_proj = nullptr;
};

ggml_tensor* rms_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight, float eps);

class OutputHead {
public:
    explicit OutputHead(const Model& model);

    void set_compute_ctx(ggml_context* ctx) { context_ = ctx; }
    ggml_tensor* forward_logits(ggml_tensor* hidden);

private:
    float eps_ = 1e-5f;
    ggml_context* context_ = nullptr;
    ggml_tensor* norm_weight_ = nullptr;
    ggml_tensor* lm_head_weight_ = nullptr;
};

class TransformerBlock {
public:
    TransformerBlock(const Model& model, uint32_t layer_idx);
    ~TransformerBlock();

    void set_compute_ctx(ggml_context* ctx) { context_ = ctx; }
    ggml_tensor* forward(ggml_tensor* input, int pos, ggml_cgraph* gf, KVCacheLayer* kv);

private:
    ggml_tensor* self_attention(
        ggml_tensor* x, AttentionWeights* w, int pos, ggml_cgraph* gf, KVCacheLayer* kv);
    ggml_tensor* mlp(ggml_tensor* input);

    float eps_ = 1e-5f;
    uint32_t n_heads_ = 0;
    uint32_t n_kv_heads_ = 0;
    uint32_t rope_n_ctx_orig_ = 0;
    uint32_t rope_dimensions_ = 0;
    int rope_mode_ = GGML_ROPE_TYPE_NORMAL;
    float rope_freq_base_ = 10000.0f;
    float rope_freq_scale_ = 1.0f;
    ggml_tensor* attn_norm_weight_ = nullptr;
    ggml_tensor* ffn_norm_weight_ = nullptr;
    MLPWeights* mlp_weights_ = nullptr;
    AttentionWeights* attn_weights_ = nullptr;
    ggml_context* context_ = nullptr;
};

class Transformer {
public:
    explicit Transformer(const Model& model);

    void set_compute_ctx(ggml_context* ctx);
    ggml_tensor* forward_logits(ggml_tensor* input, int pos, ggml_cgraph* gf, KVCache* kv);

private:
    std::vector<TransformerBlock> blocks_;
    OutputHead output_head_;
};

}  // namespace nano_llm_infer
