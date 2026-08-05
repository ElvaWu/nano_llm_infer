#pragma once

#include <vector>

#include "ggml-cpu.h"
#include "transformer.h"

namespace nano_llm_infer {

struct AttentionWeights {
    ggml_tensor* wq = nullptr;
    ggml_tensor* wk = nullptr;
    ggml_tensor* wv = nullptr;
    ggml_tensor* wo = nullptr;
    ggml_tensor* bq = nullptr;
    ggml_tensor* bk = nullptr;
    ggml_tensor* bv = nullptr;
};

struct MLPWeights {
    ggml_tensor* gate_proj = nullptr;
    ggml_tensor* up_proj = nullptr;
    ggml_tensor* down_proj = nullptr;
};

class OutputHead {
public:
    explicit OutputHead(const Model& model);

    void set_compute_ctx(ggml_context* ctx) { context_ = ctx; }
    ggml_tensor* forward_logits(ggml_tensor* hidden);
    ggml_tensor* forward_last_token_logits(ggml_tensor* hidden);

private:
    float eps_ = 1e-5f;
    ggml_context* context_ = nullptr;
    ggml_tensor* norm_weight_ = nullptr;
    ggml_tensor* lm_head_weight_ = nullptr;
};

class TransformerBlock {
public:
    TransformerBlock(const Model& model, uint32_t layer_idx);

    void set_compute_ctx(ggml_context* ctx) { context_ = ctx; }
    ggml_tensor* forward(ggml_tensor* input, int pos, int kv_slot, KVCacheLayer* kv);
    ggml_tensor* forward_batched(
        ggml_tensor* input, const TransformerBatchInputs& inputs, KVCacheLayer* kv);
    ggml_tensor* self_attention(ggml_tensor* x, int pos, int kv_slot, KVCacheLayer* kv);
    ggml_tensor* self_attention_batched(
        ggml_tensor* x, const TransformerBatchInputs& inputs, KVCacheLayer* kv);

private:
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
    MLPWeights mlp_weights_;
    AttentionWeights attn_weights_;
    ggml_context* context_ = nullptr;
};

class Transformer::Impl {
public:
    explicit Impl(const Model& model);

    void set_compute_ctx(ggml_context* ctx);
    ggml_tensor* forward_logits(ggml_tensor* input, int pos, int kv_slot, KVCache* kv);
    ggml_tensor* forward_logits_batched(
        ggml_tensor* input, const TransformerBatchInputs& inputs, KVCache* kv);

private:
    std::vector<TransformerBlock> blocks_;
    OutputHead output_head_;
};

AttentionWeights load_attention_weights(const Model& model, uint32_t layer_idx);

}  // namespace nano_llm_infer
