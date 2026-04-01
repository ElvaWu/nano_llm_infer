#include "gguf_loader.h"
#include "gguf.h"

namespace nano_llm_infer{

struct AttentionWeights {
    ggml_tensor* wq;
    ggml_tensor* wk;
    ggml_tensor* wv;
    ggml_tensor* wo;
};

struct MLPWeights {
    ggml_tensor* gate_proj;
    ggml_tensor* up_proj;
    ggml_tensor* down_proj;
};

ggml_tensor* rms_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight, float eps);

class OutputHead {
public:
    OutputHead(const Model& model);

    ggml_tensor* forward(ggml_tensor* hidden);

private:
    float eps_;
    ggml_context* context_;
    ggml_tensor* norm_weight_;
    ggml_tensor* lm_head_weight_;
};

class TransformerBlock{
public:
    TransformerBlock(const Model& model);
    ~TransformerBlock();

    ggml_tensor* forward(ggml_tensor* input);
private:
    float eps_;
    uint32_t n_heads_;
    uint32_t n_kv_heads_;
    ggml_tensor* attn_norm_weight_;
    ggml_tensor* ffn_norm_weight_;

    MLPWeights* mlp_weights_;
    AttentionWeights* attn_weights_;

    ggml_tensor *self_attention(ggml_tensor *x, AttentionWeights *w);
    ggml_tensor* mlp(ggml_tensor* input);
    ggml_context* context_;

};
} // namespace nano_llm_infer