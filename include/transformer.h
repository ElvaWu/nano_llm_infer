#pragma once

#include <memory>

#include "gguf.h"
#include "gguf_loader.h"
#include "kv_cache.h"

namespace nano_llm_infer {

ggml_tensor* rms_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight, float eps);

class Transformer {
public:
    explicit Transformer(const Model& model);
    ~Transformer();

    Transformer(const Transformer&) = delete;
    Transformer& operator=(const Transformer&) = delete;
    Transformer(Transformer&&) noexcept;
    Transformer& operator=(Transformer&&) noexcept;

    void set_compute_ctx(ggml_context* ctx);
    ggml_tensor* forward_logits(ggml_tensor* input, int pos, KVCache* kv);
    ggml_tensor* forward_logits_batched(
        ggml_tensor* input,
        const std::vector<int>& positions,
        KVCache* kv);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nano_llm_infer
