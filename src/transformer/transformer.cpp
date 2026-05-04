#include "transformer.h"

#include <utility>

#include "internal.h"

namespace nano_llm_infer {

ggml_tensor* rms_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight, float eps) {
    ggml_tensor* norm = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, norm, weight);
}

Transformer::Impl::Impl(const Model& model)
    : output_head_(model) {
    blocks_.reserve(model.n_layers);
    for (uint32_t i = 0; i < model.n_layers; ++i) {
        blocks_.emplace_back(model, i);
    }
}

void Transformer::Impl::set_compute_ctx(ggml_context* ctx) {
    for (auto& block : blocks_) {
        block.set_compute_ctx(ctx);
    }
    output_head_.set_compute_ctx(ctx);
}

ggml_tensor* Transformer::Impl::forward_logits(
    ggml_tensor* input, int pos, KVCache* kv) {
    ggml_tensor* hidden = input;
    for (size_t i = 0; i < blocks_.size(); ++i) {
        hidden = blocks_[i].forward(hidden, pos, &kv->layers[i]);
    }
    return output_head_.forward_logits(hidden);
}

ggml_tensor* Transformer::Impl::forward_logits_batched(
    ggml_tensor* input,
    const std::vector<int>& positions,
    KVCache* kv) {
    ggml_tensor* hidden = input;
    for (size_t i = 0; i < blocks_.size(); ++i) {
        hidden = blocks_[i].forward_batched(hidden, positions, &kv->layers[i]);
    }
    return output_head_.forward_logits(hidden);
}

Transformer::Transformer(const Model& model)
    : impl_(std::make_unique<Impl>(model)) {
}

Transformer::~Transformer() = default;

Transformer::Transformer(Transformer&&) noexcept = default;

Transformer& Transformer::operator=(Transformer&&) noexcept = default;

void Transformer::set_compute_ctx(ggml_context* ctx) {
    impl_->set_compute_ctx(ctx);
}

ggml_tensor* Transformer::forward_logits(ggml_tensor* input, int pos, KVCache* kv) {
    return impl_->forward_logits(input, pos, kv);
}

ggml_tensor* Transformer::forward_logits_batched(
    ggml_tensor* input,
    const std::vector<int>& positions,
    KVCache* kv) {
    return impl_->forward_logits_batched(input, positions, kv);
}

}  // namespace nano_llm_infer
