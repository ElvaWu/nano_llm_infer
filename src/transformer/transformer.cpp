#include "transformer.h"

#include <stdexcept>
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
    ggml_tensor* input, int pos, int kv_slot, KVCache* kv) {
    if (input == nullptr) {
        throw std::invalid_argument("forward_logits received null input tensor");
    }
    if (kv == nullptr) {
        throw std::invalid_argument("forward_logits received null KV cache");
    }
    if (kv->layers.size() != blocks_.size()) {
        throw std::runtime_error("KV cache layer count does not match transformer depth");
    }
    if (pos < 0) {
        throw std::invalid_argument("forward_logits received negative position");
    }
    if (kv_slot < 0 || kv_slot >= kv->batch_size) {
        throw std::invalid_argument("forward_logits received invalid KV slot");
    }

    ggml_tensor* hidden = input;
    for (size_t i = 0; i < blocks_.size(); ++i) {
        hidden = blocks_[i].forward(hidden, pos, kv_slot, &kv->layers[i]);
    }
    return output_head_.forward_last_token_logits(hidden);
}


ggml_tensor* Transformer::Impl::forward_logits_batched(
    ggml_tensor* input,
    const std::vector<int>& positions,
    const std::vector<int>& kv_slots,
    KVCache* kv) {
    if (input == nullptr) {
        throw std::invalid_argument("forward_logits_batched received null input tensor");
    }
    if (kv == nullptr) {
        throw std::invalid_argument("forward_logits_batched received null KV cache");
    }
    if (kv->layers.size() != blocks_.size()) {
        throw std::runtime_error("KV cache layer count does not match transformer depth");
    }
    if (positions.empty()) {
        throw std::invalid_argument("forward_logits_batched requires at least one position");
    }
    if (kv_slots.size() != positions.size()) {
        throw std::invalid_argument("forward_logits_batched KV slot count does not match positions size");
    }
    if (input->ne[1] != static_cast<int64_t>(positions.size())) {
        throw std::invalid_argument("forward_logits_batched batch size does not match positions size");
    }
    for (size_t i = 0; i < positions.size(); ++i) {
        if (positions[i] < 0) {
            throw std::invalid_argument("forward_logits_batched received negative position");
        }
        if (kv_slots[i] < 0 || kv_slots[i] >= kv->batch_size) {
            throw std::invalid_argument("forward_logits_batched received invalid KV slot");
        }
    }

    ggml_tensor* hidden = input;
    for (size_t i = 0; i < blocks_.size(); ++i) {
        hidden = blocks_[i].forward_batched(hidden, positions, kv_slots, &kv->layers[i]);
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

ggml_tensor* Transformer::forward_logits(ggml_tensor* input, int pos, int kv_slot, KVCache* kv) {
    return impl_->forward_logits(input, pos, kv_slot, kv);
}

ggml_tensor* Transformer::forward_logits_batched(
    ggml_tensor* input,
    const std::vector<int>& positions,
    const std::vector<int>& kv_slots,
    KVCache* kv) {
    return impl_->forward_logits_batched(input, positions, kv_slots, kv);
}

}  // namespace nano_llm_infer
