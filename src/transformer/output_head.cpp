#include "internal.h"

#include <stdexcept>

namespace nano_llm_infer {

OutputHead::OutputHead(const Model& model)
    : eps_(model.rms_norm_eps), context_(model.ctx) {
    norm_weight_ = get_tensor_by_name(model, "output_norm.weight");
    lm_head_weight_ = get_tensor_by_name(model, "output.weight");
    if (norm_weight_ == nullptr || lm_head_weight_ == nullptr) {
        throw std::runtime_error("Missing required output-head tensor");
    }
}

ggml_tensor* OutputHead::forward_logits(ggml_tensor* hidden) {
    ggml_tensor* normed = rms_norm(context_, hidden, norm_weight_, eps_);
    return ggml_mul_mat(context_, lm_head_weight_, normed);
}

ggml_tensor* OutputHead::forward_last_token_logits(ggml_tensor* hidden) {
    if (hidden == nullptr || hidden->ne[1] <= 0) {
        return nullptr;
    }
    // Prefill only samples from the last prompt position. Slice hidden state
    // first to avoid allocating vocabulary logits for every prompt token.
    ggml_tensor* last_hidden = ggml_view_2d(
        context_,
        hidden,
        hidden->ne[0],
        1,
        hidden->nb[1],
        static_cast<size_t>(hidden->ne[1] - 1) * hidden->nb[1]);
    return forward_logits(last_hidden);
}

}  // namespace nano_llm_infer
