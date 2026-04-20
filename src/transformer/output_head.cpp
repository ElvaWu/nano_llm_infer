#include "internal.h"

namespace nano_llm_infer {

OutputHead::OutputHead(const Model& model)
    : eps_(model.rms_norm_eps), context_(model.ctx) {
    norm_weight_ = get_tensor_by_name(model, "output_norm.weight");
    lm_head_weight_ = get_tensor_by_name(model, "output.weight");
}

ggml_tensor* OutputHead::forward_logits(ggml_tensor* hidden) {
    ggml_tensor* normed = rms_norm(context_, hidden, norm_weight_, eps_);
    return ggml_mul_mat(context_, lm_head_weight_, normed);
}

}  // namespace nano_llm_infer
