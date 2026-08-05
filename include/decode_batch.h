#pragma once

#include <cstdint>
#include <vector>

#include "ggml.h"

namespace nano_llm_infer {

struct DecodeBatchInputData {
    std::vector<int32_t> positions;
    std::vector<int32_t> kv_write_rows;
    std::vector<int32_t> kv_read_rows;
    std::vector<ggml_fp16_t> attention_mask;
    int attention_length = 0;
};

DecodeBatchInputData make_decode_batch_input_data(
    const std::vector<int>& positions,
    const std::vector<int>& kv_slots,
    int n_kv_heads,
    int kv_capacity,
    int kv_slot_capacity);

}  // namespace nano_llm_infer
