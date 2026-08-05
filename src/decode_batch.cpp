#include "decode_batch.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace nano_llm_infer {

namespace {

int32_t checked_kv_row(int kv_slot, int position, int kv_head, int n_kv_heads, int kv_capacity) {
    const int64_t row =
        (static_cast<int64_t>(kv_slot) * kv_capacity + position) * n_kv_heads + kv_head;
    if (row > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error("flattened KV row index exceeds int32 range");
    }
    return static_cast<int32_t>(row);
}

}  // namespace

DecodeBatchInputData make_decode_batch_input_data(
    const std::vector<int>& positions,
    const std::vector<int>& kv_slots,
    int n_kv_heads,
    int kv_capacity,
    int kv_slot_capacity) {
    if (positions.empty()) {
        throw std::invalid_argument("decode batch inputs require at least one position");
    }
    if (kv_slots.size() != positions.size()) {
        throw std::invalid_argument("decode batch KV slot count does not match positions size");
    }
    if (n_kv_heads < 1 || kv_capacity < 1 || kv_slot_capacity < 1) {
        throw std::invalid_argument("decode batch KV dimensions must be positive");
    }
    DecodeBatchInputData result;
    for (size_t batch = 0; batch < positions.size(); ++batch) {
        const int position = positions[batch];
        const int kv_slot = kv_slots[batch];
        if (position < 0 || position >= kv_capacity) {
            throw std::invalid_argument("decode batch position exceeds KV capacity");
        }
        if (kv_slot < 0 || kv_slot >= kv_slot_capacity) {
            throw std::invalid_argument("decode batch KV slot exceeds arena capacity");
        }
        result.attention_length = std::max(result.attention_length, position + 1);
    }

    result.positions.reserve(positions.size());
    result.kv_write_rows.reserve(positions.size() * static_cast<size_t>(n_kv_heads));
    result.kv_read_rows.reserve(
        positions.size() * static_cast<size_t>(n_kv_heads) * result.attention_length);
    result.attention_mask.reserve(
        positions.size() * static_cast<size_t>(result.attention_length));

    for (size_t batch = 0; batch < positions.size(); ++batch) {
        const int position = positions[batch];
        const int kv_slot = kv_slots[batch];
        result.positions.push_back(static_cast<int32_t>(position));
        for (int kv_head = 0; kv_head < n_kv_heads; ++kv_head) {
            result.kv_write_rows.push_back(
                checked_kv_row(kv_slot, position, kv_head, n_kv_heads, kv_capacity));
        }
        for (int kv_head = 0; kv_head < n_kv_heads; ++kv_head) {
            for (int attention_pos = 0; attention_pos < result.attention_length; ++attention_pos) {
                const int readable_pos = attention_pos <= position ? attention_pos : 0;
                result.kv_read_rows.push_back(
                    checked_kv_row(kv_slot, readable_pos, kv_head, n_kv_heads, kv_capacity));
            }
        }
        for (int attention_pos = 0; attention_pos < result.attention_length; ++attention_pos) {
            result.attention_mask.push_back(ggml_fp32_to_fp16(
                attention_pos <= position ? 0.0f : -INFINITY));
        }
    }

    return result;
}

}  // namespace nano_llm_infer
