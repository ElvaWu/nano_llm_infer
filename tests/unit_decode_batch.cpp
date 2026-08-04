#include "decode_batch.h"

#include <cmath>
#include <vector>

#include "test_util.h"

using namespace nano_llm_infer;

namespace {

void test_exact_attention_length() {
    const DecodeBatchInputData data =
        make_decode_batch_input_data({32}, {0}, 1, 128, 1);
    EXPECT_EQ(data.attention_length, 33);
    EXPECT_EQ(data.kv_read_rows.size(), static_cast<size_t>(33));
    EXPECT_EQ(data.attention_mask.size(), static_cast<size_t>(33));
}

void test_sparse_kv_rows_and_mask() {
    const DecodeBatchInputData data = make_decode_batch_input_data(
        std::vector<int>{1, 3},
        std::vector<int>{2, 0},
        2,
        8,
        3);

    EXPECT_EQ(data.attention_length, 4);
    EXPECT_EQ(data.positions.size(), static_cast<size_t>(2));
    EXPECT_EQ(data.positions[0], 1);
    EXPECT_EQ(data.positions[1], 3);
    EXPECT_EQ(data.kv_write_rows, (std::vector<int32_t>{34, 35, 6, 7}));
    EXPECT_EQ(
        data.kv_read_rows,
        (std::vector<int32_t>{32, 34, 32, 32, 33, 35, 33, 33, 0, 2, 4, 6, 1, 3, 5, 7}));
    EXPECT_EQ(data.attention_mask.size(), static_cast<size_t>(8));
    EXPECT_TRUE(!std::isinf(ggml_fp16_to_fp32(data.attention_mask[0])));
    EXPECT_TRUE(!std::isinf(ggml_fp16_to_fp32(data.attention_mask[1])));
    EXPECT_TRUE(std::isinf(ggml_fp16_to_fp32(data.attention_mask[2])));
    EXPECT_TRUE(!std::isinf(ggml_fp16_to_fp32(data.attention_mask[7])));
}

void test_capacity_validation() {
    EXPECT_THROW(make_decode_batch_input_data({8}, {0}, 1, 8, 1));
    EXPECT_THROW(make_decode_batch_input_data({7}, {1}, 1, 8, 1));
    EXPECT_THROW(make_decode_batch_input_data({0}, {0}, 0, 8, 1));
    EXPECT_THROW(make_decode_batch_input_data({}, {}, 1, 8, 1));
    EXPECT_THROW(make_decode_batch_input_data({0}, {}, 1, 8, 1));
}

}  // namespace

int main() {
    test_exact_attention_length();
    test_sparse_kv_rows_and_mask();
    test_capacity_validation();
    return test_util::finalize();
}
