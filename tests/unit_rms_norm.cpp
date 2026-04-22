#include "transformer.h"

#include "test_util.h"

using namespace nano_llm_infer;

int main() {
    const ggml_init_params params = {
        /* mem_size   */ 1ull * 1024 * 1024,
        /* mem_buffer */ nullptr,
        /* no_alloc   */ false,
    };

    ggml_context* ctx = ggml_init(params);
    EXPECT_TRUE(ctx != nullptr);

    if (ctx != nullptr) {
        ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);
        ggml_tensor* w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);

        ggml_tensor* y = rms_norm(ctx, x, w, 1e-5f);
        EXPECT_TRUE(y != nullptr);
        if (y != nullptr) {
            EXPECT_EQ(y->ne[0], 4);
            EXPECT_EQ(y->ne[1], 2);
            EXPECT_EQ(ggml_n_dims(y), 2);
        }

        ggml_free(ctx);
    }

    return test_util::finalize();
}
