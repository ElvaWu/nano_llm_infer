#include "kv_cache.h"

#include "test_util.h"

using namespace nano_llm_infer;

int main() {
    Model invalid_model;
    invalid_model.n_heads = 0;
    invalid_model.n_kv_heads = 0;

    EXPECT_THROW(make_kv_cache(invalid_model, 8));

    KVCache kv;
    free_kv_cache(kv);
    EXPECT_TRUE(kv.ctx == nullptr);
    EXPECT_TRUE(kv.layers.empty());
    EXPECT_EQ(kv.n_filled, 0);

    return test_util::finalize();
}
