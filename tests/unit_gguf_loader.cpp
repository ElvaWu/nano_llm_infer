#include <string>

#include "gguf_loader.h"

#include "test_util.h"

using namespace nano_llm_infer;

int main() {
    GGUFLoader loader;

    EXPECT_THROW(loader.load_model("/tmp/this_file_should_not_exist.gguf"));
    EXPECT_TRUE(!loader.has_metadata("tokenizer.ggml.tokens"));
    EXPECT_THROW(loader.get_metadata<std::string>("missing.key"));

    return test_util::finalize();
}
