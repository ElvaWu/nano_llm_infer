#include <vector>

#include "tokenizer.h"

#include "test_util.h"

using namespace nano_llm_infer;

int main() {
    GGUFLoader empty_loader;
    Tokenizer tokenizer;

    EXPECT_TRUE(tokenizer.load_from_gguf(empty_loader));
    EXPECT_EQ(tokenizer.bos(), 1);
    EXPECT_EQ(tokenizer.eos(), 2);

    const std::vector<int> ids = tokenizer.encode("Hello");
    EXPECT_TRUE(!ids.empty());

    const std::string decoded = tokenizer.decode(ids);
    EXPECT_TRUE(!decoded.empty());

    EXPECT_EQ(tokenizer.apply_chat_template({}, false), "");

    std::vector<ChatMessage> bad_role = {{"invalid", "hello"}};
    EXPECT_THROW(tokenizer.apply_chat_template(bad_role, false));

    return test_util::finalize();
}
