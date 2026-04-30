#include <string>

#include "generation.h"

#include "test_util.h"

using namespace nano_llm_infer;

int main() {
    std::string response = "hello<|assistant|>";
    EXPECT_TRUE(trim_stop_sequence(&response));
    EXPECT_EQ(response, std::string("hello"));

    response = "hello";
    EXPECT_TRUE(!trim_stop_sequence(&response));
    EXPECT_EQ(response, std::string("hello"));

    std::string streamed = "hello<|assistant|>";
    const size_t before = std::string("hello").size();
    EXPECT_TRUE(trim_stop_sequence(&streamed));
    const size_t emitted = streamed.size() - before;
    EXPECT_EQ(emitted, static_cast<size_t>(0));

    return test_util::finalize();
}
