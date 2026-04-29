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

    return test_util::finalize();
}
