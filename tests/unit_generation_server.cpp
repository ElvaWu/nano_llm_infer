#include <string>

#include "generation_server.h"

#include "test_util.h"

using namespace nano_llm_infer;

int main() {
    const std::string body =
        "{\"model\":\"tiny\",\"stream\":true,\"max_tokens\":3,"
        "\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}";
    ParseResult parsed = parse_chat_completion_request(body);
    EXPECT_TRUE(parsed.ok);
    EXPECT_EQ(parsed.request.model, std::string("tiny"));
    EXPECT_TRUE(parsed.request.stream);
    EXPECT_EQ(parsed.request.generation.options.max_tokens, 3);
    EXPECT_EQ(parsed.request.generation.messages.size(), static_cast<size_t>(1));
    EXPECT_EQ(parsed.request.generation.messages[0].role, std::string("user"));
    EXPECT_EQ(parsed.request.generation.messages[0].content, std::string("Hello"));

    ParseResult no_stream = parse_chat_completion_request(
        "{\"model\":\"tiny\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}");
    EXPECT_TRUE(no_stream.ok);
    EXPECT_TRUE(!no_stream.request.stream);

    ParseResult stream_false = parse_chat_completion_request(
        "{\"model\":\"tiny\",\"stream\":false,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}");
    EXPECT_TRUE(stream_false.ok);
    EXPECT_TRUE(!stream_false.request.stream);

    ParseResult stream_non_bool = parse_chat_completion_request(
        "{\"model\":\"tiny\",\"stream\":\"yes\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}");
    EXPECT_TRUE(!stream_non_bool.ok);

    EXPECT_EQ(json_escape("a\"b\\c\n"), std::string("a\\\"b\\\\c\\n"));

    GenerationEvent token;
    token.text = "Hi";
    const std::string chunk = make_sse_chunk("tiny", token);
    EXPECT_TRUE(chunk.find("data: ") == 0);
    EXPECT_TRUE(chunk.find("\"content\":\"Hi\"") != std::string::npos);
    EXPECT_TRUE(chunk.find("\"finish_reason\":null") != std::string::npos);

    GenerationEvent done;
    done.finish_reason = "stop";
    EXPECT_TRUE(make_sse_chunk("tiny", done).find("\"finish_reason\":\"stop\"") != std::string::npos);
    EXPECT_EQ(make_sse_done(), std::string("data: [DONE]\n\n"));

    return test_util::finalize();
}
