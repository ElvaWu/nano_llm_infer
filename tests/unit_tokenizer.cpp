#include <array>
#include <cstdlib>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "generation.h"
#include "tokenizer.h"

#include "gguf_test_util.h"
#include "test_util.h"

#ifndef NANO_LLM_TEST_QWEN_MODEL_PATH
#define NANO_LLM_TEST_QWEN_MODEL_PATH ""
#endif

using namespace nano_llm_infer;

namespace {

std::string utf8_from_codepoint(uint32_t codepoint) {
    std::string out;
    if (codepoint <= 0x7Fu) {
        out.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7FFu) {
        out.push_back(static_cast<char>(0xC0u | (codepoint >> 6)));
        out.push_back(static_cast<char>(0x80u | (codepoint & 0x3Fu)));
    } else {
        out.push_back(static_cast<char>(0xE0u | (codepoint >> 12)));
        out.push_back(static_cast<char>(0x80u | ((codepoint >> 6) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | (codepoint & 0x3Fu)));
    }
    return out;
}

struct TestBpeVocab {
    std::vector<std::string> tokens;
    std::vector<int32_t> token_types;
    std::vector<std::string> merges;
    std::array<int, 256> byte_ids{};
    std::array<std::string, 256> byte_tokens{};
    std::unordered_map<std::string, int> token_ids;
};

TestBpeVocab make_test_bpe_vocab() {
    TestBpeVocab vocab;
    vocab.byte_ids.fill(-1);

    std::vector<int> bytes;
    std::vector<uint32_t> codepoints;
    std::array<bool, 256> included{};
    auto add_range = [&](int first, int last) {
        for (int value = first; value <= last; ++value) {
            bytes.push_back(value);
            codepoints.push_back(static_cast<uint32_t>(value));
            included[static_cast<size_t>(value)] = true;
        }
    };
    add_range(0x21, 0x7E);
    add_range(0xA1, 0xAC);
    add_range(0xAE, 0xFF);

    uint32_t extra = 0;
    for (int value = 0; value < 256; ++value) {
        if (!included[static_cast<size_t>(value)]) {
            bytes.push_back(value);
            codepoints.push_back(256 + extra);
            ++extra;
        }
    }

    for (size_t i = 0; i < bytes.size(); ++i) {
        const std::string token = utf8_from_codepoint(codepoints[i]);
        const int id = static_cast<int>(vocab.tokens.size());
        vocab.tokens.push_back(token);
        vocab.token_types.push_back(1);
        vocab.byte_ids[static_cast<size_t>(bytes[i])] = id;
        vocab.byte_tokens[static_cast<size_t>(bytes[i])] = token;
        vocab.token_ids[token] = id;
    }
    return vocab;
}

int add_token(TestBpeVocab* vocab, const std::string& token, int32_t type = 1) {
    const auto existing = vocab->token_ids.find(token);
    if (existing != vocab->token_ids.end()) {
        return existing->second;
    }
    const int id = static_cast<int>(vocab->tokens.size());
    vocab->tokens.push_back(token);
    vocab->token_types.push_back(type);
    vocab->token_ids[token] = id;
    return id;
}

int add_merge_chain(TestBpeVocab* vocab, const std::vector<std::string>& symbols) {
    std::string current = symbols.front();
    for (size_t i = 1; i < symbols.size(); ++i) {
        vocab->merges.push_back(current + " " + symbols[i]);
        current += symbols[i];
        add_token(vocab, current);
    }
    return vocab->token_ids.at(current);
}

void append_byte_ids(
    const TestBpeVocab& vocab,
    const std::string& text,
    std::vector<int>* output) {
    for (unsigned char byte : text) {
        output->push_back(vocab.byte_ids[static_cast<size_t>(byte)]);
    }
}

gguf_test_util::SyntheticGGUFSpec make_qwen_tokenizer_spec(
    const TestBpeVocab& vocab,
    int bos_token,
    int eos_token,
    bool add_bos,
    bool add_eos) {
    gguf_test_util::SyntheticGGUFSpec spec;
    spec.architecture = "qwen2";
    spec.tokenizer_model = "gpt2";
    spec.tokenizer_pre = "qwen2";
    spec.chat_template =
        "{% if add_generation_prompt %}<|im_start|><|im_end|>{% endif %}";
    spec.tokens = vocab.tokens;
    spec.token_types = vocab.token_types;
    spec.merges = vocab.merges;
    spec.bos_token_id = bos_token;
    spec.eos_token_id = eos_token;
    spec.set_add_bos = true;
    spec.add_bos = add_bos;
    spec.set_add_eos = true;
    spec.add_eos = add_eos;
    return spec;
}

std::string configured_qwen_model_path() {
    const std::string configured = NANO_LLM_TEST_QWEN_MODEL_PATH;
    if (!configured.empty()) {
        return configured;
    }
    const char* environment = std::getenv("NANO_LLM_TEST_QWEN_MODEL");
    return environment == nullptr ? "" : environment;
}

}  // namespace

int main() {
    GGUFLoader empty_loader;
    Tokenizer tokenizer;

    EXPECT_TRUE(tokenizer.load_from_gguf(empty_loader));
    EXPECT_EQ(tokenizer.bos(), 1);
    EXPECT_EQ(tokenizer.eos(), 2);
    EXPECT_TRUE(tokenizer.add_bos());
    EXPECT_TRUE(!tokenizer.add_eos());

    const std::vector<int> ids = tokenizer.encode("Hello");
    EXPECT_TRUE(!ids.empty());

    const std::string decoded = tokenizer.decode(ids);
    EXPECT_TRUE(!decoded.empty());

    const std::vector<int> sentencepiece_prompt = encode_prompt(&tokenizer, "Hello");
    EXPECT_TRUE(!sentencepiece_prompt.empty());
    EXPECT_EQ(sentencepiece_prompt.front(), tokenizer.bos());

    EXPECT_EQ(tokenizer.apply_chat_template({}, false), "");

    std::vector<ChatMessage> bad_role = {{"invalid", "hello"}};
    EXPECT_THROW(tokenizer.apply_chat_template(bad_role, false));

    TestBpeVocab vocab = make_test_bpe_vocab();
    const int hello_id = add_merge_chain(
        &vocab,
        {
            vocab.byte_tokens[static_cast<size_t>('H')],
            vocab.byte_tokens[static_cast<size_t>('e')],
            vocab.byte_tokens[static_cast<size_t>('l')],
            vocab.byte_tokens[static_cast<size_t>('l')],
            vocab.byte_tokens[static_cast<size_t>('o')],
        });
    const int world_id = add_merge_chain(
        &vocab,
        {
            vocab.byte_tokens[static_cast<size_t>(' ')],
            vocab.byte_tokens[static_cast<size_t>('w')],
            vocab.byte_tokens[static_cast<size_t>('o')],
            vocab.byte_tokens[static_cast<size_t>('r')],
            vocab.byte_tokens[static_cast<size_t>('l')],
            vocab.byte_tokens[static_cast<size_t>('d')],
        });
    const int endoftext_id = add_token(&vocab, "<|endoftext|>", 3);
    const int im_start_id = add_token(&vocab, "<|im_start|>", 3);
    const int im_end_id = add_token(&vocab, "<|im_end|>", 3);

    {
        const auto spec =
            make_qwen_tokenizer_spec(vocab, endoftext_id, im_end_id, false, true);
        const auto file = gguf_test_util::write_synthetic_gguf("qwen_tokenizer", spec);
        GGUFLoader loader;
        Model model = loader.load_model(file.path);
        Tokenizer qwen;
        EXPECT_TRUE(qwen.load_from_gguf(loader));
        EXPECT_TRUE(!qwen.add_bos());
        EXPECT_TRUE(qwen.add_eos());

        EXPECT_EQ(qwen.encode("Hello"), std::vector<int>({hello_id}));
        EXPECT_EQ(
            qwen.encode("Hello, world!"),
            std::vector<int>({
                hello_id,
                vocab.byte_ids[static_cast<size_t>(',')],
                world_id,
                vocab.byte_ids[static_cast<size_t>('!')],
            }));
        EXPECT_EQ(
            qwen.encode("1234567"),
            std::vector<int>({
                vocab.byte_ids[static_cast<size_t>('1')],
                vocab.byte_ids[static_cast<size_t>('2')],
                vocab.byte_ids[static_cast<size_t>('3')],
                vocab.byte_ids[static_cast<size_t>('4')],
                vocab.byte_ids[static_cast<size_t>('5')],
                vocab.byte_ids[static_cast<size_t>('6')],
                vocab.byte_ids[static_cast<size_t>('7')],
            }));
        EXPECT_EQ(
            qwen.decode(qwen.encode("Hello, world!")),
            std::string("Hello, world!"));

        std::vector<int> special_sequence = {im_start_id};
        append_byte_ids(vocab, "user\n", &special_sequence);
        special_sequence.push_back(hello_id);
        special_sequence.push_back(im_end_id);
        EXPECT_EQ(
            qwen.encode("<|im_start|>user\nHello<|im_end|>"),
            special_sequence);

        const std::string default_prompt = qwen.apply_chat_template(
            {{"user", "Hello"}},
            true);
        EXPECT_EQ(
            default_prompt,
            std::string(
                "<|im_start|>system\n"
                "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
                "<|im_end|>\n"
                "<|im_start|>user\nHello<|im_end|>\n"
                "<|im_start|>assistant\n"));

        const std::string explicit_prompt = qwen.apply_chat_template(
            {{"system", "S"}, {"user", "U"}},
            true);
        EXPECT_EQ(
            explicit_prompt,
            std::string(
                "<|im_start|>system\nS<|im_end|>\n"
                "<|im_start|>user\nU<|im_end|>\n"
                "<|im_start|>assistant\n"));
        EXPECT_TRUE(explicit_prompt.find("You are Qwen") == std::string::npos);

        std::vector<int> explicit_tokens = {im_start_id};
        append_byte_ids(vocab, "system\nS", &explicit_tokens);
        explicit_tokens.push_back(im_end_id);
        append_byte_ids(vocab, "\n", &explicit_tokens);
        explicit_tokens.push_back(im_start_id);
        append_byte_ids(vocab, "user\nU", &explicit_tokens);
        explicit_tokens.push_back(im_end_id);
        append_byte_ids(vocab, "\n", &explicit_tokens);
        explicit_tokens.push_back(im_start_id);
        append_byte_ids(vocab, "assistant\n", &explicit_tokens);
        EXPECT_EQ(qwen.encode(explicit_prompt), explicit_tokens);

        EXPECT_EQ(
            encode_prompt(&qwen, "Hello"),
            std::vector<int>({hello_id, im_end_id}));
        gguf_test_util::free_model(&model);
    }

    {
        const auto spec =
            make_qwen_tokenizer_spec(vocab, endoftext_id, im_end_id, true, false);
        const auto file = gguf_test_util::write_synthetic_gguf("qwen_bos", spec);
        GGUFLoader loader;
        Model model = loader.load_model(file.path);
        Tokenizer qwen;
        EXPECT_TRUE(qwen.load_from_gguf(loader));
        EXPECT_TRUE(qwen.add_bos());
        EXPECT_TRUE(!qwen.add_eos());
        EXPECT_EQ(
            encode_prompt(&qwen, "Hello"),
            std::vector<int>({endoftext_id, hello_id}));
        gguf_test_util::free_model(&model);
    }

    const std::string qwen_model_path = configured_qwen_model_path();
    if (qwen_model_path.empty()) {
        std::cerr << "[SKIP] unit_tokenizer Qwen model checks: "
                  << "NANO_LLM_TEST_QWEN_MODEL is empty\n";
    } else {
        GGUFLoader loader;
        Model model = loader.load_model(qwen_model_path);
        Tokenizer qwen;
        EXPECT_TRUE(qwen.load_from_gguf(loader));
        EXPECT_TRUE(!qwen.add_bos());
        EXPECT_TRUE(!qwen.add_eos());
        EXPECT_EQ(qwen.encode("Hello"), std::vector<int>({9707}));
        EXPECT_EQ(
            qwen.encode("Hello, world!"),
            std::vector<int>({9707, 11, 1879, 0}));
        EXPECT_EQ(
            qwen.encode("你好，世界！"),
            std::vector<int>({108386, 3837, 99489, 6313}));
        EXPECT_EQ(
            qwen.encode("1234567"),
            std::vector<int>({16, 17, 18, 19, 20, 21, 22}));
        gguf_test_util::free_model(&model);
    }

    return test_util::finalize();
}
