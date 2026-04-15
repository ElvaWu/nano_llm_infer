#include <iostream>
#include "gguf_loader.h"
#include "tokenizer.h"

int main() {
    nano_llm_infer::GGUFLoader loader;
    loader.load_model("model_file/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");

    auto vocab = loader.get_metadata<std::vector<std::string>>("tokenizer.ggml.tokens");

    // Find longest token
    size_t max_len = 0;
    for (const auto& token : vocab) {
        if (token.length() > max_len) {
            max_len = token.length();
        }
    }

    std::cerr << "Longest token in vocab: " << max_len << " bytes\n";
    std::cerr << "Token 15043 (▁Hello) length: " << vocab[15043].length() << " bytes\n";
    std::cerr << "Token 379 ( H) length: " << vocab[379].length() << " bytes\n";
    std::cerr << "Token 295 (el) length: " << vocab[295].length() << " bytes\n";

    return 0;
}
