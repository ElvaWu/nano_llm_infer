#include <iostream>
#include "gguf_loader.h"

int main() {
    nano_llm_infer::GGUFLoader loader;
    loader.load_model("model_file/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");

    auto vocab = loader.get_metadata<std::vector<std::string>>("tokenizer.ggml.tokens");

    std::cerr << "Vocab size: " << vocab.size() << "\n\n";

    // Search for "Hello" related tokens
    std::cerr << "Searching for 'Hello' tokens:\n";
    for (size_t i = 0; i < vocab.size(); i++) {
        if (vocab[i].find("Hello") != std::string::npos ||
            vocab[i].find("hello") != std::string::npos ||
            vocab[i].find("HELLO") != std::string::npos) {
            std::cerr << i << ": '" << vocab[i] << "'\n";
        }
    }

    // Check specific tokens
    std::cerr << "\nChecking specific tokens:\n";
    std::cerr << "Token 229: '" << vocab[229] << "'\n";
    std::cerr << "Token 153: '" << vocab[153] << "'\n";
    std::cerr << "Token 132: '" << vocab[132] << "'\n";
    std::cerr << "Token 75: '" << vocab[75] << "'\n";
    std::cerr << "Token 104: '" << vocab[104] << "'\n";

    // Check first 50 tokens
    std::cerr << "\nFirst 50 tokens:\n";
    for (int i = 0; i < 50 && i < vocab.size(); i++) {
        std::cerr << i << ": '" << vocab[i] << "'\n";
    }

    return 0;
}
