#include <iostream>
#include "gguf_loader.h"
#include "tokenizer.h"

int main() {
    nano_llm_infer::GGUFLoader loader;
    loader.load_model("model_file/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");

    auto vocab = loader.get_metadata<std::vector<std::string>>("tokenizer.ggml.tokens");
    auto scores = loader.get_metadata<std::vector<float>>("tokenizer.ggml.scores");

    std::cerr << "Checking token scores:\n";
    std::cerr << "Token 15043 (▁Hello): score=" << scores[15043] << "\n";
    std::cerr << "Token 10994 (Hello): score=" << scores[10994] << "\n";
    std::cerr << "Token 229 (<0xE2>): score=" << scores[229] << "\n";
    std::cerr << "Token 75 (<0x48>): score=" << scores[75] << "\n";

    // Check a few more common tokens
    std::cerr << "\nSample token scores:\n";
    for (int i = 0; i < 10; i++) {
        std::cerr << "Token " << i << " ('" << vocab[i] << "'): score=" << scores[i] << "\n";
    }

    return 0;
}
