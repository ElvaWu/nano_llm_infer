#include <iostream>
#include "gguf_loader.h"

int main() {
    nano_llm_infer::GGUFLoader loader;
    loader.load_model("model_file/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");

    auto vocab = loader.get_metadata<std::vector<std::string>>("tokenizer.ggml.tokens");
    auto scores = loader.get_metadata<std::vector<float>>("tokenizer.ggml.scores");

    // Check tokens around numbers
    std::cerr << "Checking number-related tokens:\n";
    std::cerr << "Token 29906 (2): '" << vocab[29906] << "' score=" << scores[29906] << "\n";
    std::cerr << "Token 29974 (+): '" << vocab[29974] << "' score=" << scores[29974] << "\n";

    // Search for space+2 token
    for (size_t i = 0; i < vocab.size(); i++) {
        if (vocab[i] == "▁2" || vocab[i] == " 2") {
            std::cerr << "Found space+2 token: " << i << " '" << vocab[i] << "' score=" << scores[i] << "\n";
        }
    }

    return 0;
}
