#include <iostream>
#include "gguf_loader.h"
#include "tokenizer.h"

int main() {
    nano_llm_infer::GGUFLoader loader;
    loader.load_model("model_file/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");

    auto vocab = loader.get_metadata<std::vector<std::string>>("tokenizer.ggml.tokens");

    std::cerr << "Token 15043: '" << vocab[15043] << "'\n";
    std::cerr << "Token 10994: '" << vocab[10994] << "'\n";

    // Check if the space marker is correct
    const char* space_marker = "\xE2\x96\x81";  // U+2581
    std::cerr << "Space marker bytes: ";
    for (int i = 0; i < 3; i++) {
        std::cerr << (int)(unsigned char)space_marker[i] << " ";
    }
    std::cerr << "\n";

    // Build the expected token
    std::string expected = std::string(space_marker) + "Hello";
    std::cerr << "Expected token: '" << expected << "'\n";
    std::cerr << "Expected length: " << expected.length() << "\n";
    std::cerr << "Vocab[15043] length: " << vocab[15043].length() << "\n";
    std::cerr << "Match: " << (expected == vocab[15043]) << "\n";

    return 0;
}
