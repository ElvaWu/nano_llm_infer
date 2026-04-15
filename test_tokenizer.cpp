#include <iostream>
#include "gguf_loader.h"
#include "tokenizer.h"

int main() {
    nano_llm_infer::GGUFLoader loader;
    loader.load_model("model_file/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");

    // Check if tokenizer metadata exists
    std::cerr << "Has tokens: " << loader.has_metadata("tokenizer.ggml.tokens") << "\n";
    std::cerr << "Has scores: " << loader.has_metadata("tokenizer.ggml.scores") << "\n";
    std::cerr << "Has merges: " << loader.has_metadata("tokenizer.ggml.merges") << "\n";
    std::cerr << "Has model: " << loader.has_metadata("tokenizer.ggml.model") << "\n";

    if (loader.has_metadata("tokenizer.ggml.model")) {
        auto model_type = loader.get_metadata<std::string>("tokenizer.ggml.model");
        std::cerr << "Model type: " << model_type << "\n";
    }

    nano_llm_infer::Tokenizer tokenizer;
    tokenizer.load_from_gguf(loader);

    // Test simple encoding
    std::string test = "Hello";
    auto tokens = tokenizer.encode(test);

    std::cerr << "\nEncoding '" << test << "':\n";
    std::cerr << "Tokens: ";
    for (int t : tokens) {
        std::cerr << t << " ";
    }
    std::cerr << "\nDecoded: ";
    for (int t : tokens) {
        std::cerr << "'" << tokenizer.decode(t) << "' ";
    }
    std::cerr << "\n";

    return 0;
}
