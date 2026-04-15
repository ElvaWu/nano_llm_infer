#include <iostream>
#include "gguf_loader.h"
#include "tokenizer.h"

int main() {
    nano_llm_infer::GGUFLoader loader;
    loader.load_model("model_file/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");

    nano_llm_infer::Tokenizer tokenizer;
    tokenizer.load_from_gguf(loader);

    std::cerr << "Vocab size: " << tokenizer.vocab_size << "\n";
    std::cerr << "BOS: " << tokenizer.bos() << "\n";
    std::cerr << "EOS: " << tokenizer.eos() << "\n";

    // Check first 20 tokens
    std::cerr << "\nFirst 20 tokens:\n";
    for (int i = 0; i < 20 && i < tokenizer.vocab_size; i++) {
        std::cerr << i << ": '" << tokenizer.decode(i) << "'\n";
    }

    // Test encoding
    std::string test = "Hello";
    auto tokens = tokenizer.encode(test);
    std::cerr << "\nEncoding 'Hello': ";
    for (int t : tokens) {
        std::cerr << t << " ";
    }
    std::cerr << "\n";

    return 0;
}
