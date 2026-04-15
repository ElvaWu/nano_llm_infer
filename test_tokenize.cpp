#include <iostream>
#include "tokenizer.h"
#include "gguf_loader.h"

int main() {
    nano_llm_infer::GGUFLoader loader;
    auto model = loader.load_model("model_file/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");
    
    nano_llm_infer::Tokenizer tokenizer;
    tokenizer.load_from_gguf(loader);
    
    std::string test = "<|assistant|>";
    auto tokens = tokenizer.encode(test);
    
    std::cout << "String: \"" << test << "\"\n";
    std::cout << "Tokens: ";
    for (auto t : tokens) {
        std::cout << t << " ";
    }
    std::cout << "\n";
    
    std::cout << "Decoded back: \"" << tokenizer.decode(tokens) << "\"\n";
    
    // Also test the individual tokens
    std::cout << "\nToken 29966: \"" << tokenizer.decode({29966}) << "\"\n";
    std::cout << "Token 29989: \"" << tokenizer.decode({29989}) << "\"\n";
    std::cout << "Token 465: \"" << tokenizer.decode({465}) << "\"\n";
    std::cout << "Token 391: \"" << tokenizer.decode({391}) << "\"\n";
    std::cout << "Token 424: \"" << tokenizer.decode({424}) << "\"\n";
    std::cout << "Token 29958: \"" << tokenizer.decode({29958}) << "\"\n";
    
    return 0;
}
