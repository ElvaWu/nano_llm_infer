#include <iostream>
#include "gguf_loader.h"

int main() {
    nano_llm_infer::GGUFLoader loader;
    loader.load_model("model_file/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");

    auto vocab = loader.get_metadata<std::vector<std::string>>("tokenizer.ggml.tokens");
    auto scores = loader.get_metadata<std::vector<float>>("tokenizer.ggml.scores");

    std::cerr << "Comparing token scores:\n";
    std::cerr << "Token 15043 (▁Hello): score=" << scores[15043] << " len=8\n";
    std::cerr << "Token 379 ( H): score=" << scores[379] << " len=4\n";
    std::cerr << "Token 295 (el): score=" << scores[295] << " len=2\n";
    std::cerr << "Token 417 (lo): score=" << scores[417] << " len=2\n";

    // Calculate Viterbi scores
    float score_full = scores[15043] + 0.001f * 8;
    float score_split = scores[379] + 0.001f * 4 + scores[295] + 0.001f * 2 + scores[417] + 0.001f * 2;

    std::cerr << "\nViterbi scores:\n";
    std::cerr << "Full token (▁Hello): " << score_full << "\n";
    std::cerr << "Split tokens ( H + el + lo): " << score_split << "\n";
    std::cerr << "Winner: " << (score_full > score_split ? "Full token" : "Split tokens") << "\n";

    return 0;
}
