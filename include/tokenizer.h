#include <vector>
#include <string>
#include "gguf_loader.h"

namespace nano_llm_infer{

enum class TokenizerMode {
    LEGACY = 0,
    STRICT_MODEL = 1,
};

class Tokenizer {
public:
    bool load_from_gguf(const GGUFLoader& loader);

    void set_mode(TokenizerMode mode) { mode_ = mode; }
    TokenizerMode mode() const { return mode_; }

    std::vector<int> encode(const std::string& text);
    std::string decode(int token_id) const;
    std::string decode(const std::vector<int>& tokens) const;

    int bos() const { return bos_token_; }
    int eos() const { return eos_token_; }

private:
    int longest_token_len_ = 1;
    int unk_token_ = 0;
    int bos_token_ = 1;
    int eos_token_ = 2;
    std::string model_type_ = "unknown";
    TokenizerMode mode_ = TokenizerMode::LEGACY;

    std::vector<std::string> vocab_;
    std::vector<float> scores_;
    std::vector<float> token_scores_;
    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_map<std::string, int> merge_rank_;
    bool has_merges_ = false;
    std::array<int, 256> byte_token_ids_ {};
};

} // namespace nano_llm_infer