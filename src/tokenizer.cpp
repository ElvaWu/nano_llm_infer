#include "tokenizer.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <unordered_map>

namespace nano_llm_infer{

namespace {

constexpr const char* kSentencePieceSpace = "\xE2\x96\x81";  // U+2581

int metadata_int_with_default(const GGUFLoader& loader, const std::string& key, int fallback) {
    if (!loader.has_metadata(key)) {
        return fallback;
    }
    try {
        return static_cast<int>(loader.get_metadata<uint32_t>(key));
    } catch (...) {
    }
    try {
        return static_cast<int>(loader.get_metadata<uint64_t>(key));
    } catch (...) {
    }
    try {
        return static_cast<int>(loader.get_metadata<int32_t>(key));
    } catch (...) {
    }
    try {
        return static_cast<int>(loader.get_metadata<int64_t>(key));
    } catch (...) {
    }
    return fallback;
}

bool parse_byte_token(const std::string& token, int* out_byte) {
    if (token.size() != 6) {
        return false;
    }
    if (token[0] != '<' || token[1] != '0' || token[2] != 'x' || token[5] != '>') {
        return false;
    }

    const auto hex = [](char c) -> int {
        if (c >= '0' && c <= '9') {
            return c - '0';
        }
        if (c >= 'A' && c <= 'F') {
            return 10 + (c - 'A');
        }
        if (c >= 'a' && c <= 'f') {
            return 10 + (c - 'a');
        }
        return -1;
    };

    const int hi = hex(token[3]);
    const int lo = hex(token[4]);
    if (hi < 0 || lo < 0) {
        return false;
    }

    *out_byte = (hi << 4) | lo;
    return true;
}

std::string normalize_for_spm(const std::string& text) {
    std::string out;
    out.reserve(text.size() + 8);

    out += kSentencePieceSpace;
    for (unsigned char c : text) {
        if (c == ' ') {
            out += kSentencePieceSpace;
        } else {
            out.push_back(static_cast<char>(c));
        }
    }
    return out;
}


bool is_sentencepiece_like(const std::string& model_type) {
    if (model_type.empty() || model_type == "unknown") {
        return true;
    }
    const std::string lowered = [&]() {
        std::string out = model_type;
        std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return out;
    }();
    if (lowered.find("gpt2") != std::string::npos || lowered.find("bpe") != std::string::npos) {
        return false;
    }
    return true;
}

void replace_all(std::string& s, const std::string& needle, const std::string& repl) {
    if (needle.empty()) {
        return;
    }
    size_t pos = 0;
    while ((pos = s.find(needle, pos)) != std::string::npos) {
        s.replace(pos, needle.size(), repl);
        pos += repl.size();
    }
}

std::vector<std::string> split_utf8(const std::string& s) {
    std::vector<std::string> out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        size_t len = 1;
        if ((c & 0x80u) == 0x00u) {
            len = 1;
        } else if ((c & 0xE0u) == 0xC0u) {
            len = 2;
        } else if ((c & 0xF0u) == 0xE0u) {
            len = 3;
        } else if ((c & 0xF8u) == 0xF0u) {
            len = 4;
        }
        if (i + len > s.size()) {
            len = 1;
        }
        out.push_back(s.substr(i, len));
        i += len;
    }
    return out;
}

inline std::string merge_key(const std::string& a, const std::string& b) {
    return a + '\t' + b;
}

} // namespace


bool Tokenizer::load_from_gguf(const GGUFLoader& loader) {
    vocab_.clear();
    scores_.clear();
    token_to_id_.clear();
    token_scores_.clear();
    merge_rank_.clear();
    has_merges_ = false;
    byte_token_ids_.fill(-1);
    longest_token_len_ = 1;

    if (loader.has_metadata("tokenizer.ggml.tokens")) {
        try {
            vocab_ = loader.get_metadata<std::vector<std::string>>("tokenizer.ggml.tokens");
        } catch (...) {
            vocab_.clear();
        }
    }

    if (loader.has_metadata("tokenizer.ggml.scores")) {
        try {
            scores_ = loader.get_metadata<std::vector<float>>("tokenizer.ggml.scores");
        } catch (...) {
            scores_.clear();
        }
    }

    if (loader.has_metadata("tokenizer.ggml.model")) {
        try {
            model_type_ = loader.get_metadata<std::string>("tokenizer.ggml.model");
        } catch (...) {
            model_type_ = "unknown";
        }
    }

    if (loader.has_metadata("tokenizer.ggml.merges")) {
        try {
            const auto merges = loader.get_metadata<std::vector<std::string>>("tokenizer.ggml.merges");
            for (size_t i = 0; i < merges.size(); ++i) {
                const std::string& line = merges[i];
                const size_t sp = line.find(' ');
                if (sp == std::string::npos) {
                    continue;
                }
                const std::string left = line.substr(0, sp);
                const std::string right = line.substr(sp + 1);
                merge_rank_.emplace(merge_key(left, right), static_cast<int>(i));
            }
            has_merges_ = !merge_rank_.empty();
        } catch (...) {
            merge_rank_.clear();
            has_merges_ = false;
        }
    }

    if (vocab_.empty()) {
        vocab_.reserve(256);
        for (int i = 0; i < 256; ++i) {
            vocab_.push_back(std::string(1, static_cast<char>(i)));
        }
    }

    token_scores_.assign(vocab_.size(), 0.0f);
    if (scores_.size() == vocab_.size()) {
        token_scores_ = scores_;
    }

    for (size_t i = 0; i < vocab_.size(); ++i) {
        token_to_id_[vocab_[i]] = static_cast<int>(i);
        longest_token_len_ = std::max(longest_token_len_, static_cast<int>(vocab_[i].size()));

        int b = -1;
        if (parse_byte_token(vocab_[i], &b)) {
            byte_token_ids_[static_cast<size_t>(b)] = static_cast<int>(i);
        }
    }

    bos_token_ = metadata_int_with_default(loader, "tokenizer.ggml.bos_token_id", 1);
    eos_token_ = metadata_int_with_default(loader, "tokenizer.ggml.eos_token_id", 2);
    unk_token_ = metadata_int_with_default(loader, "tokenizer.ggml.unknown_token_id", 0);

    auto unk_it = token_to_id_.find("<unk>");
    if (unk_it != token_to_id_.end()) {
        unk_token_ = unk_it->second;
    }

    if (bos_token_ < 0 || bos_token_ >= static_cast<int>(vocab_.size())) {
        bos_token_ = 1;
    }
    if (eos_token_ < 0 || eos_token_ >= static_cast<int>(vocab_.size())) {
        eos_token_ = 2;
    }
    if (unk_token_ < 0 || unk_token_ >= static_cast<int>(vocab_.size())) {
        unk_token_ = 0;
    }

    return true;
}

std::vector<int> Tokenizer::encode(const std::string& text) {
    std::vector<int> tokens;
    
    if (vocab_.empty()) {
        return tokens;
    }

    const std::string normalized =
        is_sentencepiece_like(model_type_) ? normalize_for_spm(text) : text;
    const bool strict_mode = mode_ == TokenizerMode::STRICT_MODEL;

    if (has_merges_) {
        std::vector<std::string> symbols = split_utf8(normalized);
        while (symbols.size() > 1) {
            int best_rank = std::numeric_limits<int>::max();
            int best_pos = -1;
            for (size_t i = 0; i + 1 < symbols.size(); ++i) {
                auto it = merge_rank_.find(merge_key(symbols[i], symbols[i + 1]));
                if (it != merge_rank_.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_pos = static_cast<int>(i);
                }
            }
            if (best_pos < 0) {
                break;
            }
            symbols[static_cast<size_t>(best_pos)] += symbols[static_cast<size_t>(best_pos + 1)];
            symbols.erase(symbols.begin() + best_pos + 1);
        }

        for (const std::string& piece : symbols) {
            auto it = token_to_id_.find(piece);
            if (it != token_to_id_.end()) {
                tokens.push_back(it->second);
                continue;
            }

            bool emitted_bytes = true;
            for (unsigned char byte : piece) {
                const int byte_id = byte_token_ids_[static_cast<size_t>(byte)];
                if (byte_id >= 0) {
                    tokens.push_back(byte_id);
                    continue;
                }
                emitted_bytes = false;
                break;
            }

            if (emitted_bytes) {
                continue;
            }

            if (strict_mode && unk_token_ >= 0) {
                tokens.push_back(unk_token_);
                continue;
            }

            for (unsigned char byte : piece) {
                const int byte_id = byte_token_ids_[static_cast<size_t>(byte)];
                if (byte_id >= 0) {
                    tokens.push_back(byte_id);

                } else {

                    tokens.push_back(static_cast<int>(static_cast<size_t>(byte) % vocab_.size()));
                }
            }
        }

        return tokens;
    }

    const size_t n = normalized.size();
    std::vector<float> dp(n + 1, -std::numeric_limits<float>::infinity());
    std::vector<int> prev_id(n + 1, -1);
    std::vector<size_t> prev_pos(n + 1, 0);
    std::vector<uint8_t> prev_kind(n + 1, 0);  // 0=token, 1=byte fallback, 2=unknown fallback
    dp[0] = 0.0f;

    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(dp[i])) {
            continue;
        }

        const size_t max_len = std::min(static_cast<size_t>(longest_token_len_), n - i);
        for (size_t len = max_len; len >= 1; --len) {
            const std::string piece = normalized.substr(i, len);
            auto it = token_to_id_.find(piece);
            if (it != token_to_id_.end()) {
                const int id = it->second;
                const float score =
                    dp[i] + token_scores_[static_cast<size_t>(id)] + 0.001f * static_cast<float>(len);
                const size_t next = i + len;
                if (score > dp[next]) {
                    dp[next] = score;
                    prev_id[next] = id;
                    prev_pos[next] = i;
                    prev_kind[next] = 0;
                }
            }
            if (len == 1) {
                break;
            }
        }

        // Byte fallback via <0xXX> tokens.
        const unsigned char byte = static_cast<unsigned char>(normalized[i]);
        const int byte_id = byte_token_ids_[static_cast<size_t>(byte)];
        int fallback_id = 0;
        if (byte_id >= 0) {
            fallback_id = byte_id;
        } else if (strict_mode && unk_token_ >= 0) {
            fallback_id = unk_token_;
        } else {
            fallback_id = static_cast<int>(static_cast<size_t>(byte) % vocab_.size());
        }
        const size_t next = i + 1;
        const float fallback_score = strict_mode ? (dp[i] - 6.0f) : (dp[i] - 12.0f);
        if (fallback_score > dp[next]) {
            dp[next] = fallback_score;
            prev_id[next] = fallback_id;
            prev_pos[next] = i;
            prev_kind[next] = (byte_id >= 0 ? 1 : 2);
        }
    }

    if (!std::isfinite(dp[n])) {
        return tokens;
    }

    std::vector<int> rev;
    int byte_fallbacks = 0;
    int unknown_fallbacks = 0;
    for (size_t cur = n; cur > 0;) {
        const int id = prev_id[cur];
        if (id < 0) {
            break;
        }
        if (prev_kind[cur] == 1) {
            ++byte_fallbacks;
        } else if (prev_kind[cur] == 2) {
            ++unknown_fallbacks;
        }
        rev.push_back(id);
        cur = prev_pos[cur];
    }
    for (auto it = rev.rbegin(); it != rev.rend(); ++it) {
        tokens.push_back(*it);
    }

    return tokens;
}

std::string Tokenizer::decode(int token_id) const {
    if (token_id < 0 || token_id >= static_cast<int>(vocab_.size())) {
        return "";
    }

    std::string piece = vocab_[static_cast<size_t>(token_id)];

    int byte_value = -1;
    if (parse_byte_token(piece, &byte_value)) {
        return std::string(1, static_cast<char>(byte_value));
    }

    if (piece == "<s>" || piece == "</s>" || piece == "<bos>" || piece == "<eos>" ||
        piece == "<unk>" || piece == "<pad>") {
        return "";
    }

    if (is_sentencepiece_like(model_type_)) {
        replace_all(piece, kSentencePieceSpace, " ");
    }
    return piece;
}

std::string Tokenizer::decode(const std::vector<int>& tokens) const {
    std::string out;
    for (int token : tokens) {
        out += decode(token);
    }
    return out;
}

}