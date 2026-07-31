#include "tokenizer.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace nano_llm_infer {

namespace {

constexpr const char* kSentencePieceSpace = "\xE2\x96\x81";
constexpr const char* kQwenDefaultSystemMessage =
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
constexpr int32_t kTokenTypeNormal = 1;
constexpr int32_t kTokenTypeControl = 3;
constexpr int32_t kTokenTypeUserDefined = 4;

// TinyLlama chat prompts rely on these markers appearing literally in the token stream.
constexpr std::array<const char*, 3> kControlMarkers = {
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
};

struct Utf8Char {
    size_t offset = 0;
    size_t length = 1;
    uint32_t codepoint = 0;
};

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

bool metadata_bool_with_default(const GGUFLoader& loader, const std::string& key, bool fallback) {
    if (!loader.has_metadata(key)) {
        return fallback;
    }
    try {
        return loader.get_metadata<bool>(key);
    } catch (...) {
        return fallback;
    }
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

std::string normalize_for_spm(const std::string& text, bool add_dummy_prefix) {
    std::string out;
    out.reserve(text.size() + 8);

    if (add_dummy_prefix) {
        out += kSentencePieceSpace;
    }
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

    std::string lowered = model_type;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return lowered.find("gpt2") == std::string::npos &&
        lowered.find("bpe") == std::string::npos;
}

void replace_all(std::string& text, const std::string& needle, const std::string& replacement) {
    if (needle.empty()) {
        return;
    }

    size_t pos = 0;
    while ((pos = text.find(needle, pos)) != std::string::npos) {
        text.replace(pos, needle.size(), replacement);
        pos += replacement.size();
    }
}

std::string utf8_from_codepoint(uint32_t codepoint) {
    std::string out;
    if (codepoint <= 0x7Fu) {
        out.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7FFu) {
        out.push_back(static_cast<char>(0xC0u | (codepoint >> 6)));
        out.push_back(static_cast<char>(0x80u | (codepoint & 0x3Fu)));
    } else if (codepoint <= 0xFFFFu) {
        out.push_back(static_cast<char>(0xE0u | (codepoint >> 12)));
        out.push_back(static_cast<char>(0x80u | ((codepoint >> 6) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | (codepoint & 0x3Fu)));
    } else {
        out.push_back(static_cast<char>(0xF0u | (codepoint >> 18)));
        out.push_back(static_cast<char>(0x80u | ((codepoint >> 12) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | ((codepoint >> 6) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | (codepoint & 0x3Fu)));
    }
    return out;
}

std::vector<Utf8Char> decode_utf8(const std::string& text) {
    std::vector<Utf8Char> chars;
    chars.reserve(text.size());

    size_t offset = 0;
    while (offset < text.size()) {
        const unsigned char lead = static_cast<unsigned char>(text[offset]);
        size_t length = 1;
        uint32_t codepoint = lead;

        if ((lead & 0xE0u) == 0xC0u) {
            length = 2;
            codepoint = lead & 0x1Fu;
        } else if ((lead & 0xF0u) == 0xE0u) {
            length = 3;
            codepoint = lead & 0x0Fu;
        } else if ((lead & 0xF8u) == 0xF0u) {
            length = 4;
            codepoint = lead & 0x07u;
        }

        bool valid = offset + length <= text.size();
        for (size_t i = 1; valid && i < length; ++i) {
            const unsigned char continuation = static_cast<unsigned char>(text[offset + i]);
            if ((continuation & 0xC0u) != 0x80u) {
                valid = false;
                break;
            }
            codepoint = (codepoint << 6) | (continuation & 0x3Fu);
        }
        if (!valid) {
            length = 1;
            codepoint = lead;
        }

        chars.push_back({offset, length, codepoint});
        offset += length;
    }
    return chars;
}

std::vector<std::string> split_utf8(const std::string& text) {
    std::vector<std::string> out;
    const std::vector<Utf8Char> chars = decode_utf8(text);
    out.reserve(chars.size());
    for (const Utf8Char& character : chars) {
        out.push_back(text.substr(character.offset, character.length));
    }
    return out;
}

bool codepoint_in_range(uint32_t codepoint, uint32_t first, uint32_t last) {
    return codepoint >= first && codepoint <= last;
}

bool is_unicode_number(uint32_t codepoint) {
    return codepoint_in_range(codepoint, '0', '9') ||
        codepoint_in_range(codepoint, 0x0660, 0x0669) ||
        codepoint_in_range(codepoint, 0x06F0, 0x06F9) ||
        codepoint_in_range(codepoint, 0x0966, 0x096F) ||
        codepoint_in_range(codepoint, 0x09E6, 0x09EF) ||
        codepoint_in_range(codepoint, 0x0E50, 0x0E59) ||
        codepoint_in_range(codepoint, 0xFF10, 0xFF19);
}

bool is_unicode_letter(uint32_t codepoint) {
    if (codepoint_in_range(codepoint, 'A', 'Z') ||
        codepoint_in_range(codepoint, 'a', 'z')) {
        return true;
    }
    if (codepoint_in_range(codepoint, 0x00C0, 0x02AF) &&
        codepoint != 0x00D7 && codepoint != 0x00F7) {
        return true;
    }
    if (codepoint_in_range(codepoint, 0x0370, 0x052F) ||
        codepoint_in_range(codepoint, 0x0531, 0x058F) ||
        codepoint_in_range(codepoint, 0x0590, 0x05FF) ||
        codepoint_in_range(codepoint, 0x0600, 0x06FF) ||
        codepoint_in_range(codepoint, 0x0700, 0x08FF) ||
        codepoint_in_range(codepoint, 0x0900, 0x1FFF)) {
        return true;
    }
    return codepoint_in_range(codepoint, 0x3041, 0x3096) ||
        codepoint_in_range(codepoint, 0x30A1, 0x30FA) ||
        codepoint_in_range(codepoint, 0x3105, 0x312F) ||
        codepoint_in_range(codepoint, 0x3131, 0x318E) ||
        codepoint_in_range(codepoint, 0x3400, 0x4DBF) ||
        codepoint_in_range(codepoint, 0x4E00, 0x9FFF) ||
        codepoint_in_range(codepoint, 0xAC00, 0xD7A3) ||
        codepoint_in_range(codepoint, 0xF900, 0xFAFF) ||
        codepoint_in_range(codepoint, 0xFF21, 0xFF3A) ||
        codepoint_in_range(codepoint, 0xFF41, 0xFF5A) ||
        codepoint_in_range(codepoint, 0x20000, 0x2FA1F);
}

bool is_unicode_whitespace(uint32_t codepoint) {
    return codepoint == 0x09 || codepoint == 0x0A || codepoint == 0x0B ||
        codepoint == 0x0C || codepoint == 0x0D || codepoint == 0x20 ||
        codepoint == 0x85 || codepoint == 0xA0 || codepoint == 0x1680 ||
        codepoint_in_range(codepoint, 0x2000, 0x200A) ||
        codepoint == 0x2028 || codepoint == 0x2029 || codepoint == 0x202F ||
        codepoint == 0x205F || codepoint == 0x3000;
}

bool is_newline(uint32_t codepoint) {
    return codepoint == '\r' || codepoint == '\n';
}

char ascii_lower(char value) {
    if (value >= 'A' && value <= 'Z') {
        return static_cast<char>(value - 'A' + 'a');
    }
    return value;
}

size_t contraction_length(const std::string& text, size_t offset) {
    constexpr std::array<const char*, 7> contractions = {
        "'re", "'ve", "'ll", "'s", "'t", "'m", "'d",
    };
    for (const char* contraction : contractions) {
        const size_t length = std::char_traits<char>::length(contraction);
        if (offset + length > text.size()) {
            continue;
        }

        bool matches = true;
        for (size_t i = 0; i < length; ++i) {
            if (ascii_lower(text[offset + i]) != contraction[i]) {
                matches = false;
                break;
            }
        }
        if (matches) {
            return length;
        }
    }
    return 0;
}

// Mirrors Qwen2's regex-style pre-tokenizer, including one-token digits and
// optional punctuation/space prefixes before letter runs.
std::vector<std::string> split_qwen_utf8(const std::string& text) {
    const std::vector<Utf8Char> chars = decode_utf8(text);
    std::vector<std::string> pieces;
    pieces.reserve(chars.size());

    auto append_range = [&](size_t first, size_t last) {
        const size_t byte_start = chars[first].offset;
        const size_t byte_end = last == chars.size()
            ? text.size()
            : chars[last].offset;
        pieces.push_back(text.substr(byte_start, byte_end - byte_start));
    };

    size_t index = 0;
    while (index < chars.size()) {
        const Utf8Char& current = chars[index];
        if (current.codepoint == '\'' && current.length == 1) {
            const size_t length = contraction_length(text, current.offset);
            if (length > 0) {
                size_t end = index;
                size_t bytes = 0;
                while (end < chars.size() && bytes < length) {
                    bytes += chars[end].length;
                    ++end;
                }
                if (bytes == length) {
                    append_range(index, end);
                    index = end;
                    continue;
                }
            }
        }

        const bool current_letter = is_unicode_letter(current.codepoint);
        const bool current_number = is_unicode_number(current.codepoint);
        if (!is_newline(current.codepoint) && !current_letter && !current_number &&
            index + 1 < chars.size() && is_unicode_letter(chars[index + 1].codepoint)) {
            size_t end = index + 2;
            while (end < chars.size() && is_unicode_letter(chars[end].codepoint)) {
                ++end;
            }
            append_range(index, end);
            index = end;
            continue;
        }

        if (current_letter) {
            size_t end = index + 1;
            while (end < chars.size() && is_unicode_letter(chars[end].codepoint)) {
                ++end;
            }
            append_range(index, end);
            index = end;
            continue;
        }

        if (current_number) {
            append_range(index, index + 1);
            ++index;
            continue;
        }

        size_t punctuation_start = index;
        if (current.codepoint == ' ' && index + 1 < chars.size()) {
            const uint32_t next = chars[index + 1].codepoint;
            if (!is_unicode_whitespace(next) && !is_unicode_letter(next) &&
                !is_unicode_number(next)) {
                ++punctuation_start;
            }
        }
        if (!is_unicode_whitespace(chars[punctuation_start].codepoint) &&
            !is_unicode_letter(chars[punctuation_start].codepoint) &&
            !is_unicode_number(chars[punctuation_start].codepoint)) {
            size_t end = punctuation_start + 1;
            while (end < chars.size() && !is_unicode_whitespace(chars[end].codepoint) &&
                   !is_unicode_letter(chars[end].codepoint) &&
                   !is_unicode_number(chars[end].codepoint)) {
                ++end;
            }
            while (end < chars.size() && is_newline(chars[end].codepoint)) {
                ++end;
            }
            append_range(index, end);
            index = end;
            continue;
        }

        if (is_unicode_whitespace(current.codepoint)) {
            size_t end = index + 1;
            while (end < chars.size() && is_unicode_whitespace(chars[end].codepoint)) {
                ++end;
            }
            append_range(index, end);
            index = end;
            continue;
        }

        append_range(index, index + 1);
        ++index;
    }

    return pieces;
}

std::string merge_key(const std::string& left, const std::string& right) {
    return left + '\t' + right;
}

bool is_control_marker(const std::string& token) {
    return std::any_of(kControlMarkers.begin(), kControlMarkers.end(), [&](const char* marker) {
        return token == marker;
    });
}

// GPT-2 BPE maps every byte to a printable Unicode symbol before applying merges.
void initialize_byte_maps(
    std::array<std::string, 256>* encoder,
    std::unordered_map<std::string, uint8_t>* decoder) {
    std::array<bool, 256> mapped{};
    auto map_direct = [&](int first, int last) {
        for (int value = first; value <= last; ++value) {
            (*encoder)[static_cast<size_t>(value)] = utf8_from_codepoint(static_cast<uint32_t>(value));
            mapped[static_cast<size_t>(value)] = true;
        }
    };

    map_direct(0x21, 0x7E);
    map_direct(0xA1, 0xAC);
    map_direct(0xAE, 0xFF);

    uint32_t extra = 0;
    for (int value = 0; value < 256; ++value) {
        if (!mapped[static_cast<size_t>(value)]) {
            (*encoder)[static_cast<size_t>(value)] = utf8_from_codepoint(256 + extra);
            ++extra;
        }
        (*decoder)[(*encoder)[static_cast<size_t>(value)]] = static_cast<uint8_t>(value);
    }
}

}  // namespace

bool Tokenizer::load_from_gguf(const GGUFLoader& loader) {
    vocab_.clear();
    scores_.clear();
    token_scores_.clear();
    token_types_.clear();
    special_tokens_.clear();
    token_to_id_.clear();
    merge_rank_.clear();
    byte_decoder_.clear();
    has_merges_ = false;
    byte_token_ids_.fill(-1);
    bpe_byte_token_ids_.fill(-1);
    longest_token_len_ = 1;
    model_type_ = "unknown";
    tokenizer_pre_.clear();
    chat_template_.clear();

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

    if (loader.has_metadata("tokenizer.ggml.token_type")) {
        try {
            token_types_ = loader.get_metadata<std::vector<int32_t>>("tokenizer.ggml.token_type");
        } catch (...) {
            token_types_.clear();
        }
    }

    if (loader.has_metadata("tokenizer.ggml.model")) {
        try {
            model_type_ = loader.get_metadata<std::string>("tokenizer.ggml.model");
        } catch (...) {
            model_type_ = "unknown";
        }
    }

    if (loader.has_metadata("tokenizer.ggml.pre")) {
        try {
            tokenizer_pre_ = loader.get_metadata<std::string>("tokenizer.ggml.pre");
        } catch (...) {
            tokenizer_pre_.clear();
        }
    }

    if (loader.has_metadata("tokenizer.chat_template")) {
        try {
            chat_template_ = loader.get_metadata<std::string>("tokenizer.chat_template");
        } catch (...) {
            chat_template_.clear();
        }
    }

    if (loader.has_metadata("tokenizer.ggml.merges")) {
        try {
            const auto merges = loader.get_metadata<std::vector<std::string>>("tokenizer.ggml.merges");
            for (size_t i = 0; i < merges.size(); ++i) {
                const std::string& line = merges[i];
                const size_t space = line.find(' ');
                if (space == std::string::npos) {
                    continue;
                }
                const std::string left = line.substr(0, space);
                const std::string right = line.substr(space + 1);
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
    if (token_types_.size() != vocab_.size()) {
        token_types_.assign(vocab_.size(), kTokenTypeNormal);
    }

    initialize_byte_maps(&byte_encoder_, &byte_decoder_);

    for (size_t i = 0; i < vocab_.size(); ++i) {
        token_to_id_[vocab_[i]] = static_cast<int>(i);
        longest_token_len_ = std::max(longest_token_len_, static_cast<int>(vocab_[i].size()));

        int byte_value = -1;
        if (parse_byte_token(vocab_[i], &byte_value)) {
            byte_token_ids_[static_cast<size_t>(byte_value)] = static_cast<int>(i);
        }

        // Control and user-defined tokens must be split out before normal
        // SentencePiece or BPE processing so ChatML markers remain atomic.
        const int32_t token_type = token_types_[i];
        if (token_type == kTokenTypeControl || token_type == kTokenTypeUserDefined) {
            special_tokens_.push_back(vocab_[i]);
        }
    }
    for (size_t byte = 0; byte < byte_encoder_.size(); ++byte) {
        const auto it = token_to_id_.find(byte_encoder_[byte]);
        if (it != token_to_id_.end()) {
            bpe_byte_token_ids_[byte] = it->second;
        }
    }

    if (is_sentencepiece_like(model_type_)) {
        special_tokens_.insert(special_tokens_.end(), kControlMarkers.begin(), kControlMarkers.end());
    }
    std::sort(special_tokens_.begin(), special_tokens_.end(), [](const std::string& left, const std::string& right) {
        if (left.size() != right.size()) {
            return left.size() > right.size();
        }
        return left < right;
    });
    special_tokens_.erase(
        std::unique(special_tokens_.begin(), special_tokens_.end()),
        special_tokens_.end());

    bos_token_ = metadata_int_with_default(loader, "tokenizer.ggml.bos_token_id", 1);
    eos_token_ = metadata_int_with_default(loader, "tokenizer.ggml.eos_token_id", 2);
    unk_token_ = metadata_int_with_default(loader, "tokenizer.ggml.unknown_token_id", 0);
    // Older SentencePiece GGUFs omit add_bos_token; defaulting them to true
    // preserves TinyLlama's existing prompt behavior while Qwen sets it explicitly.
    add_bos_ = metadata_bool_with_default(
        loader,
        "tokenizer.ggml.add_bos_token",
        is_sentencepiece_like(model_type_));
    add_eos_ = metadata_bool_with_default(loader, "tokenizer.ggml.add_eos_token", false);

    const auto unk_it = token_to_id_.find("<unk>");
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

    vocab_size_ = static_cast<int>(vocab_.size());
    return true;
}

std::vector<int> Tokenizer::encode(const std::string& text) {
    std::vector<int> tokens;
    if (vocab_.empty()) {
        return tokens;
    }

    const bool sentencepiece = is_sentencepiece_like(model_type_);
    const bool strict_mode = mode_ == TokenizerMode::STRICT_MODEL;

    auto encode_sentencepiece_segment = [&](const std::string& segment, bool add_dummy_prefix) {
        if (segment.empty()) {
            return;
        }

        const auto exact = token_to_id_.find(segment);
        if (exact != token_to_id_.end()) {
            tokens.push_back(exact->second);
            return;
        }

        const bool use_dummy_prefix =
            add_dummy_prefix && !is_control_marker(segment) &&
            segment[0] != '\n' && segment[0] != '\r';
        const std::string normalized = normalize_for_spm(segment, use_dummy_prefix);
        const size_t length = normalized.size();
        std::vector<float> scores(length + 1, -std::numeric_limits<float>::infinity());
        std::vector<int> previous_ids(length + 1, -1);
        std::vector<size_t> previous_positions(length + 1, 0);
        scores[0] = 0.0f;

        for (size_t position = 0; position < length; ++position) {
            if (!std::isfinite(scores[position])) {
                continue;
            }

            const size_t max_length =
                std::min(static_cast<size_t>(longest_token_len_), length - position);
            for (size_t piece_length = max_length; piece_length >= 1; --piece_length) {
                const std::string piece = normalized.substr(position, piece_length);
                const auto piece_it = token_to_id_.find(piece);
                if (piece_it != token_to_id_.end()) {
                    const int id = piece_it->second;
                    const float score = scores[position] +
                        token_scores_[static_cast<size_t>(id)] +
                        0.001f * static_cast<float>(piece_length);
                    const size_t next = position + piece_length;
                    if (score > scores[next]) {
                        scores[next] = score;
                        previous_ids[next] = id;
                        previous_positions[next] = position;
                    }
                }
                if (piece_length == 1) {
                    break;
                }
            }

            const unsigned char byte = static_cast<unsigned char>(normalized[position]);
            const int byte_id = byte_token_ids_[static_cast<size_t>(byte)];
            int fallback_id = 0;
            if (byte_id >= 0) {
                fallback_id = byte_id;
            } else if (strict_mode && unk_token_ >= 0) {
                fallback_id = unk_token_;
            } else {
                fallback_id = static_cast<int>(static_cast<size_t>(byte) % vocab_.size());
            }

            const size_t next = position + 1;
            const float fallback_score = scores[position] - 100000.0f;
            if (fallback_score > scores[next]) {
                scores[next] = fallback_score;
                previous_ids[next] = fallback_id;
                previous_positions[next] = position;
            }
        }

        if (!std::isfinite(scores[length])) {
            return;
        }

        std::vector<int> reversed;
        for (size_t position = length; position > 0;) {
            const int id = previous_ids[position];
            if (id < 0) {
                break;
            }
            reversed.push_back(id);
            position = previous_positions[position];
        }
        for (auto it = reversed.rbegin(); it != reversed.rend(); ++it) {
            tokens.push_back(*it);
        }
    };

    auto encode_bpe_segment = [&](const std::string& segment) {
        // Pretokenize UTF-8 text first, then byte-encode each chunk before
        // applying the globally ranked GPT-2 merge table.
        for (const std::string& word : split_qwen_utf8(segment)) {
            std::vector<std::string> symbols;
            symbols.reserve(word.size());
            for (unsigned char byte : word) {
                symbols.push_back(byte_encoder_[static_cast<size_t>(byte)]);
            }

            if (has_merges_) {
                while (symbols.size() > 1) {
                    int best_rank = std::numeric_limits<int>::max();
                    int best_position = -1;
                    for (size_t i = 0; i + 1 < symbols.size(); ++i) {
                        const auto merge_it =
                            merge_rank_.find(merge_key(symbols[i], symbols[i + 1]));
                        if (merge_it != merge_rank_.end() && merge_it->second < best_rank) {
                            best_rank = merge_it->second;
                            best_position = static_cast<int>(i);
                        }
                    }
                    if (best_position < 0) {
                        break;
                    }

                    symbols[static_cast<size_t>(best_position)] +=
                        symbols[static_cast<size_t>(best_position + 1)];
                    symbols.erase(symbols.begin() + best_position + 1);
                }
            }

            for (const std::string& piece : symbols) {
                const auto piece_it = token_to_id_.find(piece);
                if (piece_it != token_to_id_.end()) {
                    tokens.push_back(piece_it->second);
                    continue;
                }

                std::vector<int> byte_ids;
                bool has_all_bytes = true;
                for (const std::string& symbol : split_utf8(piece)) {
                    const auto byte_it = byte_decoder_.find(symbol);
                    if (byte_it == byte_decoder_.end()) {
                        has_all_bytes = false;
                        break;
                    }
                    const int byte_id = bpe_byte_token_ids_[byte_it->second];
                    if (byte_id < 0) {
                        has_all_bytes = false;
                        break;
                    }
                    byte_ids.push_back(byte_id);
                }
                if (has_all_bytes) {
                    tokens.insert(tokens.end(), byte_ids.begin(), byte_ids.end());
                } else if (strict_mode && unk_token_ >= 0) {
                    tokens.push_back(unk_token_);
                } else {
                    for (unsigned char byte : piece) {
                        tokens.push_back(static_cast<int>(
                            static_cast<size_t>(byte) % vocab_.size()));
                    }
                }
            }
        }
    };

    auto encode_normal_segment = [&](const std::string& segment, bool add_dummy_prefix) {
        if (sentencepiece) {
            encode_sentencepiece_segment(segment, add_dummy_prefix);
        } else {
            encode_bpe_segment(segment);
        }
    };

    size_t position = 0;
    bool add_dummy_prefix = true;
    while (position <= text.size()) {
        size_t best_position = std::string::npos;
        size_t best_length = 0;
        const std::string* best_special = nullptr;
        for (const std::string& special : special_tokens_) {
            if (special.empty()) {
                continue;
            }
            const size_t found = text.find(special, position);
            if (found != std::string::npos &&
                (best_position == std::string::npos || found < best_position ||
                 (found == best_position && special.size() > best_length))) {
                best_position = found;
                best_length = special.size();
                best_special = &special;
            }
        }

        if (best_position == std::string::npos) {
            encode_normal_segment(text.substr(position), add_dummy_prefix);
            break;
        }
        if (best_position > position) {
            encode_normal_segment(
                text.substr(position, best_position - position),
                add_dummy_prefix);
            add_dummy_prefix = false;
        }

        const auto special_it = token_to_id_.find(*best_special);
        if (special_it != token_to_id_.end()) {
            tokens.push_back(special_it->second);
        } else {
            encode_sentencepiece_segment(*best_special, false);
        }
        position = best_position + best_length;
        add_dummy_prefix = false;
    }

    return tokens;
}

std::string Tokenizer::decode(int token_id) const {
    if (token_id < 0 || token_id >= vocab_size_) {
        return "";
    }

    const size_t index = static_cast<size_t>(token_id);
    std::string piece = vocab_[index];
    const int32_t token_type = token_types_[index];
    if (token_type == kTokenTypeControl || token_type == kTokenTypeUserDefined) {
        return "";
    }

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
        return piece;
    }

    std::string decoded;
    for (const std::string& symbol : split_utf8(piece)) {
        const auto byte_it = byte_decoder_.find(symbol);
        if (byte_it != byte_decoder_.end()) {
            decoded.push_back(static_cast<char>(byte_it->second));
        } else {
            decoded += symbol;
        }
    }
    return decoded;
}

std::string Tokenizer::decode(const std::vector<int>& tokens) const {
    std::string out;
    for (int token : tokens) {
        out += decode(token);
    }
    return out;
}

std::string Tokenizer::apply_chat_template(
    const std::vector<ChatMessage>& messages,
    bool add_generation_prompt) const {
    if (messages.empty()) {
        return "";
    }

    const bool has_template = !chat_template_.empty();
    const bool is_tinyllama_template =
        has_template &&
        chat_template_.find("<|user|>") != std::string::npos &&
        chat_template_.find("<|assistant|>") != std::string::npos &&
        chat_template_.find("eos_token") != std::string::npos &&
        chat_template_.find("add_generation_prompt") != std::string::npos;
    const bool is_qwen_template =
        has_template &&
        chat_template_.find("<|im_start|>") != std::string::npos &&
        chat_template_.find("<|im_end|>") != std::string::npos &&
        chat_template_.find("add_generation_prompt") != std::string::npos;

    if (has_template && !is_tinyllama_template && !is_qwen_template) {
        throw std::runtime_error("Unsupported tokenizer.chat_template format");
    }

    for (const auto& message : messages) {
        if (message.role != "system" && message.role != "user" && message.role != "assistant") {
            throw std::runtime_error("Unsupported chat role: " + message.role);
        }
    }

    if (is_qwen_template) {
        // This is the text-only ChatML branch of Qwen's template. Tool calls and
        // multimodal message variants intentionally remain outside this runtime.
        std::string prompt;
        if (messages.front().role != "system") {
            prompt += "<|im_start|>system\n";
            prompt += kQwenDefaultSystemMessage;
            prompt += "<|im_end|>\n";
        }
        for (const auto& message : messages) {
            prompt += "<|im_start|>" + message.role + "\n" +
                message.content + "<|im_end|>\n";
        }
        if (add_generation_prompt) {
            prompt += "<|im_start|>assistant\n";
        }
        return prompt;
    }

    const std::string eos_text = vocab_[static_cast<size_t>(eos_token_)];
    std::string prompt;
    for (const auto& message : messages) {
        prompt += "<|" + message.role + "|>\n" + message.content + eos_text;
    }
    if (add_generation_prompt) {
        prompt += "<|assistant|>";
    }
    return prompt;
}

}  // namespace nano_llm_infer
