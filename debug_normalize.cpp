#include <iostream>
#include <string>

const char* kSentencePieceSpace = "\xE2\x96\x81";  // U+2581

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

int main() {
    std::string test = "Hello";
    std::string normalized = normalize_for_spm(test);

    std::cerr << "Input: '" << test << "'\n";
    std::cerr << "Normalized: '" << normalized << "'\n";
    std::cerr << "Normalized length: " << normalized.length() << "\n";
    std::cerr << "Normalized bytes: ";
    for (unsigned char c : normalized) {
        std::cerr << (int)c << " ";
    }
    std::cerr << "\n";

    return 0;
}
