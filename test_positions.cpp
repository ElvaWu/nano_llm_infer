#include <iostream>
#include <vector>

int main() {
    // Simulate what happens in decode loop
    std::vector<int> all_tokens = {1, 2, 3, 4, 5}; // 5 tokens initially
    
    for (int step = 0; step < 3; ++step) {
        int next_token = 100 + step;
        all_tokens.push_back(next_token);
        
        std::cout << "Step " << step << ": all_tokens.size() = " << all_tokens.size() << "\n";
        std::cout << "  Logit index: " << (static_cast<int64_t>(all_tokens.size()) - 1) << "\n";
        std::cout << "  Tokens: ";
        for (auto t : all_tokens) std::cout << t << " ";
        std::cout << "\n\n";
    }
    
    return 0;
}
