#include <iostream>

// Simulate llama.cpp attention computation

int main() {
    std::cout << "=== llama.cpp attention flow ===\n\n";

    // Initial shapes after RoPE
    int head_dim = 64;
    int n_heads = 32;
    int n_kv_heads = 4;
    int seq_len = 1;  // decode
    int n_attn = 22;  // cache filled

    std::cout << "Initial (after RoPE):\n";
    std::cout << "Q: [" << head_dim << ", " << n_heads << ", " << seq_len << "]\n";
    std::cout << "K: [" << head_dim << ", " << n_kv_heads << ", " << n_attn << "]\n";
    std::cout << "V: [" << head_dim << ", " << n_kv_heads << ", " << n_attn << "]\n\n";

    // After permute(0, 2, 1, 3) - swaps dims 1 and 2
    std::cout << "After permute(0, 2, 1, 3):\n";
    std::cout << "Q: [" << head_dim << ", " << seq_len << ", " << n_heads << "]\n";
    std::cout << "K: [" << head_dim << ", " << n_attn << ", " << n_kv_heads << "]\n";
    std::cout << "V: [" << head_dim << ", " << n_attn << ", " << n_kv_heads << "]\n\n";

    // mul_mat(K, Q)
    // For mul_mat(a, b): result is [a->ne[1], b->ne[1], b->ne[2], b->ne[3]]
    // and requires a->ne[0] == b->ne[0]
    std::cout << "mul_mat(K, Q):\n";
    std::cout << "  K->ne[0] == Q->ne[0]? " << head_dim << " == " << head_dim << " YES\n";
    std::cout << "  Result: [K->ne[1], Q->ne[1], Q->ne[2], Q->ne[3]]\n";
    std::cout << "  Result: [" << n_attn << ", " << seq_len << ", " << n_heads << ", 1]\n";
    std::cout << "  kq: [" << n_attn << ", " << seq_len << ", " << n_heads << "]\n\n";

    // V after transpose (if !v_trans)
    // transpose swaps ne[0] and ne[1]
    std::cout << "V after transpose:\n";
    std::cout << "V: [" << n_attn << ", " << head_dim << ", " << n_kv_heads << "]\n\n";

    // mul_mat(V, kq)
    std::cout << "mul_mat(V, kq):\n";
    std::cout << "  V->ne[0] == kq->ne[0]? " << n_attn << " == " << n_attn << " YES\n";
    std::cout << "  Result: [V->ne[1], kq->ne[1], kq->ne[2], kq->ne[3]]\n";
    std::cout << "  Result: [" << head_dim << ", " << seq_len << ", " << n_heads << ", 1]\n";
    std::cout << "  kqv: [" << head_dim << ", " << seq_len << ", " << n_heads << "]\n\n";

    std::cout << "This works! The key is:\n";
    std::cout << "1. permute(0,2,1,3) swaps middle two dimensions\n";
    std::cout << "2. transpose V to get n_attn in ne[0] to match kq->ne[0]\n";
    std::cout << "3. GQA broadcasting happens automatically in mul_mat\n";

    return 0;
}
