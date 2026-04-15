#include <iostream>
#include <cstdio>
#include "ggml.h"

int main() {
    std::cout << "=== Testing ggml_mul_mat for attention computation ===\n\n";

    // After permute: Q=[seq_len, head_dim, n_heads], K=[n_attn, head_dim, n_kv_heads]

    // Prefill case
    std::cout << "--- Prefill case ---\n";
    ggml_tensor q_prefill, k_prefill;
    q_prefill.ne[0] = 21;  // seq_len
    q_prefill.ne[1] = 64;  // head_dim
    q_prefill.ne[2] = 32;  // n_heads
    q_prefill.ne[3] = 1;

    k_prefill.ne[0] = 21;  // n_attn (same as seq_len in prefill)
    k_prefill.ne[1] = 64;  // head_dim
    k_prefill.ne[2] = 4;   // n_kv_heads
    k_prefill.ne[3] = 1;

    bool c1 = (k_prefill.ne[0] == q_prefill.ne[0]);
    bool c2 = (q_prefill.ne[2] % k_prefill.ne[2] == 0);
    bool c3 = (q_prefill.ne[3] % k_prefill.ne[3] == 0);

    std::cout << "K: [" << k_prefill.ne[0] << ", " << k_prefill.ne[1] << ", " << k_prefill.ne[2] << ", " << k_prefill.ne[3] << "]\n";
    std::cout << "Q: [" << q_prefill.ne[0] << ", " << q_prefill.ne[1] << ", " << q_prefill.ne[2] << ", " << q_prefill.ne[3] << "]\n";
    std::cout << "ggml_can_mul_mat(K, Q):\n";
    std::cout << "  k.ne[0] == q.ne[0]: " << k_prefill.ne[0] << " == " << q_prefill.ne[0] << " = " << c1 << "\n";
    std::cout << "  q.ne[2] % k.ne[2]: " << q_prefill.ne[2] << " % " << k_prefill.ne[2] << " = " << (q_prefill.ne[2] % k_prefill.ne[2]) << " (check: " << c2 << ")\n";
    std::cout << "  q.ne[3] % k.ne[3]: " << q_prefill.ne[3] << " % " << k_prefill.ne[3] << " = " << (q_prefill.ne[3] % k_prefill.ne[3]) << " (check: " << c3 << ")\n";
    std::cout << "Result: " << (c1 && c2 && c3 ? "PASS" : "FAIL") << "\n\n";

    // Decode case
    std::cout << "--- Decode case (pos=21) ---\n";
    ggml_tensor q_decode, k_decode;
    q_decode.ne[0] = 1;   // seq_len (single token)
    q_decode.ne[1] = 64;  // head_dim
    q_decode.ne[2] = 32;  // n_heads
    q_decode.ne[3] = 1;

    k_decode.ne[0] = 22;  // n_attn (cache has 22 positions)
    k_decode.ne[1] = 64;  // head_dim
    k_decode.ne[2] = 4;   // n_kv_heads
    k_decode.ne[3] = 1;

    c1 = (k_decode.ne[0] == q_decode.ne[0]);
    c2 = (q_decode.ne[2] % k_decode.ne[2] == 0);
    c3 = (q_decode.ne[3] % k_decode.ne[3] == 0);

    std::cout << "K: [" << k_decode.ne[0] << ", " << k_decode.ne[1] << ", " << k_decode.ne[2] << ", " << k_decode.ne[3] << "]\n";
    std::cout << "Q: [" << q_decode.ne[0] << ", " << q_decode.ne[1] << ", " << q_decode.ne[2] << ", " << q_decode.ne[3] << "]\n";
    std::cout << "ggml_can_mul_mat(K, Q):\n";
    std::cout << "  k.ne[0] == q.ne[0]: " << k_decode.ne[0] << " == " << q_decode.ne[0] << " = " << c1 << " <-- FAIL\n";
    std::cout << "  q.ne[2] % k.ne[2]: " << q_decode.ne[2] << " % " << k_decode.ne[2] << " = " << (q_decode.ne[2] % k_decode.ne[2]) << " (check: " << c2 << ")\n";
    std::cout << "  q.ne[3] % k.ne[3]: " << q_decode.ne[3] << " % " << k_decode.ne[3] << " = " << (q_decode.ne[3] % k_decode.ne[3]) << " (check: " << c3 << ")\n";
    std::cout << "Result: " << (c1 && c2 && c3 ? "PASS" : "FAIL") << "\n\n";

    std::cout << "PROBLEM: k.ne[0] (n_attn=22) != q.ne[0] (seq_len=1)\n";
    std::cout << "This is the fundamental issue - inner dimensions don't match!\n";

    return 0;
}
