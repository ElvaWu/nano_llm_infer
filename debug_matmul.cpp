#include <iostream>
#include <cstdio>
#include "ggml.h"

int main() {
    // Test ggml_mul_mat(k, q) with GQA dimensions

    // Simulate Q: [64, 32, 1] (head_dim, n_heads, seq_len) - decode step
    ggml_tensor q_sim;
    q_sim.ne[0] = 64;  // head_dim
    q_sim.ne[1] = 32;  // n_heads
    q_sim.ne[2] = 1;   // seq_len (single token)
    q_sim.ne[3] = 1;

    // Simulate K: [64, 4, 22] (head_dim, n_kv_heads, n_attn)
    ggml_tensor k_sim;
    k_sim.ne[0] = 64;  // head_dim
    k_sim.ne[1] = 4;   // n_kv_heads
    k_sim.ne[2] = 22;  // n_attn (cache filled to pos 21)
    k_sim.ne[3] = 1;

    // Check ggml_can_mul_mat(k, q) for ggml_mul_mat(k, q)
    bool check1 = (k_sim.ne[0] == q_sim.ne[0]);
    bool check2 = (q_sim.ne[2] % k_sim.ne[2] == 0);
    bool check3 = (q_sim.ne[3] % k_sim.ne[3] == 0);

    std::cout << "Testing ggml_mul_mat(k, q) for decode step:\n\n";
    std::cout << "K shape: [" << k_sim.ne[0] << ", " << k_sim.ne[1] << ", " << k_sim.ne[2] << ", " << k_sim.ne[3] << "]\n";
    std::cout << "Q shape: [" << q_sim.ne[0] << ", " << q_sim.ne[1] << ", " << q_sim.ne[2] << ", " << q_sim.ne[3] << "]\n";
    std::cout << "\nggml_can_mul_mat(k, q) checks:\n";
    std::cout << "  k.ne[0] == q.ne[0]: " << k_sim.ne[0] << " == " << q_sim.ne[0] << " = " << check1 << "\n";
    std::cout << "  q.ne[2] % k.ne[2] == 0: " << q_sim.ne[2] << " % " << k_sim.ne[2] << " = " << (q_sim.ne[2] % k_sim.ne[2]) << " (check: " << check2 << ")\n";
    std::cout << "  q.ne[3] % k.ne[3] == 0: " << q_sim.ne[3] << " % " << k_sim.ne[3] << " = " << (q_sim.ne[3] % k_sim.ne[3]) << " (check: " << check3 << ")\n";
    std::cout << "\nResult: " << (check1 && check2 && check3 ? "PASS" : "FAIL") << "\n";

    if (!check2) {
        std::cout << "\nFAILURE: q.ne[2]=1 is NOT divisible by k.ne[2]=22!\n";
        std::cout << "ggml_mul_mat does NOT support this case.\n";
    }

    // What about transposing K first?
    std::cout << "\n--- Alternative: Transpose K first ---\n";
    std::cout << "K^T shape would be: [" << k_sim.ne[1] << ", " << k_sim.ne[0] << ", " << k_sim.ne[2] << ", " << k_sim.ne[3] << "]\n";
    std::cout << "                     [4, 64, 22, 1]\n";
    std::cout << "But this doesn't help - we need [22, 4, ...] for matmul with Q=[64, 32, 1]\n";

    // The correct approach: permute to get seq_len in ne[0]
    std::cout << "\n--- Correct approach: Permute tensors ---\n";
    std::cout << "Need Q: [seq_len, head_dim, n_heads] = [1, 64, 32]\n";
    std::cout << "Need K: [n_attn, head_dim, n_kv_heads] = [22, 64, 4]\n";
    std::cout << "Then ggml_mul_mat(K, Q) gives [n_attn, seq_len, n_heads] = [22, 1, 32]\n";

    return 0;
}
