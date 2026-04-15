#include <iostream>
#include <cstdio>
#include "ggml.h"

int main() {
    // Test ggml_can_mul_mat with GQA dimensions

    // Simulate Q: [64, 32, 21] (head_dim, n_heads, seq_len)
    ggml_tensor q_sim;
    q_sim.ne[0] = 64;  // head_dim
    q_sim.ne[1] = 32;  // n_heads
    q_sim.ne[2] = 21;  // seq_len
    q_sim.ne[3] = 1;

    // Simulate K: [64, 4, 21] (head_dim, n_kv_heads, seq_len)
    ggml_tensor k_sim;
    k_sim.ne[0] = 64;  // head_dim
    k_sim.ne[1] = 4;   // n_kv_heads
    k_sim.ne[2] = 21;  // seq_len
    k_sim.ne[3] = 1;

    // Check ggml_can_mul_mat(k, q)
    bool check1 = (k_sim.ne[0] == q_sim.ne[0]);
    bool check2 = (q_sim.ne[2] % k_sim.ne[2] == 0);
    bool check3 = (q_sim.ne[3] % k_sim.ne[3] == 0);

    std::cout << "Q shape: [" << q_sim.ne[0] << ", " << q_sim.ne[1] << ", " << q_sim.ne[2] << ", " << q_sim.ne[3] << "]\n";
    std::cout << "K shape: [" << k_sim.ne[0] << ", " << k_sim.ne[1] << ", " << k_sim.ne[2] << ", " << k_sim.ne[3] << "]\n";
    std::cout << "\nggml_can_mul_mat(k, q) checks:\n";
    std::cout << "  k.ne[0] == q.ne[0]: " << k_sim.ne[0] << " == " << q_sim.ne[0] << " = " << check1 << "\n";
    std::cout << "  q.ne[2] % k.ne[2] == 0: " << q_sim.ne[2] << " % " << k_sim.ne[2] << " = " << (q_sim.ne[2] % k_sim.ne[2]) << " (check: " << check2 << ")\n";
    std::cout << "  q.ne[3] % k.ne[3] == 0: " << q_sim.ne[3] << " % " << k_sim.ne[3] << " = " << (q_sim.ne[3] % k_sim.ne[3]) << " (check: " << check3 << ")\n";
    std::cout << "\nResult: " << (check1 && check2 && check3 ? "PASS" : "FAIL") << "\n";

    // Now check decode case: Q: [64, 32, 1], K: [64, 4, 22]
    std::cout << "\n--- Decode step (pos=21) ---\n";
    q_sim.ne[2] = 1;   // single token
    k_sim.ne[2] = 22;  // cache now has 22 positions

    check1 = (k_sim.ne[0] == q_sim.ne[0]);
    check2 = (q_sim.ne[2] % k_sim.ne[2] == 0);
    check3 = (q_sim.ne[3] % k_sim.ne[3] == 0);

    std::cout << "Q shape: [" << q_sim.ne[0] << ", " << q_sim.ne[1] << ", " << q_sim.ne[2] << ", " << q_sim.ne[3] << "]\n";
    std::cout << "K shape: [" << k_sim.ne[0] << ", " << k_sim.ne[1] << ", " << k_sim.ne[2] << ", " << k_sim.ne[3] << "]\n";
    std::cout << "\nggml_can_mul_mat(k, q) checks:\n";
    std::cout << "  k.ne[0] == q.ne[0]: " << k_sim.ne[0] << " == " << q_sim.ne[0] << " = " << check1 << "\n";
    std::cout << "  q.ne[2] % k.ne[2] == 0: " << q_sim.ne[2] << " % " << k_sim.ne[2] << " = " << (q_sim.ne[2] % k_sim.ne[2]) << " (check: " << check2 << ")\n";
    std::cout << "  q.ne[3] % k.ne[3] == 0: " << q_sim.ne[3] << " % " << k_sim.ne[3] << " = " << (q_sim.ne[3] % k_sim.ne[3]) << " (check: " << check3 << ")\n";
    std::cout << "\nResult: " << (check1 && check2 && check3 ? "PASS" : "FAIL") << "\n";
    std::cout << "\nFAILURE: q.ne[2]=1 is not divisible by k.ne[2]=22!\n";

    return 0;
}
