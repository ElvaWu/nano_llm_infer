#include <iostream>
#include <cstdio>
#include "ggml.h"
#include "transformer.h"
#include "gguf_loader.h"
#include "kv_cache.h"

using namespace nano_llm_infer;

void print_tensor_shape(const char* name, const ggml_tensor* t) {
    std::cout << name << ": [" << t->ne[0] << ", " << t->ne[1] << ", " << t->ne[2] << ", " << t->ne[3] << "]\n";
}

int main() {
    std::cout << "Loading model...\n";
    GGUFLoader loader;
    if (!loader.load_model("model_file/TinyLlama-1.1B-Chat-v1.0.Q8_0.gguf")) {
        std::cerr << "Failed to load model\n";
        return 1;
    }

    Model model = loader.get_model();
    std::cout << "Model loaded: n_heads=" << model.n_heads << ", n_kv_heads=" << model.n_kv_heads << "\n";
    std::cout << "head_dim=" << (model.n_embd / model.n_heads) << "\n\n";

    // Create a small compute context for testing
    struct ggml_init_params params = {
        .mem_size   = 512 * 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };
    ggml_context* ctx = ggml_init(params);

    // Simulate attention tensors
    const int64_t head_dim = model.n_embd / model.n_heads;
    const int64_t seq_len = 21;
    const int64_t n_heads = model.n_heads;
    const int64_t n_kv_heads = model.n_kv_heads;

    std::cout << "=== Simulating prefill attention ===\n";
    std::cout << "seq_len=" << seq_len << ", n_heads=" << n_heads << ", n_kv_heads=" << n_kv_heads << "\n\n";

    // Create Q, K, V tensors as they would be after RoPE
    ggml_tensor* q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_heads, seq_len);
    ggml_tensor* k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_kv_heads, seq_len);
    ggml_tensor* v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_kv_heads, seq_len);

    std::cout << "After RoPE:\n";
    print_tensor_shape("Q", q);
    print_tensor_shape("K", k);
    print_tensor_shape("V", v);
    std::cout << "\n";

    // Permute for matmul
    q = ggml_cont(ctx, ggml_permute(ctx, q, 2, 0, 1, 3));
    k = ggml_cont(ctx, ggml_permute(ctx, k, 2, 0, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, v, 2, 0, 1, 3));

    std::cout << "After permute:\n";
    print_tensor_shape("Q", q);
    print_tensor_shape("K", k);
    print_tensor_shape("V", v);
    std::cout << "\n";

    // Check if mul_mat(K, Q) is valid
    bool can_mul = (k->ne[0] == q->ne[0]) &&
                   (q->ne[2] % k->ne[2] == 0) &&
                   (q->ne[3] % k->ne[3] == 0);
    std::cout << "ggml_can_mul_mat(K, Q): " << (can_mul ? "YES" : "NO") << "\n";
    std::cout << "  k->ne[0] == q->ne[0]: " << k->ne[0] << " == " << q->ne[0] << " = " << (k->ne[0] == q->ne[0]) << "\n";
    std::cout << "  q->ne[2] % k->ne[2]: " << q->ne[2] << " % " << k->ne[2] << " = " << (q->ne[2] % k->ne[2]) << "\n";
    std::cout << "  q->ne[3] % k->ne[3]: " << q->ne[3] << " % " << k->ne[3] << " = " << (q->ne[3] % k->ne[3]) << "\n\n";

    if (can_mul) {
        ggml_tensor* scores = ggml_mul_mat(ctx, k, q);
        std::cout << "After mul_mat(K, Q):\n";
        print_tensor_shape("scores", scores);
        std::cout << "\n";

        // Check if mul_mat(V, scores) is valid
        can_mul = (v->ne[0] == scores->ne[0]) &&
                  (scores->ne[2] % v->ne[2] == 0) &&
                  (scores->ne[3] % v->ne[3] == 0);
        std::cout << "ggml_can_mul_mat(V, scores): " << (can_mul ? "YES" : "NO") << "\n";
        std::cout << "  v->ne[0] == scores->ne[0]: " << v->ne[0] << " == " << scores->ne[0] << " = " << (v->ne[0] == scores->ne[0]) << "\n";
        std::cout << "  scores->ne[2] % v->ne[2]: " << scores->ne[2] << " % " << v->ne[2] << " = " << (scores->ne[2] % v->ne[2]) << "\n";
        std::cout << "  scores->ne[3] % v->ne[3]: " << scores->ne[3] << " % " << v->ne[3] << " = " << (scores->ne[3] % v->ne[3]) << "\n";
    }

    ggml_free(ctx);
    return 0;
}
