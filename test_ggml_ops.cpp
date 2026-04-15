#include <iostream>
#include <cstdio>
#include "ggml.h"

void print_shape(const char* name, const ggml_tensor* t) {
    printf("%s: [%lld, %lld, %lld, %lld]\n", name, t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
}

int main() {
    struct ggml_init_params params = {
        .mem_size   = 128 * 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };
    ggml_context* ctx = ggml_init(params);

    // Simulate decode step
    int head_dim = 64;
    int n_heads = 32;
    int n_kv_heads = 4;
    int seq_len = 1;
    int n_attn = 22;

    printf("=== Testing actual GGML operations ===\n\n");

    // Create tensors
    ggml_tensor* q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_heads, seq_len);
    ggml_tensor* k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_kv_heads, n_attn);
    ggml_tensor* v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_kv_heads, n_attn);

    printf("Initial:\n");
    print_shape("q", q);
    print_shape("k", k);
    print_shape("v", v);
    printf("\n");

    // Permute
    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    printf("After permute(0, 2, 1, 3):\n");
    print_shape("q", q);
    print_shape("k", k);
    print_shape("v", v);
    printf("\n");

    // mul_mat(k, q)
    printf("Attempting ggml_mul_mat(k, q)...\n");
    ggml_tensor* kq = ggml_mul_mat(ctx, k, q);
    print_shape("kq", kq);
    printf("\n");

    // Transpose v
    v = ggml_cont(ctx, ggml_transpose(ctx, v));
    printf("After transpose and cont:\n");
    print_shape("v", v);
    printf("\n");

    // mul_mat(v, kq)
    printf("Attempting ggml_mul_mat(v, kq)...\n");
    ggml_tensor* kqv = ggml_mul_mat(ctx, v, kq);
    print_shape("kqv", kqv);
    printf("\n");

    // Permute back
    kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));
    printf("After permute back:\n");
    print_shape("kqv", kqv);
    printf("\n");

    printf("SUCCESS! All operations completed.\n");

    ggml_free(ctx);
    return 0;
}
