# llama.cpp KV Cache Implementation Analysis

## Key Finding: The Missing Piece

After examining llama.cpp's source code, I found the critical difference in how they handle KV cache updates:

## llama.cpp's Approach

### 1. Index Tensor Creation (Build Phase)
```cpp
// In llama_kv_cache::build_input_k_idxs() - line 1166
ggml_tensor * k_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
ggml_set_input(k_idxs);  // ← CRITICAL: Mark as input tensor
return k_idxs;
```

### 2. Cache Write Operation (Graph Build Phase)
```cpp
// In llama_kv_cache::cpy_k() - line 1075
ggml_tensor * llama_kv_cache::cpy_k(ggml_context * ctx, ggml_tensor * k_cur,
                                     ggml_tensor * k_idxs, int32_t il, ...) {
    // Get cache tensor from persistent context
    ggml_tensor * k = layers[ikv].k;

    // Reshape k_cur to 2D: [n_embd_gqa, n_tokens]
    k_cur = ggml_view_2d(ctx, k_cur, n_embd_gqa, n_tokens, k_cur->nb[2], 0);

    // Reshape cache to 2D if needed
    if (n_stream > 1) {
        k = ggml_reshape_2d(ctx, k, n_embd_gqa, kv_size*n_stream);
    }

    // Use ggml_set_rows with the index tensor
    return ggml_set_rows(ctx, k, k_cur, k_idxs);
}
```

### 3. Index Population (BEFORE Graph Execution)
```cpp
// In llama_kv_cache::set_input_k_idxs() - line 1192
void llama_kv_cache::set_input_k_idxs(ggml_tensor * dst,
                                       const llama_ubatch * ubatch,
                                       const slot_info & sinfo) const {
    int64_t * data = (int64_t *) dst->data;

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const int64_t offs = sinfo.strm[s]*get_size();

        for (uint32_t i = 0; i < sinfo.size(); ++i) {
            data[s*sinfo.size() + i] = offs + sinfo.idxs[s][i];
        }
    }
}
```

### 4. Graph Execution Flow
```cpp
// In llama-graph.cpp - line 2047
// Build graph:
ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));

// Later, BEFORE graph execution:
inp->set_input(ubatch);  // This calls set_input_k_idxs()

// Then execute:
ggml_graph_compute_with_ctx(ctx, gf, n_threads);
```

## The Critical Difference

### What We Were Doing Wrong:
1. Creating index tensor with `ggml_new_tensor_1d()` in compute context
2. Manually filling `indices_data` pointer
3. Calling `ggml_set_rows()`
4. **Problem**: The index data was in compute context memory, which may not be properly accessible during graph execution

### What llama.cpp Does Right:
1. Creates index tensor and marks it with `ggml_set_input()`
2. Uses `ggml_set_rows()` with this input tensor
3. **BEFORE** graph execution, populates the index tensor data via `set_input_k_idxs()`
4. The input tensor's data is in HOST memory (verified by `ggml_backend_buffer_is_host()`)
5. Graph execution reads the pre-populated indices

## Why This Matters

`ggml_set_input()` tells GGML that this tensor's data will be provided externally before graph execution. This ensures:
- The tensor data is in the correct memory space (host-accessible)
- The data is populated BEFORE the graph runs
- GGML knows to read from this pre-populated data during execution

## Implementation Plan

We need to:
1. Create index tensors and mark them with `ggml_set_input()`
2. Store these tensors so we can populate them later
3. BEFORE calling `ggml_graph_compute_with_ctx()`, populate the index tensors
4. Use `ggml_set_rows()` with these input tensors in the graph

This is a two-phase approach:
- **Phase 1 (Graph Build)**: Create input tensors, build graph with `ggml_set_rows()`
- **Phase 2 (Pre-Execution)**: Populate input tensor data
- **Phase 3 (Execution)**: Run graph, which reads the pre-populated indices
