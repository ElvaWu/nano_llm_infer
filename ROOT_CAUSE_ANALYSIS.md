# Root Cause Analysis: Why ggml_set_rows Doesn't Work

## The Fundamental Issue

After deep analysis of llama.cpp's implementation and GGML source code, I've identified the root cause:

### How ggml_set_rows Actually Works

From `ggml.c` line 3877:
```c
struct ggml_tensor * ggml_set_rows(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,  // destination (persistent cache)
        struct ggml_tensor  * b,  // source (new K/V values)
        struct ggml_tensor  * c) { // row indices

    struct ggml_tensor * result = ggml_view_tensor(ctx, a);

    result->op     = GGML_OP_SET_ROWS;
    result->src[0] = b;
    result->src[1] = c;
    result->src[2] = a;

    return result;
}
```

**Key Points:**
1. Returns a VIEW of the destination tensor `a`
2. The operation is IN-PLACE - it modifies `a`'s underlying memory during graph execution
3. The view is for dependency tracking in the computation graph

### Why Our Implementation Fails

**Problem 1: Index Type**
- llama.cpp uses `GGML_TYPE_I64` for indices (line 1169 in llama-kv-cache.cpp)
- We used `GGML_TYPE_I32`
- This may cause index interpretation issues

**Problem 2: Context Architecture Mismatch**
The critical difference between llama.cpp and our code:

**llama.cpp:**
- Persistent graph structure
- Index tensors created once, reused across forward passes
- Index data populated BEFORE each graph execution via `set_input_k_idxs()`
- Uses `ggml_set_input()` to mark tensors as externally-provided inputs

**Our code:**
- New context created for each forward pass
- Index tensors created in compute context
- Index data populated immediately after tensor creation
- No persistent graph structure

**Problem 3: Memory Accessibility**
When we create index tensors in the compute context and immediately populate them:
```cpp
ggml_tensor* k_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len);
auto* indices_data = static_cast<int32_t*>(k_idxs->data);
for (int64_t i = 0; i < seq_len; ++i) {
    indices_data[i] = pos + static_cast<int>(i);
}
```

The issue: The tensor's `data` pointer may not point to properly allocated memory yet, or the memory may not be accessible during graph execution.

## Why ggml_cpy + ggml_view_3d Also Fails

Our original approach:
```cpp
ggml_tensor* k_slot = ggml_view_3d(ctx, kv->k, ...);
ggml_build_forward_expand(gf, ggml_cpy(ctx, k, k_slot));
```

This SHOULD work because:
- `ggml_cpy` is explicitly designed for copying between tensors
- The view points to the correct location in persistent cache memory
- Both tensors are properly allocated

But it produces gibberish output, suggesting the cache isn't being updated.

## The Real Root Cause

After all attempts, the pattern is clear: **Operations in the compute context cannot reliably modify memory in the persistent cache context.**

This is likely due to:
1. **Backend/buffer isolation**: GGML may isolate memory between contexts
2. **Execution model**: The compute context may use different memory management
3. **Buffer ownership**: The persistent cache buffer may not be writable from compute context operations

## Why llama.cpp Works

llama.cpp's architecture solves this:
1. **Single unified context management**: They use a sophisticated context/buffer system
2. **Backend scheduler**: `ggml_backend_sched` manages cross-context operations
3. **Explicit buffer management**: They use `ggml_backend_tensor_set()` for direct writes
4. **Input tensor system**: `ggml_set_input()` ensures proper memory accessibility

## The Solution

We have three options:

### Option 1: Restructure to Match llama.cpp (Complex)
- Implement persistent graph structure
- Add backend scheduler
- Use input tensor system with pre-execution population
- **Effort**: High, requires major refactoring

### Option 2: Use Direct Memory Operations (Hacky)
- After graph execution, directly memcpy into cache memory
- Bypass GGML's operation system entirely
- **Effort**: Low, but defeats the purpose of using GGML

### Option 3: Accept the Simple Attention (Current Working State)
- Commit 2a05e9b removes KV cache entirely
- Uses simple attention (very slow but works)
- **Effort**: Zero, already implemented

### Option 4: Debug Why ggml_cpy Fails (Recommended)
- The ggml_cpy + view approach SHOULD work
- Add extensive debugging to understand why cache isn't updating
- May reveal a simple bug we're missing
- **Effort**: Medium

## Recommendation

I recommend Option 4: Debug the ggml_cpy approach more thoroughly. The fact that it produces gibberish (not crashes) suggests the graph executes successfully, but something about the cache update isn't working. We should:

1. Add debug output to verify cache memory addresses
2. Check if ggml_cpy is actually executing
3. Verify the view offsets are correct
4. Check if there's a synchronization issue

If Option 4 fails, fall back to Option 3 (simple attention) as the pragmatic choice for a minimal implementation.
