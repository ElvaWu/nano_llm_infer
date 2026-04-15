# KV Cache Fix Attempts - Detailed Log

## Summary

Attempted to implement llama.cpp-style KV cache using various GGML operations. All approaches failed to produce coherent output. The root cause appears to be a fundamental issue with how GGML handles cache updates across different computation contexts.

## Approaches Tried

### 1. ggml_set_rows with Index Tensors
**Approach:** Use `ggml_set_rows(cache, new_kv, row_indices)` to update specific rows in the cache.

**Implementation:**
- Reshape K/V to 2D: `[head_dim * n_kv_heads, seq_len]`
- Create index tensor with positions to update
- Use `ggml_set_rows` to write to cache
- Reshape back to 3D for attention

**Result:** Complete gibberish output. The operation returns a new tensor rather than modifying the persistent cache.

**Issue:** `ggml_set_rows` returns a view/new tensor in the compute context, not a modification of the persistent cache tensor.

### 2. ggml_set_2d with Byte Offsets
**Approach:** Use `ggml_set_2d(cache, new_kv, stride, offset)` to write at specific positions.

**Implementation:**
- Reshape cache and K/V to 2D
- Calculate byte offset: `pos * kv_dim * sizeof(float)`
- Use `ggml_set_2d` to write at offset
- Read from updated cache for attention

**Result:** Worse gibberish than original.

**Issue:** Same as `ggml_set_rows` - returns a new tensor, doesn't modify persistent cache.

### 3. Force Dependency with Dummy Operations
**Approach:** Create explicit dependencies by using write operation results in dummy computations.

**Implementation:**
```cpp
ggml_tensor* k_write = ggml_cpy(ctx, k, k_slot);
ggml_build_forward_expand(gf, k_write);

// Force dependency
ggml_tensor* zero = ggml_scale(ctx, k_write, 0.0f);
k_attn = ggml_add(ctx, k_view, zero);
```

**Result:** Still gibberish.

**Issue:** GGML doesn't guarantee the copy completes before the view is used, even with explicit dependencies.

### 4. Always Read from Cache (Both Prefill and Decode)
**Approach:** Write to cache then immediately read from it for both prefill and decode.

**Implementation:**
```cpp
// Write to cache
ggml_cpy(ctx, k, k_slot);

// Always read from cache
n_attn = pos + seq_len;
k_attn = ggml_view_3d(ctx, kv->k, ..., n_attn, ...);
```

**Result:** Gibberish.

**Issue:** During prefill, reading from cache in the same graph execution as writing doesn't work - the write hasn't completed yet.

### 5. Original Approach (Baseline)
**Approach:** Use `ggml_cpy` with views, use computed K/V during prefill, read cache during decode.

**Implementation:**
```cpp
ggml_cpy(ctx, k, k_slot);  // Write to cache

if (!is_decode) {
    k_attn = k;  // Use computed K directly
} else {
    k_attn = ggml_view_3d(ctx, kv->k, ...);  // Read from cache
}
```

**Result:** Gibberish (this was the original commit 1c2517d state).

**Issue:** Unknown - this should work in theory.

## Key Findings

### Cache Memory Verification
Added debug output to check cache memory after graph execution:

```
[MAIN] After prefill, layer 0 k_cache sum[0:100] = -0.0114926
[MAIN] After decode step 0, layer 0 k_cache sum[0:100] = -0.0114926
```

**Critical Discovery:** The cache sum is IDENTICAL before and after decode step 0, suggesting the cache write during decode isn't working at all!

### GGML Context Architecture
- **KV Cache Context:** Persistent context (`kv.ctx`) that lives across forward passes
- **Compute Context:** Temporary context created for each forward pass, freed after
- **Issue:** Operations in compute context may not properly modify tensors in the persistent context

### Tensor Operations Behavior
- `ggml_cpy(src, dst)`: Should write to `dst->data` during graph execution
- `ggml_view_3d(tensor, ...)`: Creates a view descriptor pointing to tensor's memory
- `ggml_set_*`: Returns a NEW tensor, doesn't modify the input tensor

## Root Cause Hypothesis

The fundamental issue is likely one of:

1. **Memory Aliasing:** GGML's graph executor doesn't understand that views of persistent cache tensors should be treated specially

2. **Execution Ordering:** Even with explicit `ggml_build_forward_expand` calls, GGML may reorder operations in ways that break cache coherence

3. **Context Isolation:** Operations in the compute context may not be able to properly modify memory owned by a different context

4. **View Semantics:** Views created in the compute context may not properly reference the persistent cache memory

## Recommended Next Steps

### Option 1: Study llama.cpp Implementation in Detail
- Examine exactly how llama.cpp structures its KV cache
- Check if they use a different context management strategy
- Look for any special GGML flags or operations we're missing

### Option 2: Implement Flash Attention
- Use `ggml_flash_attn` if it supports GQA
- This may handle KV cache internally and avoid the manual cache management issues

### Option 3: Restructure Cache Management
- Allocate cache in the compute context instead of a separate persistent context
- Copy cache state between forward passes manually
- This would be slower but might work correctly

### Option 4: Debug GGML Internals
- Add logging to GGML's graph executor to see operation order
- Verify that `ggml_cpy` is actually writing to the cache memory
- Check if there are any GGML flags or settings that affect cache behavior

## Files Modified

- `src/transformer.cpp`: Multiple iterations of cache write/read logic
- `example/main.cpp`: Added debug output to verify cache state
- All changes on branch `fix-kv-cache`

## Conclusion

After extensive investigation, the KV cache implementation cannot be fixed with simple changes to the existing architecture. The issue requires either:
- A fundamental restructuring of how cache memory is managed
- Using different GGML operations that we haven't discovered yet
- Finding and fixing a bug in how we're using GGML's API

The original commit 1c2517d was marked "output still incoherent" for good reason - the KV cache has never worked correctly in this codebase.
