# Debugging Notes - nano_llm_infer

## Current Status: GQA Implementation Investigation

### Problem
Model runs without crashing but produces gibberish output. The attention mechanism with GQA (32 query heads, 4 KV heads) needs fixing.

### Root Cause Analysis - UPDATED

#### ggml_can_mul_mat Validation (ggml.c:3202-3208)
```cpp
static inline bool ggml_can_mul_mat(const struct ggml_tensor * t0, const struct ggml_tensor * t1) {
    return (t0->ne[0]           == t1->ne[0])  &&  // inner dimensions must match
           (t1->ne[2]%t0->ne[2] == 0)          &&  // t1 broadcastable over t0 in dim 2
           (t1->ne[3]%t0->ne[3] == 0);             // t1 broadcastable over t0 in dim 3
}
```

**Critical Discovery: The assertion checks if t1 (Q) is broadcastable over t0 (K), NOT the reverse!**

**Prefill case (works):**
- Q: `[64, 32, 21]` (head_dim=64, n_heads=32, seq_len=21)
- K: `[64, 4, 21]`  (head_dim=64, n_kv_heads=4, seq_len=21)
- Check: `21 % 21 == 0` ✓ PASS

**Decode case (FAILS):**
- Q: `[64, 32, 1]`  (single new token)
- K: `[64, 4, 22]`  (cache now has 22 positions)
- Check: `1 % 22 == 0` ✗ FAIL - 1 is not divisible by 22!

**The problem:** During decode, Q has seq_len=1 but K has seq_len=n_filled (growing). The validation `q.ne[2] % k.ne[2] == 0` fails because 1 is not divisible by 22, 23, 24, etc.

### Why Flash Attention Fails

Flash attention's `ggml_can_mul_mat` assertion is designed for cases where:
1. Both Q and K have the same sequence length (prefill)
2. Q's sequence length is a multiple of K's (batch processing)

It does NOT support the decode pattern where Q has seq_len=1 and K has seq_len=n_filled.

### Solution Options

#### Option 1: Use manual attention for decode, flash_attn for prefill
- Prefill: Use `ggml_flash_attn_ext` (fast, works with same seq_len)
- Decode: Use manual `ggml_mul_mat` attention (slower but works with mismatched seq_len)
- **Pros**: Leverages flash attention where possible
- **Cons**: Two code paths to maintain

#### Option 2: Use manual attention for everything
- Always use `ggml_mul_mat` based attention with proper GQA handling
- **Pros**: Single code path, works for all cases
- **Cons**: Slower prefill (but prefill is one-time cost)

#### Option 3: Patch GGML to relax the assertion
- Modify `ggml_can_mul_mat` to allow decode case
- **Pros**: Would enable flash_attn for decode
- **Cons**: Requires modifying vendored GGML code (violates CLAUDE.md)

### Recommended Approach: Option 2

Use manual attention with proper GQA broadcasting. The key is to use `ggml_mul_mat` which DOES support broadcasting when dimensions are multiples.

**Correct implementation:**
1. Keep Q, K, V in `[head_dim, n_heads/n_kv_heads, seq_len]` layout
2. Use `ggml_mul_mat(k, q)` which supports K broadcasting to Q
3. The matmul will automatically broadcast K's 4 heads to match Q's 32 heads

### Previous Attempts

#### Attempt 1: Direct flash_attn_ext
- **Status**: FAILED - assertion fails during decode
- **Reason**: `ggml_can_mul_mat` requires Q's seq_len to be divisible by K's seq_len
- **Error**: `1 % 22 == 0` is false

#### Attempt 2: Manual GQA with ggml_repeat
- **Status**: Runs but produces gibberish
- **Implementation**: Expand K/V from [64,4,seq] to [64,32,seq] using ggml_repeat
- **Possible issues**: Incorrect permutation or expansion logic

### Next Steps

1. Implement manual attention without ggml_repeat expansion
2. Rely on ggml_mul_mat's native broadcasting support
3. Verify tensor layouts match GGML's expectations

### Technical Details

#### Tensor Layouts (Target)
- Q after RoPE: `[head_dim=64, n_heads=32, seq_len]`
- K after RoPE: `[head_dim=64, n_kv_heads=4, seq_len]`
- V: `[head_dim=64, n_kv_heads=4, seq_len]`

#### GGML mul_mat Broadcasting
From ggml.c:3205-3207, `ggml_mul_mat(a, b)` requires:
- `a->ne[0] == b->ne[0]` (inner dimension match)
- `b->ne[2] % a->ne[2] == 0` (b broadcastable over a in dim 2)
- `b->ne[3] % a->ne[3] == 0` (b broadcastable over a in dim 3)

For `ggml_mul_mat(k, q)` with K=[64,4,seq] and Q=[64,32,seq]:
- `64 == 64` ✓
- `32 % 4 == 0` ✓
- `1 % 1 == 0` ✓

This should work!

### Files Modified
- `src/transformer.cpp`: Attention implementation
- `include/transformer.h`: Function signatures
- `example/main.cpp`: Two-phase generation

### Build Commands
```bash
cmake --build build -j$(nproc)
./build/nano_llm_infer_main "Who are you?"
```
