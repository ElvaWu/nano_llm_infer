# KV Cache Investigation and Fixes

## Background

The repository had two approaches to inference:
1. **Commit 2a05e9b**: "Simple attention" without KV cache - recomputes all tokens each step
2. **Commit 1c2517d**: KV cache implementation - marked as "output still incoherent"

The simple attention approach was found to be fundamentally broken (produces gibberish), so we reverted to the KV cache version to fix the coherence issues.

## Work Done

### 1. Identified Issues with Simple Attention (commit 2a05e9b)

**Problems Found:**
- Memory exhaustion: 512MB context insufficient, causing `ggml_new_object` failures
- Recomputing all tokens from scratch on each decode step is broken
- Works partially for 16-20 token prompts, fails completely for 21+ tokens
- Despite commit message claiming it "works", it produces gibberish for most inputs

**Test Results:**
- "Hi" (17 tokens): Generates "Sure|assistant|" initially, then degrades
- "Who are you?" (21 tokens): Immediate gibberish
- Root cause: Memory corruption and/or incorrect tensor indexing during recomputation

### 2. Reverted to KV Cache Implementation (commit 1c2517d)

Checked out the KV cache version which had the infrastructure in place but coherence issues.

### 3. Removed Debug Output

The code had extensive fprintf debug statements throughout `src/transformer.cpp`:
- Removed ~15 debug print statements from `self_attention()`
- Removed debug output from `example/main.cpp`
- Cleaned up code for production readiness

### 4. Investigated KV Cache Read/Write Issues

**The Core Problem:**

During decode, the flow is:
1. Compute Q/K/V for the new token
2. Write K/V at position `pos` in the cache
3. Read cache `[0..pos]` for attention computation

In GGML's deferred execution model:
- Operations are added to a computation graph
- Graph executes later with automatic dependency resolution
- The cache write (`ggml_cpy`) and cache read (via `ggml_view_3d`) must execute in correct order

**Attempted Fixes:**

a) **Original approach**: Read `[0..pos+1]` from cache
   - Issue: Reads uninitialized memory at position `pos` before write completes
   - Result: Gibberish output

b) **Fix attempt 1**: Read `[0..pos]` instead of `[0..pos+1]`
   - Issue: Excludes the current token from attention
   - Result: Program hangs or produces wrong output

c) **Fix attempt 2**: Use `ggml_concat` to combine cache `[0..pos-1]` with current K/V
   - Issue: `ggml_concat` doesn't work as expected in this context
   - Result: Complete gibberish, worse than before

d) **Current state**: Back to reading `[0..pos+1]` from cache
   - Still produces incoherent output
   - Better than alternatives but not correct

### 5. Current Test Results

**Short prompts (16-17 tokens):**
- "Hi" → "Sure|assistant|" (reasonable but stops early)
- "Hello" → "Verbat." (gibberish)

**Longer prompts (20+ tokens):**
- "Who are you?" → "How to use a|assistant @nodds|assistant|" (gibberish)
- "What is 2+2?" → "I?|user|" (gibberish)

## Root Cause Analysis

The KV cache implementation has a fundamental issue with how it synchronizes cache writes and reads in GGML's computation graph.

**The Problem:**

When we create a view of the cache using `ggml_view_3d`, we're creating a tensor descriptor that points to cache memory. When we add `ggml_cpy` to write to the cache, GGML needs to understand the dependency between:
1. The cache write operation
2. The attention computation that reads from the cache

Currently, there's no explicit dependency, so GGML may execute them in the wrong order or the cache view may be created before the write completes.

## Comparison with llama.cpp

llama.cpp uses a different approach:
- Uses `ggml_set_rows` with index tensors instead of `ggml_view_3d` + `ggml_cpy`
- This creates explicit dependencies in the computation graph
- The set_rows operation is atomic and properly ordered

## Recommended Next Steps

### Option 1: Implement llama.cpp-style KV Cache (Recommended)

Study and implement the approach used in llama.cpp:
1. Use `ggml_set_rows` instead of `ggml_view_3d` + `ggml_cpy`
2. Create index tensors to specify which rows to update
3. This ensures proper execution order in the computation graph

### Option 2: Fix Computation Graph Dependencies

Investigate GGML's dependency tracking:
1. Ensure cache write operations are marked as dependencies for attention
2. May require using `ggml_build_forward_expand` in a specific order
3. Or using GGML's synchronization primitives if available

### Option 3: Simplify with Flash Attention

Consider using GGML's flash attention if available:
- `ggml_flash_attn` may handle KV cache internally
- Would simplify the implementation significantly
- Need to verify it supports GQA (Grouped Query Attention)

## Files Modified

- `src/transformer.cpp`: Removed debug output, attempted various KV cache fixes
- `example/main.cpp`: Removed debug output, increased context size to 1GB
- `INFERENCE_ISSUES.md`: Documented the simple attention issues
- `KV_CACHE_INVESTIGATION.md`: This file

## Current Branch

All work is on branch `fix-kv-cache` (commit 05e476f).

The code compiles and runs but produces incoherent output. The KV cache implementation needs fundamental changes to work correctly with GGML's computation graph execution model.
