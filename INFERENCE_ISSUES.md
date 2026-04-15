# Inference Issues - Debugging Notes

## Problem
The model generates gibberish output, especially for longer prompts (>20 tokens).

## Root Causes Identified

### 1. Memory Exhaustion
- The compute context was set to 512MB, which is insufficient
- Error: `ggml_new_object: not enough space in the context's memory pool (needed 540296832, available 536870912)`
- Increased to 1GB but issues persist

### 2. Broken "Simple Attention" Implementation
- Commit 2a05e9b removed KV cache to use "simple attention (very slow but works)"
- **This doesn't actually work** - it produces gibberish for most inputs
- The approach of recomputing all tokens from scratch on each step has fundamental issues

### 3. Sequence Length Threshold
- Prompts with 17-20 tokens: Partially work
- Prompts with 21+ tokens: Immediate gibberish
- Suggests memory/computation issues scale with sequence length

## Test Results

### Working Cases
- Very short prompts (16-17 tokens) sometimes produce reasonable first few tokens
- Prefill phase works correctly for all prompt lengths

### Failing Cases
- "Who are you?" (21 tokens): Produces "There is the best to this, a reciprobably|system|"
- "What is 2+2?" (25 tokens): Produces "|what'srocksun|drink|user|"
- Longer sequences quickly degrade into gibberish

## Recommended Solution

**Revert to KV cache implementation** (commit 1c2517d) and fix the coherence issues there.

The "simple attention" approach without KV cache is:
1. Computationally expensive (O(n²) recomputation)
2. Memory intensive (allocates full context each iteration)
3. **Fundamentally broken** in the current implementation

The KV cache approach is the standard way to implement LLM inference and should be debugged/fixed rather than abandoned.

## Changes Made During Debugging

1. Increased context size from 512MB to 1GB in `example/main.cpp`
2. Added/removed various debug outputs
3. Tested with/without manual BOS token insertion

## Next Steps

1. Revert to commit 1c2517d (KV cache version)
2. Debug the "output still incoherent" issue in that version
3. Compare attention computation with reference implementations (llama.cpp)
4. Verify RoPE application and attention mask are correct
