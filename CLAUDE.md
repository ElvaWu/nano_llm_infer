# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and run

```bash
cmake -S . -B build
cmake --build build -j$(nproc)
```

Run the example inference binary (expects a TinyLlama GGUF under `model_file/`):

```bash
./build/nano_llm_infer_main
./build/nano_llm_infer_main "Your prompt here"
```

## Unit tests and diagnostics

The repository now includes a CTest-backed unit test suite under `tests/`:

- `unit_gguf_loader`
- `unit_tokenizer`
- `unit_kv_cache`
- `unit_rms_norm`
- `unit_model_smoke` (optional model-backed smoke test)

Run all tests:

```bash
ctest --test-dir build --output-on-failure
```

Run only the optional model smoke test:

```bash
ctest --test-dir build --output-on-failure -R unit_model_smoke
```

### Optional model-backed test (`NANO_LLM_TEST_MODEL`)

By default, `unit_model_smoke` skips when no model path is configured.
To enable it, set `NANO_LLM_TEST_MODEL` at configure time:

```bash
cmake -S . -B build \
  -DNANO_LLM_TEST_MODEL=/root/autodl-tmp/repos/nano_llm_infer/model_file/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

You can verify the configured value with:

```bash
grep NANO_LLM_TEST_MODEL build/CMakeCache.txt
```

### Existing top-level diagnostics

Manual diagnostic binaries (`test_*.cpp`, `debug_*.cpp`, `check_*.cpp`) are still available for deeper investigation.

Build and run one manually:

```bash
g++ -std=c++17 -O2 test_tokenizer.cpp \
  -Iinclude -Ibuild/third_party/ggml/include \
  -Lbuild -lnano_llm_infer_lib -lggml \
  -o test_tokenizer
./test_tokenizer
```

## Architecture overview

Minimal C++17 GGUF inference stack for TinyLlama-like LLaMA-family models, built directly on GGML.

### End-to-end flow

1. `GGUFLoader` loads metadata and tensor descriptors from GGUF, then mmaps model weights and points GGML tensors into the mmap region (no weight copies).
2. `Tokenizer` loads vocab/merges/chat template metadata and encodes prompt text to token IDs.
3. `Transformer` runs forward passes over embeddings using GGML graphs.
4. `OutputHead` projects hidden states to logits.
5. `KVCache` stores per-layer K/V states across decode steps.
6. `example/main.cpp` drives two-phase generation:
   - **Prefill**: process full prompt once
   - **Decode**: process one token at a time using KV cache

### Core modules

- `src/gguf_loader.cpp` + `include/gguf_loader.h`
  - Parses GGUF key/value metadata into typed variants.
  - Builds `Model` with architecture parameters (`n_layers`, `n_heads`, RoPE settings, etc.) and tensor map.
  - Uses mmap-backed tensor data for efficient loading.

- `src/tokenizer.cpp` + `include/tokenizer.h`
  - Supports BPE merge-rank tokenization when merges exist and a DP/Viterbi-style fallback path.
  - Handles SentencePiece-style space marker normalization.
  - Applies TinyLlama-style chat templates with `<|user|>` / `<|assistant|>` / `<|system|>` markers.

- `src/transformer.cpp` + `include/transformer.h`
  - `TransformerBlock`: pre-norm RMSNorm → GQA self-attention (with RoPE) → residual → SwiGLU MLP → residual.
  - Attention writes current K/V into cache and reads historical K/V views for causal attention.
  - `Transformer` stacks blocks; `OutputHead` performs final norm + LM head matmul.

- `src/kv_cache.cpp` + `include/kv_cache.h`
  - Allocates one GGML context for all layer caches.
  - Per layer: F32 K and V tensors shaped `[head_dim, n_kv_heads, max_seq_len]`.
  - `KVCache::n_filled` tracks populated sequence positions.

### GGML execution model and memory constraints

GGML ops are graph-building (deferred execution), not eager math. Real compute happens only when the graph is executed (e.g. `ggml_graph_compute_with_ctx`).

Because GGML uses fixed-size arenas (`ggml_context`), context sizing is critical:

- compute tensors + temporary work buffers must fit the compute context
- KV cache tensors must fit the KV cache context
- undersized contexts can cause `GGML_ABORT` or crash-level failures (null dereference paths in release)

When changing tensor shapes, sequence lengths, quantization, or attention behavior, re-check context sizing assumptions in `example/main.cpp` and `src/kv_cache.cpp`.

## Build targets

- `nano_llm_infer_lib`: core inference library (`gguf_loader`, `tokenizer`, `transformer`, `kv_cache`)
- `nano_llm_infer_main`: example generation executable
- `ggml`: vendored dependency under `third_party/ggml/` (treat as external; do not modify unless explicitly required)
