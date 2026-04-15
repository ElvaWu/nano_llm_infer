# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build

```bash
cmake -S . -B build
cmake --build build -j$(nproc)
```

Run the example binary (loads TinyLlama from `model_file/`):
```bash
./build/nano_llm_infer_main
./build/nano_llm_infer_main "Your prompt here"
```

To build individual diagnostic programs (top-level `test_*.cpp`, `debug_*.cpp`, `check_*.cpp`):
```bash
g++ -std=c++17 -O2 test_tokenizer.cpp -Iinclude -Ibuild/third_party/ggml/include \
    -Lbuild -lnano_llm_infer_lib -lggml -o test_tokenizer
```

No test framework or linting is configured.

## Architecture

Minimal C++17 inference library for GGUF-format LLMs targeting TinyLlama-1.1B. All library code is in the `nano_llm_infer` namespace.

**Execution model:** GGML is a deferred-execution DAG — `ggml_mul_mat`, `ggml_add`, etc. only record nodes; no arithmetic runs until `ggml_graph_compute_with_ctx` is called. Every op takes a `ggml_context*` (memory arena) and returns a `ggml_tensor*`. Contexts are sized upfront; OOM in a context causes `GGML_ABORT` or a null-dereference crash in release builds.

**Weight loading (`gguf_loader.h/cpp`):** `GGUFLoader::load_model(path)` mmaps the GGUF file (`MAP_SHARED`, read-only), creates a small 10 MB ggml context with `no_alloc=true` for tensor metadata, and sets each `tensor->data` to point directly into the mmap region. Metadata is stored in `GGUFLoader::metadata_` as `std::unordered_map<string, gguf_kv>` with typed variants; `get_metadata<T>()` handles retrieval. `get_tensor_by_name()` returns `nullptr` on miss — callers must null-check.

**Tokenizer (`tokenizer.h/cpp`):** `Tokenizer::load_from_gguf()` reads vocab/scores/merges from GGUF metadata. Two encoding strategies: BPE merge-rank (when merges are present) and Viterbi/DP SentencePiece-style. `TokenizerMode::LEGACY` vs `STRICT_MODEL` controls byte-fallback. SentencePiece models use U+2581 as a space marker. `apply_chat_template()` formats messages with `<|user|>` / `<|assistant|>` tags.

**Transformer (`transformer.h/cpp`):** `TransformerBlock` implements pre-norm RMSNorm → GQA self-attention → SwiGLU MLP. `Transformer` stacks `n_layers` blocks. Each block stores raw `ggml_tensor*` weight pointers from the model and a `ggml_context*` set via `set_compute_ctx()` before each forward pass. `OutputHead` applies final RMSNorm then `ggml_mul_mat` against `output.weight` to produce logits.

**KV cache (`kv_cache.h/cpp`):** `make_kv_cache(model, max_seq_len)` allocates a dedicated ggml context holding per-layer K and V tensors of shape `[head_dim, n_kv_heads, max_seq_len]` in F32. `KVCache::n_filled` tracks how many positions are populated. The attention code in `TransformerBlock::self_attention` writes K/V via `ggml_cpy` into cache slots and reads back a view over `[0..pos]` for decode steps. GQA broadcasting (n_heads / n_kv_heads) is handled by GGML's `mul_mat` when `ne[2]` ratios are valid multiples.

**Two-phase generation (`example/main.cpp`):**
1. **Prefill** — all prompt tokens processed in one forward pass (512 MB compute context). Logits for the last position select the first generated token.
2. **Decode** — one token per step, re-using the populated KV cache (64 MB compute context). Each step: new context → `set_compute_ctx` → embed single token → `forward_logits(embed, kv.n_filled, gf, &kv)` → argmax → `kv.n_filled++`.

**Build targets:**
- `nano_llm_infer_lib` — static library (`gguf_loader`, `tokenizer`, `transformer`, `kv_cache`)
- `ggml` — vendored static library in `third_party/ggml/` — **do not modify**
- `nano_llm_infer_main` — example executable

**GGML context sizing pitfall:** `ggml_graph_compute_with_ctx` allocates a work buffer (`cplan.work_size`) from the compute context after all tensors are already allocated. If the context is exhausted, the result is a null-dereference SIGSEGV (release) or `GGML_ABORT` (debug). The work buffer for a `Q8_0 × F32` matmul is `ggml_row_size(Q8_0, total_input_elements)`. Size contexts conservatively.

**Quantization note:** `mul_mat` with quantized weights converts F32 inputs to `vec_dot_type` (Q8_0 for Q8_0/Q4_0/Q5_0 weights; Q8_K for K-quants) into the work buffer before computing. The work buffer must be sized to `ggml_row_size(vec_dot_type, ggml_nelements(src1))`.
