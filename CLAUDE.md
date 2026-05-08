# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build

```bash
cmake -S . -B build
cmake --build build -j$(nproc)
```

Configure model-backed tests (optional):

```bash
cmake -S . -B build \
  -DNANO_LLM_TEST_MODEL=/absolute/path/to/model.gguf
cmake --build build -j$(nproc)
```

## Run

CLI inference:

```bash
./build/nano_llm_infer_main
./build/nano_llm_infer_main "Your prompt here"
```

HTTP server:

```bash
./build/nano_llm_infer_server
```

Benchmark entrypoint:

```bash
./build/nano_llm_benchmark
```

## Tests

Run all tests:

```bash
ctest --test-dir build --output-on-failure
```

Run one test target:

```bash
ctest --test-dir build --output-on-failure -R unit_tokenizer
```

Common focused targets:
- `unit_generation`
- `unit_generation_server`
- `unit_generation_batching`
- `unit_model_smoke`
- `unit_benchmark`

Check configured model path for model-backed tests:

```bash
grep NANO_LLM_TEST_MODEL build/CMakeCache.txt
```

## High-level architecture

The project is organized as a reusable inference runtime (`nano_llm_infer_lib`) with thin app entrypoints on top.

1. **Model load and tokenization boundary**
   - `gguf_loader` loads GGUF weights/metadata into an internal `Model`.
   - `tokenizer` is initialized from GGUF metadata and owns chat-template prompt formatting.

2. **Transformer + KV cache core**
   - `src/transformer/*` executes attention/blocks/output head over ggml tensors.
   - `kv_cache` allocates per-layer KV storage used across prefill and decode steps.

3. **Generation runtime (session-oriented)**
   - `ModelRuntime` (`src/generation.cpp`) owns model, tokenizer, and transformer.
   - Request flow is: chat messages -> templated prompt -> token ids -> **prefill** (prompt forward pass) -> iterative decode.
   - Batched decode is implemented in `decode_batch`: active sessions are packed into a temporary batched KV view, one decode step runs, then per-session KV/token state is written back.

4. **Server scheduling and streaming**
   - `GenerationService` (`src/generation_server.cpp`) is a worker-thread scheduler that manages pending/active requests, does per-request prefill, and runs round-robin decode batches across ready sessions.
   - `HttpStreamingServer` exposes `/v1/chat/completions` with OpenAI-style JSON and SSE streaming modes plus `/healthz`.

5. **Entrypoints**
   - `example/main.cpp`: local CLI generation.
   - `example/server.cpp`: HTTP chat-completions server process and flag parsing.
   - `example/benchmark_main.cpp`: benchmark executable wired to `src/benchmark.cpp`.

## Dependency boundary

`third_party/ggml/` is vendored. Avoid edits there unless intentionally updating the dependency.