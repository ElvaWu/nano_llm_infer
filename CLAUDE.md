# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and run

```bash
cmake -S . -B build
cmake --build build -j$(nproc)
```

Run CLI inference:

```bash
./build/nano_llm_infer_main
./build/nano_llm_infer_main "Your prompt here"
```

Run HTTP server:

```bash
./build/nano_llm_infer_server
```

## Tests

Run all tests:

```bash
ctest --test-dir build --output-on-failure
```

Run a single test by name pattern:

```bash
ctest --test-dir build --output-on-failure -R unit_tokenizer
```

Other common single-test targets include:
- `unit_generation`
- `unit_generation_server`
- `unit_generation_batching`
- `unit_model_smoke`

## Optional model-backed tests

`unit_model_smoke` and `unit_generation_batching` use `NANO_LLM_TEST_MODEL` configured at CMake time.

```bash
cmake -S . -B build \
  -DNANO_LLM_TEST_MODEL=/absolute/path/to/model.gguf
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure -R unit_model_smoke
```

Check the configured value:

```bash
grep NANO_LLM_TEST_MODEL build/CMakeCache.txt
```

## High-level architecture

Inference and serving are split into reusable runtime pieces plus thin entrypoints:

1. **Model + tokenizer loading**
   - `src/gguf_loader.cpp` / `include/gguf_loader.h`: parses GGUF metadata and maps tensors.
   - `src/tokenizer.cpp` / `include/tokenizer.h`: tokenization and chat template handling.

2. **Core transformer execution**
   - `src/transformer/*` / `include/transformer.h`: block stack, attention, output head.
   - `src/kv_cache.cpp` / `include/kv_cache.h`: per-layer KV cache allocation/state.

3. **Generation runtime**
   - `src/generation.cpp` / `include/generation.h`: `ModelRuntime`, session lifecycle, prefill + token-by-token decode stream.

4. **HTTP batching/streaming layer**
   - `src/generation_server.cpp` / `include/generation_server.h`: request parsing, stream/non-stream responses, queueing active requests via `GenerationService`.

5. **Entrypoints**
   - `example/main.cpp`: local CLI generation path.
   - `example/server.cpp`: HTTP chat-completions server path.

## Dependency boundary

`third_party/ggml/` is vendored. Avoid edits there unless intentionally updating the upstream dependency.
