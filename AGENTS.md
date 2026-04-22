# Repository Guidelines

## Project Structure & Module Organization
Core library code lives in `src/` and public headers live in `include/`. The main runtime pieces are `gguf_loader`, `tokenizer`, `transformer`, and `kv_cache`. The reference executable is `example/main.cpp`. Unit tests live under `tests/`. Top-level `test_*.cpp`, `check_*.cpp`, and `debug_*.cpp` files are manual diagnostics. Model assets are stored in `model_file/`. Treat `third_party/ggml/` as vendored code and avoid editing it unless you are intentionally updating GGML.

## Build, Test, and Development Commands
Use an out-of-source CMake build:

```bash
rm -rf build
cmake -S . -B build
cmake --build build -j"$(nproc)"
```

Run the example binary with:

```bash
./build/nano_llm_infer_main
./build/nano_llm_infer_main "Hello"
```

Run the CTest unit suite:

```bash
ctest --test-dir build --output-on-failure
```

Run only the optional model-backed smoke unit test:

```bash
ctest --test-dir build --output-on-failure -R unit_model_smoke
```

To enable model-backed smoke testing, configure with:

```bash
cmake -S . -B build \
  -DNANO_LLM_TEST_MODEL=/root/autodl-tmp/repos/nano_llm_infer/model_file/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

Manual diagnostic helpers are still available when deeper debugging is needed, for example:

```bash
g++ -std=c++17 -Iinclude -Ithird_party/ggml/include test_tokenizer.cpp build/libnano_llm_infer_lib.a -o /tmp/test_tokenizer
/tmp/test_tokenizer
```

## Coding Style & Naming Conventions
Target C++17. Keep declarations in `include/` synchronized with implementations in `src/`. Follow the existing style: 4-space indentation, opening braces on the same line, `snake_case` for functions and local variables, and `PascalCase` for types such as `TransformerBlock`. Keep code inside the `nano_llm_infer` namespace. Favor small, direct changes over broad refactors.

## Testing Guidelines
Every behavior change should include reproducible validation. Prefer adding or extending CTest unit tests under `tests/` for deterministic checks. Keep unit tests model-independent by default.

If a model-backed check is needed, gate it behind `NANO_LLM_TEST_MODEL` and make missing model files skip gracefully. Use top-level `test_*.cpp` / `debug_*.cpp` helpers for targeted investigations when unit tests are not the right fit. Document exact build and run commands in your PR.

## Commit & Pull Request Guidelines
Recent history uses short, imperative commit subjects such as `fix cache writing & rm test files` and `Add comprehensive investigation document for KV cache issues`. Keep commits focused and descriptive. PRs should explain the behavioral change, list validation commands, note required model files, and include sample output when generation, tokenizer, or cache behavior changes.
