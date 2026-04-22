# Repository Guidelines

## Project Structure & Module Organization
Core library code lives in `src/` and public headers live in `include/`. The main runtime pieces are `gguf_loader`, `tokenizer`, `transformer`, and `kv_cache`. The reference executable is `example/main.cpp`. Top-level `test_*.cpp`, `check_*.cpp`, and `debug_*.cpp` files are manual diagnostics rather than a formal test suite. Model assets are stored in `model_file/`. Treat `third_party/ggml/` as vendored code and avoid editing it unless you are intentionally updating GGML.

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

There is no unified `ctest` target. Build and run the relevant helper for your change, for example:

```bash
g++ -std=c++17 -Iinclude -Ithird_party/ggml/include test_tokenizer.cpp build/libnano_llm_infer_lib.a -o /tmp/test_tokenizer
/tmp/test_tokenizer
```

Prefer reusing existing `test_*` or `debug_*` programs before creating new ones.

## Coding Style & Naming Conventions
Target C++17. Keep declarations in `include/` synchronized with implementations in `src/`. Follow the existing style: 4-space indentation, opening braces on the same line, `snake_case` for functions and local variables, and `PascalCase` for types such as `TransformerBlock`. Keep code inside the `nano_llm_infer` namespace. Favor small, direct changes over broad refactors.

## Testing Guidelines
Every behavior change should include a reproducible manual check. Extend an existing top-level helper such as `test_tokenizer.cpp`, `test_lookup.cpp`, or a `debug_*.cpp` file when possible. Name new helpers `test_*.cpp` or `debug_*.cpp` so their purpose is obvious. Document the exact compile and run commands in your PR.

## Commit & Pull Request Guidelines
Recent history uses short, imperative commit subjects such as `fix cache writing & rm test files` and `Add comprehensive investigation document for KV cache issues`. Keep commits focused and descriptive. PRs should explain the behavioral change, list validation commands, note required model files, and include sample output when generation, tokenizer, or cache behavior changes.
