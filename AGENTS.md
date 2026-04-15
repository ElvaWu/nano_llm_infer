# Repository Guidelines

## Project Structure & Module Organization
Core library code lives in `src/` with public headers in `include/`. The main components are `gguf_loader`, `tokenizer`, and `transformer`, which together load GGUF weights and run inference. `example/main.cpp` is the reference executable for local generation. Top-level `test_*.cpp`, `check_*.cpp`, and `debug_*.cpp` files are lightweight diagnostics and manual test programs rather than a formal test suite. Model assets are stored in `model_file/`, and `third_party/ggml/` is a vendored dependency that should not be edited unless you are intentionally updating GGML.

## Build, Test, and Development Commands
Use CMake with an out-of-source build:

```bash
rm -rf build
cmake -S . -B build
cmake --build build -j"$(nproc)"
```

Run the example inference binary with:

```bash
./build/nano_llm_infer_main
./build/nano_llm_infer_main "Hello"
```

There is no unified `ctest` target yet. Build and run individual diagnostics manually when changing tokenizer or inference behavior, for example by compiling the existing `test_*.cpp` or `debug_*.cpp` helpers if they are not already present in `build/`.

## Coding Style & Naming Conventions
Target C++17 and keep headers in `include/` synchronized with implementations in `src/`. Follow the existing style: 4-space indentation, opening braces on the same line, `snake_case` for functions and local variables, and `PascalCase` for types such as `TransformerBlock`. Keep code inside the `nano_llm_infer` namespace. Prefer small, direct changes over broad refactors, and avoid modifying `third_party/ggml/` in normal feature work.

## Testing Guidelines
Because the repository does not use a test framework yet, every behavior change should include a reproducible manual check. Reuse or extend `test_tokenizer.cpp`, `test_lookup.cpp`, or similar top-level programs, and name new helpers with `test_*.cpp` or `debug_*.cpp` so their purpose is obvious. Document the exact build and run command in your PR.

## Commit & Pull Request Guidelines
The current history uses short, informal commit subjects. Prefer concise, imperative messages such as `fix tokenizer merge ranks` or `add logits sampling test`. Keep each commit focused. PRs should describe the behavioral change, list validation commands, note any model files required, and include sample output when changing generation or tokenizer behavior.
