# nano_llm_infer

`nano_llm_infer` is a small C++17 GGUF language-model inference runtime built on top of
vendored `ggml`. It includes:

- GGUF metadata, tensor, and tokenizer loading.
- Transformer execution with KV-cache backed generation.
- Streaming chat generation through a C++ API.
- An OpenAI-compatible HTTP `/v1/chat/completions` endpoint.
- In-process benchmark tooling for offline throughput and online serving workloads.
- Lightweight standalone unit tests registered with CTest.

The default runtime model path is:

```text
model_file/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

The runtime accepts `llama` and `qwen2` GGUF architectures. Pass `--model PATH`
to the direct-generation, server, or benchmark binaries, or set
`RuntimeConfig::model.path` in code. TinyLlama remains the default.

## Repository Layout

```text
include/                 Public headers
src/                     Core implementation
src/transformer/         Transformer block, attention, and output-head internals
example/run_generation.cpp
                         Direct streaming generation example
example/run_http_server.cpp
                         HTTP chat-completions server entrypoint
example/run_benchmark.cpp
                         Benchmark entrypoint
tests/                   Standalone unit tests using tests/test_util.h
third_party/ggml         Vendored ggml backend
third_party/cpp-httplib  Vendored HTTP server dependency
third_party/nlohmann_json
                         Vendored JSON dependency
model_file/              Local model artifacts for smoke tests and examples
```

Build directories such as `build/`, `build_cuda/`, `build_debug/`, and `build_test/`
are generated artifacts and should not be edited manually.

## Requirements

- CMake 3.20 or newer.
- A C++17 compiler.
- POSIX threads.
- A GGUF model file.
- Optional: CUDA-capable `ggml` configuration for GPU execution.
- Optional: NVML headers and library for benchmark GPU monitoring.

The project vendors `ggml`, `cpp-httplib`, and `nlohmann/json`, so no external package
manager is required for those dependencies.

## Build

Configure and build a normal CPU-oriented build:

```bash
cmake -S . -B build
cmake --build build -j4
```

The main targets are:

```text
nano_llm_infer_lib       Static library used by examples and tests
nano_llm_infer_main      Direct generation CLI example
nano_llm_infer_server    HTTP server
nano_llm_benchmark       In-process benchmark runner
unit_*                   Standalone test executables
```

### CUDA Build

CUDA is enabled through `ggml` CMake options. For example:

```bash
cmake -S . -B build_cuda -DGGML_CUDA=ON
cmake --build build_cuda -j4
```

When `GGML_CUDA` is enabled, the project compiles with `NANO_LLM_ENABLE_CUDA`.
If NVML is found, benchmark GPU utilization and VRAM metrics are enabled with
`NANO_LLM_ENABLE_NVML`.

## Run Direct Generation

Build the example:

```bash
cmake --build build --target nano_llm_infer_main -j4
```

Run with the default TinyLlama model path:

```bash
./build/nano_llm_infer_main "Who are you?"
```

Run Qwen2.5 1.5B Instruct:

```bash
./build/nano_llm_infer_main \
  --model model_file/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  "Introduce yourself in one sentence."
```

`nano_llm_infer_main` constructs a `ModelRuntime`, formats the prompt with the
model's text chat template, and streams generated text to stdout.

## Run the HTTP Server

Build the server:

```bash
cmake --build build --target nano_llm_infer_server -j4
```

Start it:

```bash
./build/nano_llm_infer_server \
  --model model_file/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --threads 4 \
  --max-active 2 \
  --max-decode-batch 2 \
  --max-tokens 128 \
  --max-seq-len 512
```

Qwen uses the same server options:

```bash
./build/nano_llm_infer_server \
  --model model_file/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --threads 4 \
  --max-active 2 \
  --max-decode-batch 2 \
  --max-tokens 128 \
  --max-seq-len 512
```

Available server options:

```text
--model PATH
--host IPv4
--port N
--threads N
--max-active N
--max-decode-batch N
--max-tokens N
--max-seq-len N
```

Health check:

```bash
curl http://127.0.0.1:8080/healthz
```

Non-streaming chat completion:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "tinyllama",
    "stream": false,
    "messages": [
      {"role": "user", "content": "Write a short haiku about inference."}
    ],
    "max_tokens": 64
  }'
```

Streaming chat completion:

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "tinyllama",
    "stream": true,
    "messages": [
      {"role": "user", "content": "Explain KV cache in one paragraph."}
    ],
    "max_tokens": 128
  }'
```

The endpoint accepts OpenAI-style `model`, `messages`, `stream`, and `max_tokens`
fields. Streaming responses use server-sent events and terminate with `[DONE]`.

## Benchmark

Build the benchmark runner:

```bash
cmake --build build --target nano_llm_benchmark -j4
```

Run a small offline benchmark:

```bash
./build/nano_llm_benchmark \
  --model model_file/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --mode offline \
  --requests 4 \
  --concurrency 2 \
  --input-tokens 128 \
  --output-tokens 32
```

Run the same workload with Qwen:

```bash
./build/nano_llm_benchmark \
  --model model_file/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --mode offline \
  --requests 4 \
  --concurrency 2 \
  --input-tokens 128 \
  --output-tokens 32
```

Run one of the presets:

```bash
./build/nano_llm_benchmark --preset chat
./build/nano_llm_benchmark --preset throughput
./build/nano_llm_benchmark --preset latency
./build/nano_llm_benchmark --preset long_context
```

Useful benchmark options:

```text
Runtime:
  --model PATH
  --threads N
  --context-tokens N

Benchmark:
  --preset chat|throughput|latency|long_context
  --mode offline|online

Traffic:
  --requests N
  --concurrency N
  --request-rate QPS
  --warmup-requests N
  --seed N

Workload:
  --input-tokens N
  --output-tokens N
  --stop-sequences on|off
  --ignore-eos on|off

Serving:
  --decode-batch-size N

Output:
  --export-csv on|off
```

The benchmark reports output throughput, prompt throughput, total model throughput,
request throughput, latency, time to first token, batch size, finish reasons, and
optional GPU monitoring. With `--export-csv on`, summaries are appended to a
timestamped `benchmark_YYYY_MM_DD_HH_MM.csv` file in the current directory.

## Tests

Configure, build, and run the CTest suite:

```bash
cmake -S . -B build
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

Run a focused test while iterating:

```bash
cmake --build build --target unit_benchmark -j4
./build/unit_benchmark
```

Model-backed TinyLlama and Qwen tests can use explicit GGUF paths:

```bash
cmake -S . -B build_test \
  -DNANO_LLM_TEST_MODEL=/abs/path/tinyllama.gguf \
  -DNANO_LLM_TEST_QWEN_MODEL=/abs/path/qwen2.5-1.5b-instruct.gguf
cmake --build build_test -j4
ctest --test-dir build_test --output-on-failure
```

If either option is empty, the corresponding model-backed checks are skipped.
`NANO_LLM_TEST_QWEN_MODEL` may also be supplied as an environment variable when
running `unit_tokenizer` or `unit_model_smoke` directly.

## Library Usage

The public generation API lives in `include/generation.h`.

Minimal streaming example:

```cpp
#include <iostream>

#include "generation.h"

int main() {
    nano_llm_infer::RuntimeConfig config;
    config.model.path = "model_file/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";
    config.model.compute_threads = 4;
    config.limits.context_tokens = 512;

    nano_llm_infer::ModelRuntime runtime(config);

    nano_llm_infer::GenerationRequest request;
    request.messages.push_back({"user", "What is KV cache?"});
    request.options.max_tokens = 128;

    runtime.generate_stream(request, [](const nano_llm_infer::GenerationEvent& event) {
        if (!event.error.empty()) {
            std::cerr << "error: " << event.error << '\n';
            return false;
        }
        std::cout << event.text << std::flush;
        return event.finish_reason.empty();
    });

    std::cout << '\n';
    return 0;
}
```

`ModelRuntime` also exposes lower-level session APIs for schedulers and services:

- `create_session()`
- `prefill()` and `prefill_batch()`
- `decode_next()` and `decode_batch()`
- `assign_kv_slot()` and `release_kv_slot()`

The HTTP server and benchmark runner use these APIs to manage concurrent generation
and decode batching.

## Runtime Defaults

The default `RuntimeConfig` values are:

```text
model.path              model_file/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
model.compute_threads   4
limits.context_tokens   512
limits.default_max_new_tokens 600
memory.prefill_ctx_bytes       512 MiB
memory.decode_ctx_bytes        256 MiB
```

The benchmark entrypoint increases graph arena defaults to 768 MiB for prefill and
512 MiB for decode.

## Development Notes

- Use C++17 and the style already established in the edited file.
- Keep public headers in `include/` minimal and pair them with source files in `src/`.
- Register new tests in `CMakeLists.txt` as `unit_*` executables.
- Prefer deterministic tests for token counts, boundaries, scheduler behavior, and
  error paths instead of exact timing.
- Do not commit generated build directories, local GGUF files, benchmark CSVs, or
  profiler outputs.
