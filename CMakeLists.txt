cmake_minimum_required(VERSION 3.20)
project(nano_llm_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(BUILD_SHARED_LIBS   OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_METAL          OFF CACHE BOOL "" FORCE)
set(NANO_LLM_TEST_MODEL "" CACHE STRING "Optional GGUF model path for model-backed smoke tests")

add_subdirectory(third_party/ggml)

find_path(NVML_INCLUDE_DIR
    NAMES nvml.h
    PATHS
        /usr/local/cuda/include
        /usr/local/cuda-11.8/targets/x86_64-linux/include
        /usr/local/cuda/targets/x86_64-linux/include
)
find_library(NVML_LIBRARY
    NAMES nvidia-ml
    PATHS
        /usr/lib/x86_64-linux-gnu
        /usr/local/cuda/lib64
        /usr/local/cuda-11.8/targets/x86_64-linux/lib/stubs
        /usr/local/cuda/targets/x86_64-linux/lib/stubs
)

add_library(nano_llm_infer_lib
    src/gguf_loader.cpp
    src/tokenizer.cpp
    src/transformer/attention.cpp
    src/transformer/output_head.cpp
    src/transformer/transformer.cpp
    src/transformer/block.cpp
    src/kv_cache.cpp
    src/generation.cpp
    src/perf_instrumentation.cpp
    src/generation_server.cpp
    src/benchmark.cpp
)

target_include_directories(nano_llm_infer_lib PUBLIC
      include
      third_party/cpp-httplib/include
      third_party/nlohmann_json/include
  )

target_link_libraries(nano_llm_infer_lib PUBLIC ggml pthread)
if(NVML_INCLUDE_DIR AND NVML_LIBRARY)
    target_include_directories(nano_llm_infer_lib PRIVATE ${NVML_INCLUDE_DIR})
    target_link_libraries(nano_llm_infer_lib PRIVATE ${NVML_LIBRARY})
    target_compile_definitions(nano_llm_infer_lib PRIVATE NANO_LLM_ENABLE_NVML)
endif()
target_compile_definitions(nano_llm_infer_lib PRIVATE
    $<$<BOOL:${GGML_CUDA}>:NANO_LLM_ENABLE_CUDA>
)

add_executable(nano_llm_infer_main
    example/run_generation.cpp
)

target_link_libraries(nano_llm_infer_main PRIVATE nano_llm_infer_lib)

add_executable(nano_llm_infer_server
    example/run_http_server.cpp
)

target_link_libraries(nano_llm_infer_server PRIVATE nano_llm_infer_lib)

add_executable(nano_llm_benchmark
    example/run_benchmark.cpp
)

target_link_libraries(nano_llm_benchmark PRIVATE nano_llm_infer_lib)


enable_testing()

add_executable(unit_gguf_loader
    tests/unit_gguf_loader.cpp
)
target_link_libraries(unit_gguf_loader PRIVATE nano_llm_infer_lib)
add_test(NAME unit_gguf_loader COMMAND unit_gguf_loader)

add_executable(unit_tokenizer
    tests/unit_tokenizer.cpp
)
target_link_libraries(unit_tokenizer PRIVATE nano_llm_infer_lib)
add_test(NAME unit_tokenizer COMMAND unit_tokenizer)

add_executable(unit_kv_cache
    tests/unit_kv_cache.cpp
)
target_link_libraries(unit_kv_cache PRIVATE nano_llm_infer_lib)
add_test(NAME unit_kv_cache COMMAND unit_kv_cache)

add_executable(unit_rms_norm
    tests/unit_rms_norm.cpp
)
target_link_libraries(unit_rms_norm PRIVATE nano_llm_infer_lib)
add_test(NAME unit_rms_norm COMMAND unit_rms_norm)

add_executable(unit_generation
    tests/unit_generation.cpp
)
target_link_libraries(unit_generation PRIVATE nano_llm_infer_lib)
add_test(NAME unit_generation COMMAND unit_generation)

add_executable(unit_generation_server
    tests/unit_generation_server.cpp
)
target_link_libraries(unit_generation_server PRIVATE nano_llm_infer_lib)
add_test(NAME unit_generation_server COMMAND unit_generation_server)

add_executable(unit_model_smoke
    tests/unit_model_smoke.cpp
)
target_link_libraries(unit_model_smoke PRIVATE nano_llm_infer_lib)
target_compile_definitions(unit_model_smoke PRIVATE NANO_LLM_TEST_MODEL_PATH="${NANO_LLM_TEST_MODEL}")
add_test(NAME unit_model_smoke COMMAND unit_model_smoke)

add_executable(unit_generation_batching
    tests/unit_generation_batching.cpp
)
target_link_libraries(unit_generation_batching PRIVATE nano_llm_infer_lib)
target_compile_definitions(unit_generation_batching PRIVATE NANO_LLM_TEST_MODEL_PATH="${NANO_LLM_TEST_MODEL}")
add_test(NAME unit_generation_batching COMMAND unit_generation_batching)

add_executable(unit_benchmark
    tests/unit_benchmark.cpp
)
target_link_libraries(unit_benchmark PRIVATE nano_llm_infer_lib)
target_compile_definitions(unit_benchmark PRIVATE NANO_LLM_TEST_MODEL_PATH="${NANO_LLM_TEST_MODEL}")
add_test(NAME unit_benchmark COMMAND unit_benchmark)
