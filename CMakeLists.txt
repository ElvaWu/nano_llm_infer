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

add_library(nano_llm_infer_lib
    src/gguf_loader.cpp
    src/tokenizer.cpp
    src/transformer/attention.cpp
    src/transformer/output_head.cpp
    src/transformer/transformer.cpp
    src/transformer/block.cpp
    src/kv_cache.cpp
    src/generation.cpp
    src/generation_server.cpp
)

target_include_directories(nano_llm_infer_lib PUBLIC
      include
      third_party/cpp-httplib/include
      third_party/nlohmann_json/include
  )

target_link_libraries(nano_llm_infer_lib PUBLIC ggml pthread)

add_executable(nano_llm_infer_main
    example/main.cpp
)

target_link_libraries(nano_llm_infer_main PRIVATE nano_llm_infer_lib)

add_executable(nano_llm_infer_server
    example/server.cpp
)

target_link_libraries(nano_llm_infer_server PRIVATE nano_llm_infer_lib)


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
