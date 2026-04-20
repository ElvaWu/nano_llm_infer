cmake_minimum_required(VERSION 3.20)
project(nano_llm_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(BUILD_SHARED_LIBS   OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_METAL          OFF CACHE BOOL "" FORCE)

add_subdirectory(third_party/ggml)

add_library(nano_llm_infer_lib
    src/gguf_loader.cpp
    src/tokenizer.cpp
    src/transformer/attention.cpp
    src/transformer/output_head.cpp
    src/transformer/transformer.cpp
    src/transformer/block.cpp
    src/kv_cache.cpp
)

target_include_directories(nano_llm_infer_lib PUBLIC include)

target_link_libraries(nano_llm_infer_lib PUBLIC ggml)

add_executable(nano_llm_infer_main
    example/main.cpp
)

target_link_libraries(nano_llm_infer_main PRIVATE nano_llm_infer_lib)
