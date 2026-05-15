#pragma once

#include <cstring>
#include <unordered_map>
#include <vector>

#include "gguf.h"

namespace nano_llm_infer {

inline thread_local std::unordered_map<const ggml_tensor*, std::vector<uint8_t>> graph_input_data;

inline void set_graph_input_data(ggml_tensor* tensor, const void* data, size_t size) {
    ggml_set_input(tensor);
    graph_input_data[tensor] = std::vector<uint8_t>(
        static_cast<const uint8_t*>(data),
        static_cast<const uint8_t*>(data) + size);
    if (tensor->data != nullptr) {
        std::memcpy(tensor->data, data, size);
    }
}

inline const std::vector<uint8_t>* get_graph_input_data(const ggml_tensor* tensor) {
    const auto it = graph_input_data.find(tensor);
    return it == graph_input_data.end() ? nullptr : &it->second;
}

inline void clear_graph_input_data() {
    graph_input_data.clear();
}

}  // namespace nano_llm_infer
