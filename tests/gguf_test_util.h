#pragma once

#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gguf.h"
#include "gguf_loader.h"

namespace gguf_test_util {

struct SyntheticGGUFSpec {
    std::string architecture = "llama";
    bool include_embedding_length = true;
    bool include_attention_biases = false;
    bool include_conflicting_llama_metadata = false;

    std::string tokenizer_model;
    std::string tokenizer_pre;
    std::string chat_template;
    std::vector<std::string> tokens;
    std::vector<int32_t> token_types;
    std::vector<std::string> merges;
    int bos_token_id = -1;
    int eos_token_id = -1;
    bool set_add_bos = false;
    bool add_bos = false;
    bool set_add_eos = false;
    bool add_eos = false;
};

class SyntheticGGUFFile {
public:
    SyntheticGGUFFile(std::string file_path, size_t tensor_end)
        : path(std::move(file_path)), tensor_data_end(tensor_end) {
    }

    ~SyntheticGGUFFile() {
        if (!path.empty()) {
            std::remove(path.c_str());
        }
    }

    SyntheticGGUFFile(const SyntheticGGUFFile&) = delete;
    SyntheticGGUFFile& operator=(const SyntheticGGUFFile&) = delete;

    SyntheticGGUFFile(SyntheticGGUFFile&& other) noexcept
        : path(std::move(other.path)), tensor_data_end(other.tensor_data_end) {
        other.path.clear();
        other.tensor_data_end = 0;
    }

    SyntheticGGUFFile& operator=(SyntheticGGUFFile&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        if (!path.empty()) {
            std::remove(path.c_str());
        }
        path = std::move(other.path);
        tensor_data_end = other.tensor_data_end;
        other.path.clear();
        other.tensor_data_end = 0;
        return *this;
    }

    void truncate_tensor_data() const {
        if (tensor_data_end == 0 || ::truncate(path.c_str(), tensor_data_end - 1) != 0) {
            throw std::runtime_error("failed to truncate synthetic GGUF");
        }
    }

    std::string path;
    size_t tensor_data_end = 0;
};

inline std::string make_temp_path(const std::string& label) {
    static int counter = 0;
    return "/tmp/nano_llm_" + label + "_" + std::to_string(getpid()) + "_" +
        std::to_string(counter++) + ".gguf";
}

inline void set_architecture_metadata(gguf_context* context, const std::string& architecture) {
    const std::string prefix = architecture + ".";
    gguf_set_val_str(context, "general.architecture", architecture.c_str());
    gguf_set_val_u32(context, (prefix + "block_count").c_str(), 1);
    gguf_set_val_u32(context, (prefix + "context_length").c_str(), 16);
    gguf_set_val_u32(context, (prefix + "attention.head_count").c_str(), 2);
    gguf_set_val_u32(context, (prefix + "attention.head_count_kv").c_str(), 1);
    gguf_set_val_f32(
        context,
        (prefix + "attention.layer_norm_rms_epsilon").c_str(),
        1e-5f);
}

inline void set_string_array(
    gguf_context* context,
    const char* key,
    const std::vector<std::string>& values) {
    std::vector<const char*> pointers;
    pointers.reserve(values.size());
    for (const std::string& value : values) {
        pointers.push_back(value.c_str());
    }
    gguf_set_arr_str(context, key, pointers.data(), pointers.size());
}

inline SyntheticGGUFFile write_synthetic_gguf(
    const std::string& label,
    const SyntheticGGUFSpec& spec) {
    const ggml_init_params params = {
        /* mem_size   */ 1ull * 1024 * 1024,
        /* mem_buffer */ nullptr,
        /* no_alloc   */ false,
    };
    ggml_context* tensor_context = ggml_init(params);
    if (tensor_context == nullptr) {
        throw std::runtime_error("failed to create synthetic tensor context");
    }

    gguf_context* gguf_context_ptr = gguf_init_empty();
    if (gguf_context_ptr == nullptr) {
        ggml_free(tensor_context);
        throw std::runtime_error("failed to create synthetic GGUF context");
    }

    const std::string path = make_temp_path(label);
    try {
        set_architecture_metadata(gguf_context_ptr, spec.architecture);
        const std::string prefix = spec.architecture + ".";
        if (spec.include_embedding_length) {
            gguf_set_val_u32(gguf_context_ptr, (prefix + "embedding_length").c_str(), 8);
        }
        if (spec.include_conflicting_llama_metadata) {
            set_architecture_metadata(gguf_context_ptr, "llama");
            gguf_set_val_str(gguf_context_ptr, "general.architecture", spec.architecture.c_str());
            gguf_set_val_u32(gguf_context_ptr, "llama.embedding_length", 64);
        }

        if (!spec.tokenizer_model.empty()) {
            gguf_set_val_str(
                gguf_context_ptr,
                "tokenizer.ggml.model",
                spec.tokenizer_model.c_str());
        }
        if (!spec.tokenizer_pre.empty()) {
            gguf_set_val_str(
                gguf_context_ptr,
                "tokenizer.ggml.pre",
                spec.tokenizer_pre.c_str());
        }
        if (!spec.chat_template.empty()) {
            gguf_set_val_str(
                gguf_context_ptr,
                "tokenizer.chat_template",
                spec.chat_template.c_str());
        }
        if (!spec.tokens.empty()) {
            set_string_array(gguf_context_ptr, "tokenizer.ggml.tokens", spec.tokens);
        }
        if (!spec.token_types.empty()) {
            gguf_set_arr_data(
                gguf_context_ptr,
                "tokenizer.ggml.token_type",
                GGUF_TYPE_INT32,
                spec.token_types.data(),
                spec.token_types.size());
        }
        if (!spec.merges.empty()) {
            set_string_array(gguf_context_ptr, "tokenizer.ggml.merges", spec.merges);
        }
        if (spec.bos_token_id >= 0) {
            gguf_set_val_u32(
                gguf_context_ptr,
                "tokenizer.ggml.bos_token_id",
                static_cast<uint32_t>(spec.bos_token_id));
        }
        if (spec.eos_token_id >= 0) {
            gguf_set_val_u32(
                gguf_context_ptr,
                "tokenizer.ggml.eos_token_id",
                static_cast<uint32_t>(spec.eos_token_id));
        }
        if (spec.set_add_bos) {
            gguf_set_val_bool(
                gguf_context_ptr,
                "tokenizer.ggml.add_bos_token",
                spec.add_bos);
        }
        if (spec.set_add_eos) {
            gguf_set_val_bool(
                gguf_context_ptr,
                "tokenizer.ggml.add_eos_token",
                spec.add_eos);
        }

        ggml_tensor* base = ggml_new_tensor_2d(tensor_context, GGML_TYPE_F32, 4, 4);
        ggml_set_name(base, "test.weight");
        std::fill_n(static_cast<float*>(base->data), 16, 0.0f);
        gguf_add_tensor(gguf_context_ptr, base);

        if (spec.include_attention_biases) {
            for (const char* name : {
                     "blk.0.attn_q.bias",
                     "blk.0.attn_k.bias",
                     "blk.0.attn_v.bias",
                 }) {
                ggml_tensor* bias =
                    ggml_new_tensor_1d(tensor_context, GGML_TYPE_F32, 8);
                ggml_set_name(bias, name);
                std::fill_n(static_cast<float*>(bias->data), 8, 0.0f);
                gguf_add_tensor(gguf_context_ptr, bias);
            }
        }

        const size_t tensor_data_end =
            gguf_get_meta_size(gguf_context_ptr) + gguf_get_tensor_size(gguf_context_ptr, 0);
        if (!gguf_write_to_file(gguf_context_ptr, path.c_str(), false)) {
            throw std::runtime_error("failed to write synthetic GGUF");
        }

        gguf_free(gguf_context_ptr);
        ggml_free(tensor_context);
        return SyntheticGGUFFile(path, tensor_data_end);
    } catch (...) {
        gguf_free(gguf_context_ptr);
        ggml_free(tensor_context);
        std::remove(path.c_str());
        throw;
    }
}

inline void free_model(nano_llm_infer::Model* model) {
    if (model == nullptr) {
        return;
    }
    if (model->weights_buffer != nullptr) {
        ggml_backend_buffer_free(model->weights_buffer);
        model->weights_buffer = nullptr;
    }
    if (model->ctx != nullptr) {
        ggml_free(model->ctx);
        model->ctx = nullptr;
    }
    if (model->mmap_ptr != nullptr) {
        munmap(model->mmap_ptr, model->mmap_size);
        model->mmap_ptr = nullptr;
        model->mmap_size = 0;
    }
    model->tensors.clear();
    model->infos.clear();
}

}  // namespace gguf_test_util
