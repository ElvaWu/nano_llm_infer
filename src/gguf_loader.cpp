#include "gguf_loader.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace nano_llm_infer {

namespace {

[[noreturn]] void throw_system_error(const std::string& action, const std::string& path) {
    throw std::runtime_error(action + " failed for '" + path + "': " + std::strerror(errno));
}

template <typename T>
T metadata_or(
    const std::unordered_map<std::string, gguf_kv>& metadata,
    const std::string& key,
    T fallback) {
    const auto it = metadata.find(key);
    if (it == metadata.end()) {
        return fallback;
    }
    if (const auto* value = std::get_if<T>(&it->second.value)) {
        return *value;
    }
    return fallback;
}

}  // namespace

ggml_tensor* get_tensor_by_name(const Model& model, const std::string& name) {
    const auto it = model.tensors.find(name);
    if (it == model.tensors.end()) {
        std::cerr << "Tensor not found: " << name << '\n';
        return nullptr;
    }
    return it->second;
}

GGUFLoader::~GGUFLoader() {
    close();
}

void GGUFLoader::close() {
    metadata_.clear();
}

bool GGUFLoader::has_metadata(const std::string& key) const {
    return metadata_.find(key) != metadata_.end();
}

Model GGUFLoader::load_model(const std::string& path) {
    Model model{};

    gguf_init_params params = {
        /* no_alloc */ true,
        /* ctx */ &model.ctx,
    };

    gguf_context* meta = gguf_init_from_file(path.c_str(), params);
    if (meta == nullptr || model.ctx == nullptr) {
        throw std::runtime_error("Failed to load GGUF metadata from: " + path);
    }

    metadata_.clear();

    const int64_t n_kv = gguf_get_n_kv(meta);
    for (int64_t i = 0; i < n_kv; ++i) {
        const char* key = gguf_get_key(meta, i);
        const gguf_type kv_type = gguf_get_kv_type(meta, i);

        gguf_kv kv;
        if (kv_type == GGUF_TYPE_ARRAY) {
            const gguf_type arr_type = gguf_get_arr_type(meta, i);
            const size_t arr_n = gguf_get_arr_n(meta, i);

            if (arr_type == GGUF_TYPE_STRING) {
                std::vector<std::string> values(arr_n);
                for (size_t j = 0; j < arr_n; ++j) {
                    values[j] = gguf_get_arr_str(meta, i, j);
                }
                kv.value = GGUFArray{std::move(values)};
            } else {
                const void* raw = gguf_get_arr_data(meta, i);
                switch (arr_type) {
                    case GGUF_TYPE_UINT8: {
                        const auto* p = static_cast<const uint8_t*>(raw);
                        kv.value = GGUFArray{std::vector<uint8_t>(p, p + arr_n)};
                        break;
                    }
                    case GGUF_TYPE_INT8: {
                        const auto* p = static_cast<const int8_t*>(raw);
                        kv.value = GGUFArray{std::vector<int8_t>(p, p + arr_n)};
                        break;
                    }
                    case GGUF_TYPE_UINT16: {
                        const auto* p = static_cast<const uint16_t*>(raw);
                        kv.value = GGUFArray{std::vector<uint16_t>(p, p + arr_n)};
                        break;
                    }
                    case GGUF_TYPE_INT16: {
                        const auto* p = static_cast<const int16_t*>(raw);
                        kv.value = GGUFArray{std::vector<int16_t>(p, p + arr_n)};
                        break;
                    }
                    case GGUF_TYPE_UINT32: {
                        const auto* p = static_cast<const uint32_t*>(raw);
                        kv.value = GGUFArray{std::vector<uint32_t>(p, p + arr_n)};
                        break;
                    }
                    case GGUF_TYPE_INT32: {
                        const auto* p = static_cast<const int32_t*>(raw);
                        kv.value = GGUFArray{std::vector<int32_t>(p, p + arr_n)};
                        break;
                    }
                    case GGUF_TYPE_UINT64: {
                        const auto* p = static_cast<const uint64_t*>(raw);
                        kv.value = GGUFArray{std::vector<uint64_t>(p, p + arr_n)};
                        break;
                    }
                    case GGUF_TYPE_INT64: {
                        const auto* p = static_cast<const int64_t*>(raw);
                        kv.value = GGUFArray{std::vector<int64_t>(p, p + arr_n)};
                        break;
                    }
                    case GGUF_TYPE_FLOAT32: {
                        const auto* p = static_cast<const float*>(raw);
                        kv.value = GGUFArray{std::vector<float>(p, p + arr_n)};
                        break;
                    }
                    case GGUF_TYPE_FLOAT64: {
                        const auto* p = static_cast<const double*>(raw);
                        kv.value = GGUFArray{std::vector<double>(p, p + arr_n)};
                        break;
                    }
                    case GGUF_TYPE_BOOL: {
                        const auto* p = static_cast<const int8_t*>(raw);
                        std::vector<bool> values(arr_n);
                        for (size_t j = 0; j < arr_n; ++j) {
                            values[j] = (p[j] != 0);
                        }
                        kv.value = GGUFArray{std::move(values)};
                        break;
                    }
                    default:
                        break;
                }
            }
        } else {
            switch (kv_type) {
                case GGUF_TYPE_UINT8:
                    kv.value = gguf_get_val_u8(meta, i);
                    break;
                case GGUF_TYPE_INT8:
                    kv.value = gguf_get_val_i8(meta, i);
                    break;
                case GGUF_TYPE_UINT16:
                    kv.value = gguf_get_val_u16(meta, i);
                    break;
                case GGUF_TYPE_INT16:
                    kv.value = gguf_get_val_i16(meta, i);
                    break;
                case GGUF_TYPE_UINT32:
                    kv.value = gguf_get_val_u32(meta, i);
                    break;
                case GGUF_TYPE_INT32:
                    kv.value = gguf_get_val_i32(meta, i);
                    break;
                case GGUF_TYPE_FLOAT32:
                    kv.value = gguf_get_val_f32(meta, i);
                    break;
                case GGUF_TYPE_UINT64:
                    kv.value = gguf_get_val_u64(meta, i);
                    break;
                case GGUF_TYPE_INT64:
                    kv.value = gguf_get_val_i64(meta, i);
                    break;
                case GGUF_TYPE_FLOAT64:
                    kv.value = gguf_get_val_f64(meta, i);
                    break;
                case GGUF_TYPE_BOOL:
                    kv.value = gguf_get_val_bool(meta, i);
                    break;
                case GGUF_TYPE_STRING:
                    kv.value = std::string(gguf_get_val_str(meta, i));
                    break;
                default:
                    break;
            }
        }

        metadata_[key] = std::move(kv);
    }

    const int n_tensors = gguf_get_n_tensors(meta);
    model.n_layers = std::get<uint32_t>(metadata_.at("llama.block_count").value);
    model.n_heads = std::get<uint32_t>(metadata_.at("llama.attention.head_count").value);
    model.n_kv_heads = std::get<uint32_t>(metadata_.at("llama.attention.head_count_kv").value);
    model.rms_norm_eps = std::get<float>(metadata_.at("llama.attention.layer_norm_rms_epsilon").value);
    model.n_ctx_train = metadata_or<uint32_t>(metadata_, "llama.context_length", 0);
    model.rope_dimension_count = metadata_or<uint32_t>(metadata_, "llama.rope.dimension_count", 0);
    model.rope_freq_base = metadata_or<float>(metadata_, "llama.rope.freq_base", 10000.0f);
    model.rope_freq_scale = metadata_or<float>(metadata_, "llama.rope.freq_scale", 1.0f);
    model.architecture = metadata_or<std::string>(metadata_, "general.architecture", "");
    model.rope_mode = model.architecture == "gptneox" ? GGML_ROPE_TYPE_NEOX : GGML_ROPE_TYPE_NORMAL;

    for (int i = 0; i < n_tensors; ++i) {
        TensorInfo info;
        info.name = gguf_get_tensor_name(meta, i);

        ggml_tensor* cur = ggml_get_tensor(model.ctx, info.name.c_str());
        const int nd = ggml_n_dims(cur);
        info.shape.resize(static_cast<size_t>(nd));
        for (int d = 0; d < nd; ++d) {
            info.shape[static_cast<size_t>(d)] = cur->ne[d];
        }

        info.type = gguf_get_tensor_type(meta, i);
        info.offset = gguf_get_tensor_offset(meta, i);
        model.infos.push_back(std::move(info));
    }

    ggml_free(model.ctx);
    model.ctx = nullptr;

    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        gguf_free(meta);
        throw_system_error("open", path);
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        ::close(fd);
        gguf_free(meta);
        throw_system_error("fstat", path);
    }

    model.mmap_size = static_cast<size_t>(st.st_size);
    model.mmap_ptr = mmap(nullptr, model.mmap_size, PROT_READ, MAP_SHARED, fd, 0);
    const int saved_errno = errno;
    ::close(fd);
    if (model.mmap_ptr == MAP_FAILED) {
        gguf_free(meta);
        errno = saved_errno;
        throw_system_error("mmap", path);
    }

    (void)madvise(model.mmap_ptr, model.mmap_size, MADV_SEQUENTIAL);

    const size_t data_offset = gguf_get_data_offset(meta);
    const ggml_init_params init_params = {
        /* mem_size   */ 10ull * 1024 * 1024,
        /* mem_buffer */ nullptr,
        /* no_alloc   */ true,
    };
    model.ctx = ggml_init(init_params);
    if (model.ctx == nullptr) {
        munmap(model.mmap_ptr, model.mmap_size);
        model.mmap_ptr = nullptr;
        model.mmap_size = 0;
        gguf_free(meta);
        throw std::runtime_error("Failed to create GGML tensor metadata context");
    }

    for (auto& info : model.infos) {
        ggml_tensor* tensor = ggml_new_tensor(
            model.ctx,
            info.type,
            static_cast<int>(info.shape.size()),
            info.shape.data());
        tensor->data = static_cast<char*>(model.mmap_ptr) + data_offset + info.offset;
        model.tensors[info.name] = tensor;
    }

    gguf_free(meta);
    return model;
}

}  // namespace nano_llm_infer
