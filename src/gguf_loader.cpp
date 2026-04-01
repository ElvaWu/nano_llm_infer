#include <iostream>
#include "gguf_loader.h"

namespace nano_llm_infer{

ggml_tensor* get_tensor_by_name(const Model& model, const std::string& name) {
    auto it = model.tensors.find(name);
    if (it == model.tensors.end()) {
    std::cout << "Tensor not found: " << name << std::endl;
        return nullptr;
    }
    return it->second;
}
// bool GGUFLoader::load(const std::string& filepath) {
    
//     return true;
// }
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

    // === 1. Load metadata; GGUF creates model.ctx with tensor stubs ===
    gguf_init_params params = {
        /* no_alloc */ true,   // create tensor descriptors without allocating weight data
        /* ctx */ &model.ctx   // GGUF will allocate and populate this ggml_context
    };

    gguf_context* meta = gguf_init_from_file(path.c_str(), params);

    // === 1b. Populate metadata_ from GGUF KV pairs ===
    metadata_.clear();

    int64_t n_kv = gguf_get_n_kv(meta);
    for (int64_t i = 0; i < n_kv; i++) {
        const char* key = gguf_get_key(meta, i);
        gguf_type kv_type = gguf_get_kv_type(meta, i);

        gguf_kv kv;
        if (kv_type == GGUF_TYPE_ARRAY) {
            gguf_type arr_type = gguf_get_arr_type(meta, i);
            size_t arr_n      = gguf_get_arr_n(meta, i);

            if (arr_type == GGUF_TYPE_STRING) {
                std::vector<std::string> v(arr_n);
                for (size_t j = 0; j < arr_n; j++)
                    v[j] = gguf_get_arr_str(meta, i, j);
                kv.value = GGUFArray{std::move(v)};
            } else {
                const void* raw = gguf_get_arr_data(meta, i);
                switch (arr_type) {
                    case GGUF_TYPE_UINT8: { auto* p = static_cast<const uint8_t*>(raw);  kv.value = GGUFArray{std::vector<uint8_t> (p, p+arr_n)}; break; }
                    case GGUF_TYPE_INT8:  { auto* p = static_cast<const int8_t*>(raw);   kv.value = GGUFArray{std::vector<int8_t>  (p, p+arr_n)}; break; }
                    case GGUF_TYPE_UINT16:{ auto* p = static_cast<const uint16_t*>(raw); kv.value = GGUFArray{std::vector<uint16_t>(p, p+arr_n)}; break; }
                    case GGUF_TYPE_INT16: { auto* p = static_cast<const int16_t*>(raw);  kv.value = GGUFArray{std::vector<int16_t> (p, p+arr_n)}; break; }
                    case GGUF_TYPE_UINT32:{ auto* p = static_cast<const uint32_t*>(raw); kv.value = GGUFArray{std::vector<uint32_t>(p, p+arr_n)}; break; }
                    case GGUF_TYPE_INT32: { auto* p = static_cast<const int32_t*>(raw);  kv.value = GGUFArray{std::vector<int32_t> (p, p+arr_n)}; break; }
                    case GGUF_TYPE_UINT64:{ auto* p = static_cast<const uint64_t*>(raw); kv.value = GGUFArray{std::vector<uint64_t>(p, p+arr_n)}; break; }
                    case GGUF_TYPE_INT64: { auto* p = static_cast<const int64_t*>(raw);  kv.value = GGUFArray{std::vector<int64_t> (p, p+arr_n)}; break; }
                    case GGUF_TYPE_FLOAT32:{ auto* p = static_cast<const float*>(raw);   kv.value = GGUFArray{std::vector<float>  (p, p+arr_n)}; break; }
                    case GGUF_TYPE_FLOAT64:{ auto* p = static_cast<const double*>(raw);  kv.value = GGUFArray{std::vector<double> (p, p+arr_n)}; break; }
                    case GGUF_TYPE_BOOL: {
                        auto* p = static_cast<const int8_t*>(raw);
                        std::vector<bool> v(arr_n);
                        for (size_t j = 0; j < arr_n; j++) v[j] = (p[j] != 0);
                        kv.value = GGUFArray{std::move(v)};
                        break;
                    }
                    default: break;
                }
            }
        } else {
            switch (kv_type) {
                case GGUF_TYPE_UINT8:   kv.value = gguf_get_val_u8  (meta, i); break;
                case GGUF_TYPE_INT8:    kv.value = gguf_get_val_i8  (meta, i); break;
                case GGUF_TYPE_UINT16:  kv.value = gguf_get_val_u16 (meta, i); break;
                case GGUF_TYPE_INT16:   kv.value = gguf_get_val_i16 (meta, i); break;
                case GGUF_TYPE_UINT32:  kv.value = gguf_get_val_u32 (meta, i); break;
                case GGUF_TYPE_INT32:   kv.value = gguf_get_val_i32 (meta, i); break;
                case GGUF_TYPE_FLOAT32: kv.value = gguf_get_val_f32 (meta, i); break;
                case GGUF_TYPE_UINT64:  kv.value = gguf_get_val_u64 (meta, i); break;
                case GGUF_TYPE_INT64:   kv.value = gguf_get_val_i64 (meta, i); break;
                case GGUF_TYPE_FLOAT64: kv.value = gguf_get_val_f64 (meta, i); break;
                case GGUF_TYPE_BOOL:    kv.value = gguf_get_val_bool(meta, i); break;
                case GGUF_TYPE_STRING:  kv.value = std::string(gguf_get_val_str(meta, i)); break;
                default: break;
            }
        }
        metadata_[key] = kv;
    }

    int n = gguf_get_n_tensors(meta);

    model.n_layers     = std::get<uint32_t>(metadata_.at("llama.block_count").value);
    model.n_heads      = std::get<uint32_t>(metadata_.at("llama.attention.head_count").value);
    model.n_kv_heads   = std::get<uint32_t>(metadata_.at("llama.attention.head_count_kv").value);
    model.rms_norm_eps = std::get<float>   (metadata_.at("llama.attention.layer_norm_rms_epsilon").value);

    // === 2. 收集 tensor 信息 (model.ctx is now valid) ===
    for (int i = 0; i < n; i++) {
        TensorInfo info;

        info.name = gguf_get_tensor_name(meta, i);

        std::cout << "tensor " << i << ": " << info.name << std::endl;

        // model.ctx is now initialized by gguf_init_from_file above
        struct ggml_tensor * cur = ggml_get_tensor(model.ctx, info.name.c_str());

        int nd = ggml_n_dims(cur);
        info.shape.resize(nd);

        // read shape directly from the tensor descriptor's ne[] array
        for (int d = 0; d < nd; d++) {
            info.shape[d] = cur->ne[d];
        }

        info.type = gguf_get_tensor_type(meta, i);
        info.offset = gguf_get_tensor_offset(meta, i);

        model.infos.push_back(info);
    }
    std::cout << std::endl;


    // Free the no-alloc context; we'll recreate it with actual storage below
    ggml_free(model.ctx);
    model.ctx = nullptr;

    // === 3. 创建 ggml context (with allocation for weights) ===
    ggml_init_params p = {
        .mem_size = 1024ull * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = false
    };

    model.ctx = ggml_init(p);

    // === 4. 创建 tensor ===
    for (auto& info : model.infos) {
        ggml_tensor* t = ggml_new_tensor(
            model.ctx,
            info.type,
            info.shape.size(),
            info.shape.data()
        );

        model.tensors[info.name] = t;
    }

    // === 5. 加载权重 ===
    FILE* f = fopen(path.c_str(), "rb");

    // tensor offsets are relative to the start of the data section
    size_t data_offset = gguf_get_data_offset(meta);

    for (auto& info : model.infos) {
        auto t = model.tensors[info.name];

        fseek(f, data_offset + info.offset, SEEK_SET);
        fread(t->data, ggml_nbytes(t), 1, f);
    }

    fclose(f);
    gguf_free(meta);

    return model;
}
}
