#pragma once
#include <string>
#include <variant>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include "gguf.h"

namespace nano_llm_infer{

using GGUFArray = std::variant<
    std::vector<uint8_t>,  std::vector<int8_t>,
    std::vector<uint16_t>, std::vector<int16_t>,
    std::vector<uint32_t>, std::vector<int32_t>,
    std::vector<uint64_t>, std::vector<int64_t>,
    std::vector<float>,    std::vector<double>,
    std::vector<bool>,     std::vector<std::string>
>;

using GGUFValue = std::variant<
    uint8_t, int8_t, uint16_t, int16_t,
    uint32_t, int32_t, uint64_t, int64_t,
    float, double, bool, std::string, GGUFArray
>;

struct gguf_kv {
    GGUFValue value;
};

struct TensorInfo {
    std::string name;
    std::vector<int64_t> shape;
    ggml_type type;
    size_t offset;
};

// struct AttentionWeights {
//     ggml_tensor* wq;
//     ggml_tensor* wk;
//     ggml_tensor* wv;
//     ggml_tensor* wo;
// };

// struct LayerWeights {
//     AttentionWeights attn;
//     // other weights
// };

struct Model {
    ggml_context* ctx;
    uint32_t n_layers = 0;
    uint32_t n_heads = 0;
    uint32_t n_kv_heads = 0;
    float rms_norm_eps = 1e-5f;
    std::vector<TensorInfo> infos;
    // std::vector<LayerWeights> layers;
    std::unordered_map<std::string, ggml_tensor*> tensors;
};

ggml_tensor* get_tensor_by_name(const Model& model, const std::string& name);

class GGUFLoader {
public:
    GGUFLoader() = default;
    ~GGUFLoader();

    void close();

    GGUFLoader(const GGUFLoader&) = delete;
    GGUFLoader& operator=(const GGUFLoader&) = delete;

    // GGUFLoader(GGUFLoader&& other) noexcept;
    // GGUFLoader& operator=(GGUFLoader&& other) noexcept;

    bool has_metadata(const std::string& key) const;

    template <typename T>
    T get_metadata(const std::string& key) const {
        auto it = metadata_.find(key);
        if (it == metadata_.end()) {
            throw std::out_of_range("metadata key not found: " + key);
        }
        const GGUFValue& value = it->second.value;
        if constexpr (std::disjunction_v<std::is_same<T, uint8_t>, std::is_same<T, int8_t>,
                                          std::is_same<T, uint16_t>, std::is_same<T, int16_t>,
                                          std::is_same<T, uint32_t>, std::is_same<T, int32_t>,
                                          std::is_same<T, uint64_t>, std::is_same<T, int64_t>,
                                          std::is_same<T, float>, std::is_same<T, double>,
                                          std::is_same<T, bool>, std::is_same<T, std::string>,
                                          std::is_same<T, GGUFArray>>) {
            if (const auto* exact = std::get_if<T>(&value)) {
                return *exact;
            }
        }
        if constexpr (std::disjunction_v<std::is_same<T, std::vector<uint8_t>>,
                                          std::is_same<T, std::vector<int8_t>>,
                                          std::is_same<T, std::vector<uint16_t>>,
                                          std::is_same<T, std::vector<int16_t>>,
                                          std::is_same<T, std::vector<uint32_t>>,
                                          std::is_same<T, std::vector<int32_t>>,
                                          std::is_same<T, std::vector<uint64_t>>,
                                          std::is_same<T, std::vector<int64_t>>,
                                          std::is_same<T, std::vector<float>>,
                                          std::is_same<T, std::vector<double>>,
                                          std::is_same<T, std::vector<bool>>,
                                          std::is_same<T, std::vector<std::string>>>) {
            if (const auto* arr = std::get_if<GGUFArray>(&value)) {
                if (const auto* arr_t = std::get_if<T>(arr)) {
                    return *arr_t;
                }
            }
        }
        throw std::bad_variant_access();
    }

    Model load_model(const std::string& path);


private:
    std::unordered_map<std::string, gguf_kv> metadata_;

};

} // namespace nano_llm_infer