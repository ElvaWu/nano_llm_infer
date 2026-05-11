#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "gguf_loader.h"
#include "kv_cache.h"
#include "tokenizer.h"
#include "transformer.h"

namespace nano_llm_infer {

class GenerationSession;

// Model loading and graph execution options.
struct RuntimeModelConfig {
    std::string path = "model_file/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";
    // CPU threads used by ggml graph execution.
    int compute_threads = 4;
};

// Token limits enforced by the runtime.
struct RuntimeLimitsConfig {
    // Maximum prompt plus generated token context stored in KV cache.
    int context_tokens = 512;
    // Fallback generation cap when a request does not set max_tokens.
    int default_max_new_tokens = 600;
};

// ggml arena sizes for per-request compute graphs.
struct RuntimeMemoryConfig {
    // Arena size in bytes for prompt prefill graphs.
    size_t prefill_ctx_bytes = 512ull * 1024 * 1024;
    // Arena size in bytes for decode graphs.
    size_t decode_ctx_bytes = 256ull * 1024 * 1024;
};

struct RuntimeConfig {
    RuntimeModelConfig model;
    RuntimeLimitsConfig limits;
    RuntimeMemoryConfig memory;
};

struct GenerationOptions {
    int max_tokens = 0;
};

struct GenerationRequest {
    std::vector<ChatMessage> messages;
    GenerationOptions options;
};

struct GenerationEvent {
    std::string text;
    std::string finish_reason;
    std::string error;
};

struct DecodeBatchItem {
    GenerationSession* session = nullptr;
    const GenerationRequest* request = nullptr;
    GenerationEvent event;
    bool produced = false;
};

using GenerationCallback = std::function<bool(const GenerationEvent&)>;

class GenerationSession {
public:
    GenerationSession(KVCache kv, int eos_token);
    ~GenerationSession();

    GenerationSession(const GenerationSession&) = delete;
    GenerationSession& operator=(const GenerationSession&) = delete;
    GenerationSession(GenerationSession&&) noexcept;
    GenerationSession& operator=(GenerationSession&&) noexcept;

private:
    friend class ModelRuntime;

    KVCache kv_;
    int eos_token_ = -1;
    int next_token_ = -1;
    int emitted_tokens_ = 0;
    std::string response_;
    bool started_ = false;
    bool finished_ = false;
    std::string finish_reason_;
};

class ModelRuntime {
public:
    explicit ModelRuntime(RuntimeConfig config);
    ~ModelRuntime();

    ModelRuntime(const ModelRuntime&) = delete;
    ModelRuntime& operator=(const ModelRuntime&) = delete;

    const RuntimeConfig& config() const { return config_; }
    const std::string& model_id() const { return model_id_; }
    size_t prompt_token_count(const std::string& prompt) const;
    size_t request_prompt_token_count(const GenerationRequest& request) const;

    std::unique_ptr<GenerationSession> create_session() const;
    void prefill(GenerationSession* session, const GenerationRequest& request);
    bool decode_next(GenerationSession* session, const GenerationRequest& request, GenerationEvent* event);
    void decode_batch(std::vector<DecodeBatchItem>* batch);
    void generate_stream(const GenerationRequest& request, const GenerationCallback& callback);

private:
    RuntimeConfig config_;
    GGUFLoader loader_;
    Model model_;
    Tokenizer tokenizer_;
    std::unique_ptr<Transformer> transformer_;
    std::string model_id_;
};

std::string build_prompt(const Tokenizer& tokenizer, const std::vector<ChatMessage>& messages);
std::vector<int> encode_prompt(Tokenizer* tokenizer, const std::string& prompt);
bool trim_stop_sequence(std::string* response);
void ensure_valid_omp_threads(int fallback_threads);

}  // namespace nano_llm_infer
