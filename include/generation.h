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

struct RuntimeConfig {
    std::string model_path = "model_file/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";
    int compute_threads = 4;
    int max_seq_len = 512;
    int max_new_tokens = 600;
    size_t prefill_ctx_bytes = 512ull * 1024 * 1024;
    size_t decode_ctx_bytes = 256ull * 1024 * 1024;
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

    std::unique_ptr<GenerationSession> create_session() const;
    void prefill(GenerationSession* session, const GenerationRequest& request);
    bool decode_next(GenerationSession* session, const GenerationRequest& request, GenerationEvent* event);
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
