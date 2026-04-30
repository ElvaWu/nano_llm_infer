#include "generation.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <utility>

#include "ggml-cpu.h"

namespace nano_llm_infer {

namespace {

ggml_context* create_context(size_t mem_size) {
    const ggml_init_params params = {
        /* mem_size   */ mem_size,
        /* mem_buffer */ nullptr,
        /* no_alloc   */ false,
    };
    return ggml_init(params);
}

ggml_tensor* embedding_lookup(
    ggml_context* compute_ctx,
    const Model& model,
    const std::vector<int>& tokens) {
    const int64_t n_tokens = static_cast<int64_t>(tokens.size());
    ggml_tensor* token_tensor = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_I32, n_tokens);
    auto* data = static_cast<int32_t*>(token_tensor->data);
    for (int64_t i = 0; i < n_tokens; ++i) {
        data[i] = tokens[static_cast<size_t>(i)];
    }

    ggml_tensor* token_embeddings = get_tensor_by_name(model, "token_embd.weight");
    return ggml_get_rows(compute_ctx, token_embeddings, token_tensor);
}

int select_argmax(const float* logits, int64_t vocab_size) {
    return static_cast<int>(std::max_element(logits, logits + vocab_size) - logits);
}

int effective_max_tokens(const RuntimeConfig& config, const GenerationRequest& request) {
    if (request.options.max_tokens > 0) {
        return std::min(request.options.max_tokens, config.max_new_tokens);
    }
    return config.max_new_tokens;
}

}  // namespace

GenerationSession::GenerationSession(KVCache kv, int eos_token)
    : kv_(std::move(kv)),
      eos_token_(eos_token) {
}

GenerationSession::~GenerationSession() {
    free_kv_cache(kv_);
}

GenerationSession::GenerationSession(GenerationSession&& other) noexcept
    : kv_(std::move(other.kv_)),
      eos_token_(other.eos_token_),
      next_token_(other.next_token_),
      emitted_tokens_(other.emitted_tokens_),
      response_(std::move(other.response_)),
      started_(other.started_),
      finished_(other.finished_),
      finish_reason_(std::move(other.finish_reason_)) {
    other.kv_.ctx = nullptr;
    other.kv_.layers.clear();
    other.kv_.n_filled = 0;
}

GenerationSession& GenerationSession::operator=(GenerationSession&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    free_kv_cache(kv_);
    kv_ = std::move(other.kv_);
    eos_token_ = other.eos_token_;
    next_token_ = other.next_token_;
    emitted_tokens_ = other.emitted_tokens_;
    response_ = std::move(other.response_);
    started_ = other.started_;
    finished_ = other.finished_;
    finish_reason_ = std::move(other.finish_reason_);
    other.kv_.ctx = nullptr;
    other.kv_.layers.clear();
    other.kv_.n_filled = 0;
    return *this;
}

ModelRuntime::ModelRuntime(RuntimeConfig config)
    : config_(std::move(config)) {
    ensure_valid_omp_threads(config_.compute_threads);
    model_ = loader_.load_model(config_.model_path);
    if (!tokenizer_.load_from_gguf(loader_)) {
        throw std::runtime_error("failed to load tokenizer from GGUF metadata");
    }
    transformer_ = std::make_unique<Transformer>(model_);
    model_id_ = config_.model_path;
}

ModelRuntime::~ModelRuntime() = default;

std::unique_ptr<GenerationSession> ModelRuntime::create_session() const {
    return std::make_unique<GenerationSession>(
        make_kv_cache(model_, config_.max_seq_len),
        tokenizer_.eos());
}

void ModelRuntime::prefill(GenerationSession* session, const GenerationRequest& request) {
    if (session == nullptr) {
        throw std::invalid_argument("generation session is null");
    }
    if (request.messages.empty()) {
        throw std::invalid_argument("generation request must include at least one message");
    }

    const std::string prompt = build_prompt(tokenizer_, request.messages);
    std::vector<int> prompt_tokens = encode_prompt(&tokenizer_, prompt);
    if (prompt_tokens.empty()) {
        throw std::runtime_error("prompt encoded to zero tokens");
    }
    if (static_cast<int>(prompt_tokens.size()) >= config_.max_seq_len) {
        throw std::runtime_error("prompt exceeds max sequence length");
    }

    ggml_context* ctx = create_context(config_.prefill_ctx_bytes);
    transformer_->set_compute_ctx(ctx);

    ggml_tensor* embeds = embedding_lookup(ctx, model_, prompt_tokens);
    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_tensor* logits = transformer_->forward_logits(embeds, 0, &session->kv_);
    ggml_build_forward_expand(graph, logits);
    ggml_graph_compute_with_ctx(ctx, graph, config_.compute_threads);

    auto* logit_data = static_cast<float*>(logits->data);
    const int64_t vocab_size = logits->ne[0];
    const int64_t prompt_len = static_cast<int64_t>(prompt_tokens.size());
    const float* last_logits = logit_data + (prompt_len - 1) * vocab_size;
    session->next_token_ = select_argmax(last_logits, vocab_size);
    session->kv_.n_filled = static_cast<int>(prompt_tokens.size());
    session->started_ = true;

    ggml_free(ctx);
}

bool ModelRuntime::decode_next(
    GenerationSession* session,
    const GenerationRequest& request,
    GenerationEvent* event) {
    if (session == nullptr || event == nullptr) {
        throw std::invalid_argument("decode_next received null argument");
    }
    if (!session->started_) {
        prefill(session, request);
    }
    if (session->finished_) {
        return false;
    }

    if (session->next_token_ == session->eos_token_ ||
        session->kv_.n_filled >= config_.max_seq_len ||
        session->emitted_tokens_ >= effective_max_tokens(config_, request)) {
        session->finished_ = true;
        session->finish_reason_ = session->next_token_ == session->eos_token_ ? "stop" : "length";
        event->finish_reason = session->finish_reason_;
        return true;
    }

    const int token = session->next_token_;
    std::string decoded = tokenizer_.decode(token);
    // Snapshot response size before appending decoded text so we can compute
    // the exact text emitted by this step after optional stop-sequence trimming.
    const size_t response_before_decode = session->response_.size();
    session->response_ += decoded;
    session->emitted_tokens_++;

    if (trim_stop_sequence(&session->response_)) {
        session->finished_ = true;
        session->finish_reason_ = "stop";
        event->finish_reason = session->finish_reason_;
        // Trimming can remove a suffix from `decoded`; emit only the surviving prefix.
        const size_t emitted_length = session->response_.size() - response_before_decode;
        event->text = decoded.substr(0, emitted_length);
        return true;
    }

    event->text = decoded;

    ggml_context* ctx = create_context(config_.decode_ctx_bytes);
    transformer_->set_compute_ctx(ctx);

    ggml_tensor* embed = embedding_lookup(ctx, model_, {token});
    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_tensor* logits = transformer_->forward_logits(embed, session->kv_.n_filled, &session->kv_);
    ggml_build_forward_expand(graph, logits);
    ggml_graph_compute_with_ctx(ctx, graph, config_.compute_threads);

    auto* logit_data = static_cast<float*>(logits->data);
    const int64_t vocab_size = logits->ne[0];
    session->next_token_ = select_argmax(logit_data, vocab_size);
    session->kv_.n_filled++;

    ggml_free(ctx);
    return true;
}

void ModelRuntime::generate_stream(
    const GenerationRequest& request,
    const GenerationCallback& callback) {
    std::unique_ptr<GenerationSession> session = create_session();
    while (true) {
        GenerationEvent event;
        if (!decode_next(session.get(), request, &event)) {
            break;
        }
        if (!callback(event)) {
            break;
        }
        if (!event.finish_reason.empty() || !event.error.empty()) {
            break;
        }
    }
}

std::string build_prompt(const Tokenizer& tokenizer, const std::vector<ChatMessage>& messages) {
    return tokenizer.apply_chat_template(messages, true);
}

std::vector<int> encode_prompt(Tokenizer* tokenizer, const std::string& prompt) {
    std::vector<int> prompt_tokens = tokenizer->encode(prompt);
    prompt_tokens.insert(prompt_tokens.begin(), tokenizer->bos());
    return prompt_tokens;
}

bool trim_stop_sequence(std::string* response) {
    const std::vector<std::string> stop_strings = {
        "<|assistant|>", "<|user|>", "<|system|>",
        "|assistant|",   "|user|",   "|system|",
    };
    for (const auto& stop : stop_strings) {
        if (response->size() >= stop.size() &&
            response->compare(response->size() - stop.size(), stop.size(), stop) == 0) {
            response->resize(response->size() - stop.size());
            return true;
        }
    }
    return false;
}

void ensure_valid_omp_threads(int fallback_threads) {
    const char* omp_env = std::getenv("OMP_NUM_THREADS");
    if (omp_env == nullptr || std::string(omp_env).empty() || std::atoi(omp_env) < 1) {
        setenv("OMP_NUM_THREADS", std::to_string(fallback_threads).c_str(), 1);
    }
}

}  // namespace nano_llm_infer
