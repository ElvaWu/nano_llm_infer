#include "generation.h"

#include <algorithm>
#include <cstring>
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
        return std::min(request.options.max_tokens, config.limits.default_max_new_tokens);
    }
    return config.limits.default_max_new_tokens;
}

size_t kv_slot_stride_bytes(const KVCache& kv) {
    return static_cast<size_t>(kv.head_dim) * kv.n_kv_heads * kv.max_seq_len * sizeof(float);
}

size_t kv_pos_stride_bytes(const KVCache& kv) {
    return static_cast<size_t>(kv.head_dim) * kv.n_kv_heads * sizeof(float);
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
    ensure_valid_omp_threads(config_.model.compute_threads);
    model_ = loader_.load_model(config_.model.path);
    if (!tokenizer_.load_from_gguf(loader_)) {
        throw std::runtime_error("failed to load tokenizer from GGUF metadata");
    }
    transformer_ = std::make_unique<Transformer>(model_);
    model_id_ = config_.model.path;
}

ModelRuntime::~ModelRuntime() = default;

size_t ModelRuntime::prompt_token_count(const std::string& prompt) const {
    Tokenizer tokenizer = tokenizer_;
    return encode_prompt(&tokenizer, prompt).size();
}

size_t ModelRuntime::request_prompt_token_count(const GenerationRequest& request) const {
    Tokenizer tokenizer = tokenizer_;
    return encode_prompt(&tokenizer, build_prompt(tokenizer, request.messages)).size();
}

std::unique_ptr<GenerationSession> ModelRuntime::create_session() const {
    return std::make_unique<GenerationSession>(
        make_kv_cache(model_, config_.limits.context_tokens),
        tokenizer_.eos());
}

void ModelRuntime::prefill(GenerationSession* session, const GenerationRequest& request) {
    if (session == nullptr) {
        throw std::invalid_argument("generation session is null");
    }
    if (request.messages.empty()) {
        throw std::invalid_argument("generation request must include at least one message");
    }
    if (!transformer_) {
        throw std::runtime_error("transformer runtime is not initialized");
    }

    const std::string prompt = build_prompt(tokenizer_, request.messages);
    std::vector<int> prompt_tokens = encode_prompt(&tokenizer_, prompt);
    if (prompt_tokens.empty()) {
        throw std::runtime_error("prompt encoded to zero tokens");
    }
    if (static_cast<int>(prompt_tokens.size()) >= config_.limits.context_tokens) {
        throw std::runtime_error("prompt exceeds max sequence length");
    }

    ggml_context* ctx = create_context(config_.memory.prefill_ctx_bytes);
    if (ctx == nullptr) {
        throw std::runtime_error("failed to allocate prefill compute context");
    }
    transformer_->set_compute_ctx(ctx);

    ggml_tensor* embeds = embedding_lookup(ctx, model_, prompt_tokens);
    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_tensor* logits = transformer_->forward_logits(embeds, 0, &session->kv_);
    ggml_build_forward_expand(graph, logits);
    ggml_graph_compute_with_ctx(ctx, graph, config_.model.compute_threads);

    if (logits == nullptr || logits->data == nullptr) {
        ggml_free(ctx);
        throw std::runtime_error("prefill produced invalid logits tensor");
    }

    auto* logit_data = static_cast<float*>(logits->data);
    const int64_t vocab_size = logits->ne[0];
    if (vocab_size <= 0) {
        ggml_free(ctx);
        throw std::runtime_error("prefill logits has invalid vocabulary dimension");
    }
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
    std::vector<DecodeBatchItem> batch(1);
    batch[0].session = session;
    batch[0].request = &request;
    decode_batch(&batch);
    *event = std::move(batch[0].event);
    return batch[0].produced;
}

void ModelRuntime::decode_batch(std::vector<DecodeBatchItem>* batch) {
    if (batch == nullptr) {
        throw std::invalid_argument("decode_batch received null batch");
    }
    if (!transformer_) {
        throw std::runtime_error("transformer runtime is not initialized");
    }

    // Collect only the sessions that still need a model forward pass. Sessions
    // that finish on bookkeeping alone emit their final event immediately.
    std::vector<size_t> active_indices;
    std::vector<int> tokens;
    std::vector<int> positions;
    active_indices.reserve(batch->size());
    tokens.reserve(batch->size());
    positions.reserve(batch->size());

    for (size_t i = 0; i < batch->size(); ++i) {
        DecodeBatchItem& item = (*batch)[i];
        item.event = GenerationEvent{};
        item.produced = false;

        if (item.session == nullptr || item.request == nullptr) {
            throw std::invalid_argument("decode_batch item contains null session or request");
        }

        GenerationSession* session = item.session;

        try {
            if (!session->started_) {
                prefill(session, *item.request);
            }
            if (session->finished_) {
                continue;
            }

            const int max_tokens = effective_max_tokens(config_, *item.request);
            if (max_tokens <= 0) {
                throw std::invalid_argument("max tokens must be positive");
            }
            if (session->kv_.n_filled < 0 || session->kv_.n_filled > config_.limits.context_tokens) {
                throw std::runtime_error("session KV fill level is out of bounds");
            }

            if (session->next_token_ == session->eos_token_ ||
                session->kv_.n_filled >= config_.limits.context_tokens ||
                session->emitted_tokens_ >= max_tokens) {
                session->finished_ = true;
                session->finish_reason_ =
                    session->next_token_ == session->eos_token_ ? "stop" : "length";
                item.event.finish_reason = session->finish_reason_;
                item.produced = true;
                continue;
            }

            const int token = session->next_token_;
            std::string decoded = tokenizer_.decode(token);
            const size_t response_before_decode = session->response_.size();
            session->response_ += decoded;
            session->emitted_tokens_++;

            if (trim_stop_sequence(&session->response_)) {
                session->finished_ = true;
                session->finish_reason_ = "stop";
                item.event.finish_reason = session->finish_reason_;
                const size_t emitted_length = session->response_.size() - response_before_decode;
                item.event.text = decoded.substr(0, emitted_length);
                item.produced = true;
                continue;
            }

            item.event.text = decoded;
            item.produced = true;
            active_indices.push_back(i);
            tokens.push_back(token);
            positions.push_back(session->kv_.n_filled);
        } catch (const std::exception& e) {
            session->finished_ = true;
            session->finish_reason_.clear();
            item.event.error = e.what();
            item.produced = true;
        }
    }

    if (active_indices.empty()) {
        return;
    }

    KVCache packed = make_kv_cache(model_, config_.limits.context_tokens, static_cast<int>(active_indices.size()));
    ggml_context* ctx = nullptr;
    try {
        const size_t packed_slot_stride = kv_slot_stride_bytes(packed);
        // Pack each session's prefix cache into its own batch slot so one
        // batched decode step can reuse the already-prefilled history.
        for (size_t layer = 0; layer < packed.layers.size(); ++layer) {
            for (size_t slot = 0; slot < active_indices.size(); ++slot) {
                const GenerationSession* session = (*batch)[active_indices[slot]].session;
                const size_t used_bytes =
                    static_cast<size_t>(session->kv_.n_filled) * kv_pos_stride_bytes(session->kv_);
                if (used_bytes == 0) {
                    continue;
                }

                std::memcpy(
                    static_cast<char*>(packed.layers[layer].k->data) + slot * packed_slot_stride,
                    session->kv_.layers[layer].k->data,
                    used_bytes);
                std::memcpy(
                    static_cast<char*>(packed.layers[layer].v->data) + slot * packed_slot_stride,
                    session->kv_.layers[layer].v->data,
                    used_bytes);
            }
        }

        ctx = create_context(config_.memory.decode_ctx_bytes);
        if (ctx == nullptr) {
            throw std::runtime_error("failed to allocate decode compute context");
        }
        transformer_->set_compute_ctx(ctx);

        ggml_tensor* embeds = embedding_lookup(ctx, model_, tokens);
        ggml_cgraph* graph = ggml_new_graph(ctx);
        ggml_tensor* logits = transformer_->forward_logits_batched(embeds, positions, &packed);
        ggml_build_forward_expand(graph, logits);
        ggml_graph_compute_with_ctx(ctx, graph, config_.model.compute_threads);

        if (logits == nullptr || logits->data == nullptr) {
            throw std::runtime_error("decode produced invalid logits tensor");
        }

        const auto* logit_data = static_cast<const float*>(logits->data);
        const int64_t vocab_size = logits->ne[0];
        if (vocab_size <= 0) {
            throw std::runtime_error("decode logits has invalid vocabulary dimension");
        }
        for (size_t slot = 0; slot < active_indices.size(); ++slot) {
            GenerationSession* session = (*batch)[active_indices[slot]].session;
            session->next_token_ = select_argmax(logit_data + slot * vocab_size, vocab_size);
        }

        const size_t token_bytes = kv_pos_stride_bytes(packed);
        // Copy back only the just-appended position; the older prefix tokens
        // were copied into `packed` from the session caches unchanged.
        for (size_t layer = 0; layer < packed.layers.size(); ++layer) {
            for (size_t slot = 0; slot < active_indices.size(); ++slot) {
                GenerationSession* session = (*batch)[active_indices[slot]].session;
                const size_t pos_offset = static_cast<size_t>(positions[slot]) * token_bytes;
                std::memcpy(
                    static_cast<char*>(session->kv_.layers[layer].k->data) + pos_offset,
                    static_cast<const char*>(packed.layers[layer].k->data) + slot * packed_slot_stride + pos_offset,
                    token_bytes);
                std::memcpy(
                    static_cast<char*>(session->kv_.layers[layer].v->data) + pos_offset,
                    static_cast<const char*>(packed.layers[layer].v->data) + slot * packed_slot_stride + pos_offset,
                    token_bytes);
            }
        }
        for (size_t slot = 0; slot < active_indices.size(); ++slot) {
            (*batch)[active_indices[slot]].session->kv_.n_filled++;
        }
    } catch (...) {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
        free_kv_cache(packed);
        throw;
    }

    ggml_free(ctx);
    free_kv_cache(packed);
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
