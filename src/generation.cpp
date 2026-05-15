#include "generation.h"

#include <sys/mman.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <utility>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "graph_inputs.h"

#if defined(NANO_LLM_ENABLE_CUDA)
#include "ggml-cuda.h"
#endif

namespace nano_llm_infer {

namespace {

constexpr size_t kRuntimeGraphSize = GGML_DEFAULT_GRAPH_SIZE * 4;

ggml_tensor* embedding_lookup(
    ggml_context* compute_ctx,
    const Model& model,
    const std::vector<int>& tokens) {
    const int64_t n_tokens = static_cast<int64_t>(tokens.size());
    ggml_tensor* token_tensor = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_I32, n_tokens);
    std::vector<int32_t> data(static_cast<size_t>(n_tokens));
    for (int64_t i = 0; i < n_tokens; ++i) {
        data[static_cast<size_t>(i)] = tokens[static_cast<size_t>(i)];
    }
    set_graph_input_data(token_tensor, data.data(), data.size() * sizeof(int32_t));

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

ggml_context* create_no_alloc_context(size_t mem_size) {
    const ggml_init_params params = {
        /* mem_size   */ mem_size,
        /* mem_buffer */ nullptr,
        /* no_alloc   */ true,
    };
    return ggml_init(params);
}

void free_model(Model& model) {
    if (model.weights_buffer != nullptr) {
        ggml_backend_buffer_free(model.weights_buffer);
        model.weights_buffer = nullptr;
    }
    if (model.ctx != nullptr) {
        ggml_free(model.ctx);
        model.ctx = nullptr;
    }
    if (model.mmap_ptr != nullptr) {
        munmap(model.mmap_ptr, model.mmap_size);
        model.mmap_ptr = nullptr;
        model.mmap_size = 0;
    }
    model.tensors.clear();
    model.infos.clear();
}

void move_model_weights_to_backend(Model& model, ggml_backend_buffer_type_t buffer_type) {
    if (model.ctx == nullptr || buffer_type == nullptr || model.weights_buffer != nullptr) {
        return;
    }

    std::vector<std::pair<ggml_tensor*, const void*>> sources;
    sources.reserve(model.tensors.size());
    for (const TensorInfo& info : model.infos) {
        ggml_tensor* tensor = get_tensor_by_name(model, info.name);
        if (tensor == nullptr || tensor->data == nullptr) {
            throw std::runtime_error("model tensor missing data before backend allocation: " + info.name);
        }
        sources.push_back({tensor, tensor->data});
        tensor->data = nullptr;
        tensor->buffer = nullptr;
    }

    model.weights_buffer = ggml_backend_alloc_ctx_tensors_from_buft(model.ctx, buffer_type);
    if (model.weights_buffer == nullptr) {
        throw std::runtime_error("failed to allocate model weights backend buffer");
    }
    ggml_backend_buffer_set_usage(model.weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    for (const auto& source : sources) {
        ggml_tensor* tensor = source.first;
        ggml_backend_tensor_set(tensor, source.second, 0, ggml_nbytes(tensor));
    }
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
    other.kv_.buffer = nullptr;
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
    other.kv_.buffer = nullptr;
    other.kv_.layers.clear();
    other.kv_.n_filled = 0;
    return *this;
}

ModelRuntime::ModelRuntime(RuntimeConfig config)
    : config_(std::move(config)) {
    ensure_valid_omp_threads(config_.model.compute_threads);

#if defined(NANO_LLM_ENABLE_CUDA)
    cuda_backend_ = ggml_backend_cuda_init(0);
    if (cuda_backend_ != nullptr) {
        backends_.push_back(cuda_backend_);
        runtime_buffer_type_ = ggml_backend_cuda_buffer_type(0);
    }
#endif

    cpu_backend_ = ggml_backend_cpu_init();
    if (cpu_backend_ == nullptr) {
        throw std::runtime_error("failed to initialize ggml CPU backend");
    }
    ggml_backend_cpu_set_n_threads(cpu_backend_, config_.model.compute_threads);
    backends_.push_back(cpu_backend_);
    if (runtime_buffer_type_ == nullptr) {
        runtime_buffer_type_ = ggml_backend_get_default_buffer_type(cpu_backend_);
    }

    sched_ = ggml_backend_sched_new(
        backends_.data(),
        nullptr,
        static_cast<int>(backends_.size()),
        kRuntimeGraphSize,
        false,
        true);
    if (sched_ == nullptr) {
        throw std::runtime_error("failed to initialize ggml backend scheduler");
    }

    model_ = loader_.load_model(config_.model.path);
    move_model_weights_to_backend(model_, runtime_buffer_type_);
    if (!tokenizer_.load_from_gguf(loader_)) {
        throw std::runtime_error("failed to load tokenizer from GGUF metadata");
    }
    transformer_ = std::make_unique<Transformer>(model_);
    model_id_ = config_.model.path;
    std::cerr << "Using ggml backend: "
              << (cuda_backend_ != nullptr ? "CUDA0 + CPU fallback" : "CPU")
              << '\n';
}

ModelRuntime::~ModelRuntime() {
    free_kv_cache(decode_scratch_kv_);
    transformer_.reset();
    if (sched_ != nullptr) {
        ggml_backend_sched_free(sched_);
        sched_ = nullptr;
    }
    free_model(model_);
    if (cuda_backend_ != nullptr) {
        ggml_backend_free(cuda_backend_);
        cuda_backend_ = nullptr;
    }
    if (cpu_backend_ != nullptr) {
        ggml_backend_free(cpu_backend_);
        cpu_backend_ = nullptr;
    }
}

KVCache* ModelRuntime::decode_scratch_kv(int batch_size) {
    if (batch_size < 1) {
        throw std::invalid_argument("decode scratch batch size must be positive");
    }
    if (decode_scratch_batch_size_ < batch_size) {
        free_kv_cache(decode_scratch_kv_);
        decode_scratch_kv_ =
            make_kv_cache(model_, config_.limits.context_tokens, batch_size, runtime_buffer_type_);
        decode_scratch_batch_size_ = batch_size;
    }
    return &decode_scratch_kv_;
}

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
        make_kv_cache(model_, config_.limits.context_tokens, 1, runtime_buffer_type_),
        tokenizer_.eos());
}

void ModelRuntime::compute_graph(ggml_context* ctx, ggml_cgraph* graph) {
    (void)ctx;
    if (sched_ == nullptr) {
        ggml_graph_compute_with_ctx(ctx, graph, config_.model.compute_threads);
        return;
    }

    ggml_backend_sched_reset(sched_);
    if (!ggml_backend_sched_alloc_graph(sched_, graph)) {
        throw std::runtime_error("failed to allocate ggml backend graph");
    }
    for (const auto& input : graph_input_data) {
        auto* tensor = const_cast<ggml_tensor*>(input.first);
        ggml_backend_tensor_set(tensor, input.second.data(), 0, input.second.size());
    }

    const ggml_status status = ggml_backend_sched_graph_compute(sched_, graph);
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("ggml backend graph compute failed: ") +
                                 ggml_status_to_string(status));
    }
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

    clear_graph_input_data();
    ggml_context* ctx = create_no_alloc_context(config_.memory.prefill_ctx_bytes);
    if (ctx == nullptr) {
        throw std::runtime_error("failed to allocate prefill compute context");
    }
    transformer_->set_compute_ctx(ctx);

    ggml_tensor* embeds = embedding_lookup(ctx, model_, prompt_tokens);
    ggml_cgraph* graph = ggml_new_graph_custom(ctx, kRuntimeGraphSize, false);
    ggml_tensor* logits = transformer_->forward_logits(embeds, 0, &session->kv_);
    ggml_build_forward_expand(graph, logits);
    compute_graph(ctx, graph);

    if (logits == nullptr) {
        ggml_free(ctx);
        clear_graph_input_data();
        throw std::runtime_error("prefill produced invalid logits tensor");
    }

    const int64_t vocab_size = logits->ne[0];
    if (vocab_size <= 0) {
        ggml_free(ctx);
        clear_graph_input_data();
        throw std::runtime_error("prefill logits has invalid vocabulary dimension");
    }
    const int64_t prompt_len = static_cast<int64_t>(prompt_tokens.size());
    std::vector<float> logit_data(static_cast<size_t>(vocab_size) * prompt_tokens.size());
    ggml_backend_tensor_get(logits, logit_data.data(), 0, logit_data.size() * sizeof(float));
    session->next_token_ =
        select_argmax(logit_data.data() + (prompt_len - 1) * vocab_size, vocab_size);
    session->kv_.n_filled = static_cast<int>(prompt_tokens.size());
    session->started_ = true;

    ggml_free(ctx);
    clear_graph_input_data();
}

void ModelRuntime::prefill_batch(const std::vector<std::pair<GenerationSession*, const GenerationRequest*>>& batch) {
    for (const auto& item : batch) {
        if (item.first == nullptr || item.second == nullptr) {
            throw std::invalid_argument("prefill_batch item contains null session or request");
        }
        prefill(item.first, *item.second);
    }
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

    KVCache* packed = decode_scratch_kv(static_cast<int>(active_indices.size()));
    ggml_context* ctx = nullptr;
    try {
        const size_t packed_slot_stride = kv_slot_stride_bytes(*packed);
        // Pack each session's prefix cache into its own batch slot so one
        // batched decode step can reuse the already-prefilled history.
        for (size_t layer = 0; layer < packed->layers.size(); ++layer) {
            for (size_t slot = 0; slot < active_indices.size(); ++slot) {
                const GenerationSession* session = (*batch)[active_indices[slot]].session;
                const size_t used_bytes =
                    static_cast<size_t>(session->kv_.n_filled) * kv_pos_stride_bytes(session->kv_);

                std::vector<uint8_t> k_prefix(used_bytes);
                std::vector<uint8_t> v_prefix(used_bytes);
                ggml_backend_tensor_get(session->kv_.layers[layer].k, k_prefix.data(), 0, used_bytes);
                ggml_backend_tensor_get(session->kv_.layers[layer].v, v_prefix.data(), 0, used_bytes);
                ggml_backend_tensor_set(packed->layers[layer].k, k_prefix.data(), slot * packed_slot_stride, used_bytes);
                ggml_backend_tensor_set(packed->layers[layer].v, v_prefix.data(), slot * packed_slot_stride, used_bytes);
            }
        }

        clear_graph_input_data();
        ctx = create_no_alloc_context(config_.memory.decode_ctx_bytes);
        if (ctx == nullptr) {
            throw std::runtime_error("failed to allocate decode compute context");
        }
        transformer_->set_compute_ctx(ctx);

        ggml_tensor* embeds = embedding_lookup(ctx, model_, tokens);
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, kRuntimeGraphSize, false);
        ggml_tensor* logits = transformer_->forward_logits_batched(embeds, positions, packed);
        ggml_build_forward_expand(graph, logits);
        compute_graph(ctx, graph);

        if (logits == nullptr) {
            throw std::runtime_error("decode produced invalid logits tensor");
        }

        const int64_t vocab_size = logits->ne[0];
        if (vocab_size <= 0) {
            throw std::runtime_error("decode logits has invalid vocabulary dimension");
        }
        std::vector<float> logit_data(static_cast<size_t>(vocab_size) * active_indices.size());
        ggml_backend_tensor_get(logits, logit_data.data(), 0, logit_data.size() * sizeof(float));
        for (size_t slot = 0; slot < active_indices.size(); ++slot) {
            GenerationSession* session = (*batch)[active_indices[slot]].session;
            session->next_token_ = select_argmax(logit_data.data() + slot * vocab_size, vocab_size);
        }

        const size_t token_bytes = kv_pos_stride_bytes(*packed);
        // Copy back only the just-appended position; the older prefix tokens
        // were copied into `packed` from the session caches unchanged.
        for (size_t layer = 0; layer < packed->layers.size(); ++layer) {
            for (size_t slot = 0; slot < active_indices.size(); ++slot) {
                GenerationSession* session = (*batch)[active_indices[slot]].session;
                const size_t pos_offset = static_cast<size_t>(positions[slot]) * token_bytes;
                std::vector<uint8_t> k_token(token_bytes);
                std::vector<uint8_t> v_token(token_bytes);
                ggml_backend_tensor_get(
                    packed->layers[layer].k,
                    k_token.data(),
                    slot * packed_slot_stride + pos_offset,
                    token_bytes);
                ggml_backend_tensor_get(
                    packed->layers[layer].v,
                    v_token.data(),
                    slot * packed_slot_stride + pos_offset,
                    token_bytes);
                ggml_backend_tensor_set(session->kv_.layers[layer].k, k_token.data(), pos_offset, token_bytes);
                ggml_backend_tensor_set(session->kv_.layers[layer].v, v_token.data(), pos_offset, token_bytes);
            }
        }
        for (size_t slot = 0; slot < active_indices.size(); ++slot) {
            GenerationSession* session = (*batch)[active_indices[slot]].session;
            session->kv_.n_filled++;
        }
    } catch (...) {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
        clear_graph_input_data();
        throw;
    }

    ggml_free(ctx);
    clear_graph_input_data();
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
