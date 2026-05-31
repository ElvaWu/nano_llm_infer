#include "generation.h"

#include <sys/mman.h>

#include <cstdlib>
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

// Graph size for a single-request prefill or decode graph.
constexpr size_t kRuntimeGraphSize = GGML_DEFAULT_GRAPH_SIZE * 4;
// Maximum prefill batch size supported by the scheduler hash set.
// The scheduler is created with this multiplier so batched prefill graphs
// (which scale linearly with batch size) never exceed the hash set capacity.
constexpr size_t kMaxPrefillBatch = 16;
// Scheduler hash set must cover the largest graph we will ever submit.
constexpr size_t kSchedGraphSize = kRuntimeGraphSize * kMaxPrefillBatch;
// This tensor is intentionally kept on host for quantized GGUF models where
// CUDA GET_ROWS does not support the embedding type, for example Q4_K.
constexpr const char* kTokenEmbeddingName = "token_embd.weight";

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

    ggml_tensor* token_embeddings = get_tensor_by_name(model, kTokenEmbeddingName);
    // kTokenEmbeddingName is intentionally kept on the CPU/mmap buffer (not
    // uploaded to the CUDA backend in move_model_weights_to_backend). ggml_get_rows
    // runs on CPU; the scheduler transfers the result to the active backend.
    // Do not move this tensor to the GPU backend.
    return ggml_get_rows(compute_ctx, token_embeddings, token_tensor);
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
        // CPU embedding lookup would otherwise pull the whole quantized table
        // back from CUDA on every prefill/decode step.
        if (info.name == kTokenEmbeddingName) {
            continue;
        }
        sources.push_back({tensor, tensor->data});
        // ggml_backend_alloc_ctx_tensors_from_buft only allocates tensors with
        // data == nullptr, so we must null out the pointer before calling it.
        // The original pointer is saved in sources for upload and for restoring
        // on allocation failure.
        tensor->data = nullptr;
    }

    model.weights_buffer = ggml_backend_alloc_ctx_tensors_from_buft(model.ctx, buffer_type);
    if (model.weights_buffer == nullptr) {
        // Restore data pointers so the model remains usable on CPU.
        for (const auto& source : sources) {
            source.first->data = const_cast<void*>(source.second);
        }
        throw std::runtime_error("failed to allocate model weights backend buffer");
    }
    ggml_backend_buffer_set_usage(model.weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    for (const auto& source : sources) {
        ggml_backend_tensor_set(source.first, source.second, 0, ggml_nbytes(source.first));
    }
}

}  // namespace

GenerationSession::GenerationSession(int eos_token)
    : eos_token_(eos_token) {
}

GenerationSession::~GenerationSession() = default;

GenerationSession::GenerationSession(GenerationSession&& other) noexcept
    : kv_slot_(other.kv_slot_),
      kv_filled_(other.kv_filled_),
      eos_token_(other.eos_token_),
      next_token_(other.next_token_),
      emitted_tokens_(other.emitted_tokens_),
      response_(std::move(other.response_)),
      started_(other.started_),
      finished_(other.finished_),
      finish_reason_(std::move(other.finish_reason_)) {
    other.kv_slot_ = -1;
    other.kv_filled_ = 0;
}

GenerationSession& GenerationSession::operator=(GenerationSession&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    kv_slot_ = other.kv_slot_;
    kv_filled_ = other.kv_filled_;
    eos_token_ = other.eos_token_;
    next_token_ = other.next_token_;
    emitted_tokens_ = other.emitted_tokens_;
    response_ = std::move(other.response_);
    started_ = other.started_;
    finished_ = other.finished_;
    finish_reason_ = std::move(other.finish_reason_);
    other.kv_slot_ = -1;
    other.kv_filled_ = 0;
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
        kSchedGraphSize,
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
}

ModelRuntime::~ModelRuntime() {
    free_kv_cache(kv_arena_);
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

size_t ModelRuntime::prompt_token_count(const std::string& prompt) const {
    Tokenizer tokenizer = tokenizer_;
    return encode_prompt(&tokenizer, prompt).size();
}

Tokenizer ModelRuntime::tokenizer() const {
    return tokenizer_;
}

size_t ModelRuntime::request_prompt_token_count(const GenerationRequest& request) const {
    Tokenizer tokenizer = tokenizer_;
    return encode_prompt(&tokenizer, build_prompt(tokenizer, request.messages)).size();
}

std::unique_ptr<GenerationSession> ModelRuntime::create_session() const {
    return std::make_unique<GenerationSession>(tokenizer_.eos());
}

void ModelRuntime::ensure_kv_arena_capacity(int capacity) {
    if (capacity < 1) {
        throw std::invalid_argument("KV arena capacity must be positive");
    }
    if (kv_arena_capacity_ >= capacity) {
        return;
    }
    for (bool in_use : kv_slot_in_use_) {
        if (in_use) {
            throw std::runtime_error("cannot resize shared KV arena while slots are in use");
        }
    }

    free_kv_cache(kv_arena_);
    kv_arena_ = make_kv_cache(model_, config_.limits.context_tokens, capacity, runtime_buffer_type_);
    kv_arena_capacity_ = capacity;
    kv_slot_in_use_.assign(static_cast<size_t>(capacity), false);
}

void ModelRuntime::assign_kv_slot(GenerationSession* session) {
    if (session == nullptr) {
        throw std::invalid_argument("generation session is null");
    }
    if (session->kv_slot_ >= 0) {
        return;
    }
    if (kv_arena_capacity_ < 1) {
        ensure_kv_arena_capacity(1);
    }

    for (int slot = 0; slot < kv_arena_capacity_; ++slot) {
        if (kv_slot_in_use_[static_cast<size_t>(slot)]) {
            continue;
        }
        // A session owns exactly one arena slot while active. Reset the slot
        // before reuse so stale KV never leaks if future masks change.
        const size_t slot_offset = static_cast<size_t>(slot) * kv_slot_stride_bytes(kv_arena_);
        const size_t slot_bytes = kv_slot_stride_bytes(kv_arena_);
        for (KVCacheLayer& layer : kv_arena_.layers) {
            ggml_backend_tensor_memset(layer.k, 0, slot_offset, slot_bytes);
            ggml_backend_tensor_memset(layer.v, 0, slot_offset, slot_bytes);
        }
        kv_slot_in_use_[static_cast<size_t>(slot)] = true;
        session->kv_slot_ = slot;
        session->kv_filled_ = 0;
        session->next_token_ = -1;
        session->emitted_tokens_ = 0;
        session->response_.clear();
        session->started_ = false;
        session->finished_ = false;
        session->finish_reason_.clear();
        return;
    }

    throw std::runtime_error("shared KV arena has no free slots");
}

void ModelRuntime::release_kv_slot(GenerationSession* session) {
    if (session == nullptr || session->kv_slot_ < 0) {
        return;
    }
    const int slot = session->kv_slot_;
    if (slot >= kv_arena_capacity_) {
        throw std::runtime_error("session KV slot is outside shared arena capacity");
    }
    kv_slot_in_use_[static_cast<size_t>(slot)] = false;
    session->kv_slot_ = -1;
    session->kv_filled_ = 0;
}

void ModelRuntime::compute_graph(ggml_cgraph* graph) {
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
    assign_kv_slot(session);

    ggml_context* ctx = nullptr;
    try {
        const std::string prompt = build_prompt(tokenizer_, request.messages);
        std::vector<int> prompt_tokens = encode_prompt(&tokenizer_, prompt);
        if (prompt_tokens.empty()) {
            throw std::runtime_error("prompt encoded to zero tokens");
        }
        if (static_cast<int>(prompt_tokens.size()) >= config_.limits.context_tokens) {
            throw std::runtime_error("prompt exceeds max sequence length");
        }

        clear_graph_input_data();
        ctx = create_no_alloc_context(config_.memory.prefill_ctx_bytes);
        if (ctx == nullptr) {
            throw std::runtime_error("failed to allocate prefill compute context");
        }
        transformer_->set_compute_ctx(ctx);

        ggml_tensor* embeds = embedding_lookup(ctx, model_, prompt_tokens);
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, kRuntimeGraphSize, false);
        ggml_tensor* logits = transformer_->forward_logits(embeds, 0, session->kv_slot_, &kv_arena_);
        ggml_tensor* final_logits = nullptr;
        ggml_tensor* next_token = nullptr;
        if (logits != nullptr) {
            const int64_t vocab_size = logits->ne[0];
            const int64_t prompt_len = static_cast<int64_t>(prompt_tokens.size());
            final_logits = ggml_view_2d(
                ctx,
                logits,
                vocab_size,
                1,
                static_cast<size_t>(vocab_size) * sizeof(float),
                static_cast<size_t>(prompt_len - 1) * static_cast<size_t>(vocab_size) * sizeof(float));
            // Argmax on-device avoids copying the full final logits row to host.
            next_token = ggml_argmax(ctx, final_logits);
            ggml_build_forward_expand(graph, next_token);
        }
        compute_graph(graph);

        if (logits == nullptr || next_token == nullptr) {
            throw std::runtime_error("prefill produced invalid logits tensor");
        }

        const int64_t vocab_size = logits->ne[0];
        if (vocab_size <= 0) {
            throw std::runtime_error("prefill logits has invalid vocabulary dimension");
        }
        int32_t selected_token = -1;
        ggml_backend_tensor_get(next_token, &selected_token, 0, sizeof(selected_token));
        session->next_token_ = selected_token;
        session->kv_filled_ = static_cast<int>(prompt_tokens.size());
        session->started_ = true;
    } catch (...) {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
        clear_graph_input_data();
        release_kv_slot(session);
        throw;
    }

    ggml_free(ctx);
    clear_graph_input_data();
}

void ModelRuntime::prefill_batch(const std::vector<std::pair<GenerationSession*, const GenerationRequest*>>& batch) {
    for (const auto& item : batch) {
        if (item.first == nullptr || item.second == nullptr) {
            throw std::invalid_argument("prefill_batch item contains null session or request");
        }
    }
    if (batch.size() > kMaxPrefillBatch) {
        throw std::invalid_argument("prefill_batch size exceeds kMaxPrefillBatch scheduler limit");
    }
    if (batch.size() == 1) {
        prefill(batch[0].first, *batch[0].second);
        return;
    }
    if (kv_arena_capacity_ < static_cast<int>(batch.size())) {
        ensure_kv_arena_capacity(static_cast<int>(batch.size()));
    }

    // Encode all prompts. Variable-length prompts are fine — each request has
    // its own embedding tensor and KV cache, so the fused graph handles them
    // independently regardless of length.
    std::vector<std::vector<int>> all_tokens;
    all_tokens.reserve(batch.size());
    for (const auto& item : batch) {
        const std::string prompt = build_prompt(tokenizer_, item.second->messages);
        std::vector<int> tokens = encode_prompt(&tokenizer_, prompt);
        if (tokens.empty()) {
            throw std::runtime_error("prompt encoded to zero tokens");
        }
        if (static_cast<int>(tokens.size()) >= config_.limits.context_tokens) {
            throw std::runtime_error("prompt exceeds max sequence length");
        }
        all_tokens.push_back(std::move(tokens));
    }
    for (const auto& item : batch) {
        assign_kv_slot(item.first);
    }

    // Build one shared ggml context and graph. Each request's forward pass
    // appends ops to the same context; a single compute_graph call executes
    // all of them together, amortising scheduler reset and graph allocation.
    // Scale the graph node budget and context memory by batch size so large
    // batches don't overflow.
    const size_t batch_graph_size = kRuntimeGraphSize * batch.size();
    const size_t batch_ctx_bytes = config_.memory.prefill_ctx_bytes * batch.size();
    clear_graph_input_data();
    ggml_context* ctx = create_no_alloc_context(batch_ctx_bytes);
    if (ctx == nullptr) {
        throw std::runtime_error("failed to allocate batched prefill compute context");
    }
    transformer_->set_compute_ctx(ctx);
    ggml_cgraph* graph = ggml_new_graph_custom(ctx, batch_graph_size, false);

    std::vector<ggml_tensor*> next_token_tensors;
    next_token_tensors.reserve(batch.size());

    try {
        for (size_t i = 0; i < batch.size(); ++i) {
            GenerationSession* session = batch[i].first;
            const size_t prompt_len = all_tokens[i].size();
            ggml_tensor* embeds = embedding_lookup(ctx, model_, all_tokens[i]);
            ggml_tensor* logits =
                transformer_->forward_logits(embeds, 0, session->kv_slot_, &kv_arena_);
            if (logits == nullptr) {
                throw std::runtime_error("prefill produced null logits tensor");
            }
            const int64_t vocab_size = logits->ne[0];
            if (vocab_size <= 0) {
                throw std::runtime_error("prefill logits has invalid vocabulary dimension");
            }
            ggml_tensor* final_logits = ggml_view_2d(
                ctx,
                logits,
                vocab_size,
                1,
                static_cast<size_t>(vocab_size) * sizeof(float),
                static_cast<size_t>(prompt_len - 1) * static_cast<size_t>(vocab_size) * sizeof(float));
            ggml_tensor* next_token = ggml_argmax(ctx, final_logits);
            ggml_build_forward_expand(graph, next_token);
            next_token_tensors.push_back(next_token);
        }

        compute_graph(graph);

        for (size_t i = 0; i < batch.size(); ++i) {
            GenerationSession* session = batch[i].first;
            int32_t selected_token = -1;
            ggml_backend_tensor_get(next_token_tensors[i], &selected_token, 0, sizeof(selected_token));
            session->next_token_ = selected_token;
            session->kv_filled_ = static_cast<int>(all_tokens[i].size());
            session->started_ = true;
        }
    } catch (...) {
        ggml_free(ctx);
        clear_graph_input_data();
        throw;
    }

    ggml_free(ctx);
    clear_graph_input_data();
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
    if (kv_arena_capacity_ < static_cast<int>(batch->size())) {
        ensure_kv_arena_capacity(static_cast<int>(batch->size()));
    }

    // Collect only the sessions that still need a model forward pass. Sessions
    // that finish on bookkeeping alone emit their final event immediately.
    std::vector<size_t> active_indices;
    std::vector<int> tokens;
    std::vector<int> positions;
    std::vector<int> kv_slots;
    active_indices.reserve(batch->size());
    tokens.reserve(batch->size());
    positions.reserve(batch->size());
    kv_slots.reserve(batch->size());

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
            if (session->kv_slot_ < 0 || session->kv_slot_ >= kv_arena_capacity_) {
                throw std::runtime_error("session has no assigned KV arena slot");
            }
            if (session->kv_filled_ < 0 || session->kv_filled_ > config_.limits.context_tokens) {
                throw std::runtime_error("session KV fill level is out of bounds");
            }

            const bool eos_reached = session->next_token_ == session->eos_token_;
            if ((!item.request->options.ignore_eos && eos_reached) ||
                session->kv_filled_ >= config_.limits.context_tokens ||
                session->emitted_tokens_ >= max_tokens) {
                session->finished_ = true;
                session->finish_reason_ =
                    !item.request->options.ignore_eos && eos_reached ? "stop" : "length";
                item.event.finish_reason = session->finish_reason_;
                item.produced = true;
                continue;
            }

            const int token = session->next_token_;
            std::string decoded = tokenizer_.decode(token);
            const size_t response_before_decode = session->response_.size();
            session->response_ += decoded;
            session->emitted_tokens_++;
            item.event.token_generated = true;

            if (item.request->options.enable_stop_sequences && trim_stop_sequence(&session->response_)) {
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
            positions.push_back(session->kv_filled_);
            kv_slots.push_back(session->kv_slot_);
        } catch (const std::exception& e) {
            session->finished_ = true;
            session->finish_reason_.clear();
            item.event.error = e.what();
            item.produced = true;
            release_kv_slot(session);
        }
    }

    if (active_indices.empty()) {
        return;
    }

    ggml_context* ctx = nullptr;
    try {
        clear_graph_input_data();
        ctx = create_no_alloc_context(config_.memory.decode_ctx_bytes);
        if (ctx == nullptr) {
            throw std::runtime_error("failed to allocate decode compute context");
        }
        transformer_->set_compute_ctx(ctx);

        ggml_tensor* embeds = embedding_lookup(ctx, model_, tokens);
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, kRuntimeGraphSize * active_indices.size(), false);
        ggml_tensor* logits = transformer_->forward_logits_batched(embeds, positions, kv_slots, &kv_arena_);
        ggml_tensor* next_tokens = nullptr;
        if (logits != nullptr) {
            // Keep selection on-device; only the compact token id vector is
            // copied back below.
            next_tokens = ggml_argmax(ctx, logits);
            ggml_build_forward_expand(graph, next_tokens);
        }
        compute_graph(graph);

        if (logits == nullptr || next_tokens == nullptr) {
            throw std::runtime_error("decode produced invalid logits tensor");
        }

        const int64_t vocab_size = logits->ne[0];
        if (vocab_size <= 0) {
            throw std::runtime_error("decode logits has invalid vocabulary dimension");
        }
        std::vector<int32_t> next_token_data(active_indices.size());
        ggml_backend_tensor_get(
            next_tokens,
            next_token_data.data(),
            0,
            next_token_data.size() * sizeof(int32_t));
        for (size_t slot = 0; slot < active_indices.size(); ++slot) {
            GenerationSession* session = (*batch)[active_indices[slot]].session;
            session->next_token_ = next_token_data[slot];
        }

        // The decode graph wrote new K/V directly into each session's arena
        // slot, so completion only advances the logical fill counters.
        for (size_t slot = 0; slot < active_indices.size(); ++slot) {
            GenerationSession* session = (*batch)[active_indices[slot]].session;
            session->kv_filled_++;
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
    try {
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
    } catch (...) {
        release_kv_slot(session.get());
        throw;
    }
    release_kv_slot(session.get());
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
