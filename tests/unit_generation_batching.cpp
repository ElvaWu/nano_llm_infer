#include <atomic>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "generation.h"
#include "generation_server.h"

#include "test_util.h"

#ifndef NANO_LLM_TEST_MODEL_PATH
#define NANO_LLM_TEST_MODEL_PATH "/root/autodl-tmp/repos/nano_llm_infer/model_file/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
#endif

using namespace nano_llm_infer;

namespace {

struct ObservedEvent {
    int sequence = 0;
    GenerationEvent event;
};

GenerationRequest make_request(const std::string& prompt, int max_tokens) {
    GenerationRequest request;
    request.messages.push_back({"user", prompt});
    request.options.max_tokens = max_tokens;
    return request;
}

std::vector<ObservedEvent> collect_stream(
    const std::shared_ptr<GenerationStream>& stream,
    std::atomic<int>* sequence_counter) {
    std::vector<ObservedEvent> events;
    while (true) {
        GenerationEvent event;
        if (!stream->wait_next(&event)) {
            break;
        }
        events.push_back(ObservedEvent{++(*sequence_counter), std::move(event)});
    }
    return events;
}

bool has_finish_event(const std::vector<ObservedEvent>& events) {
    for (const auto& observed : events) {
        if (!observed.event.finish_reason.empty()) {
            return true;
        }
    }
    return false;
}

bool has_error_event(const std::vector<ObservedEvent>& events) {
    for (const auto& observed : events) {
        if (!observed.event.error.empty()) {
            return true;
        }
    }
    return false;
}

int finish_sequence(const std::vector<ObservedEvent>& events) {
    for (const auto& observed : events) {
        if (!observed.event.finish_reason.empty()) {
            return observed.sequence;
        }
    }
    return -1;
}

int first_sequence(const std::vector<ObservedEvent>& events) {
    if (events.empty()) {
        return -1;
    }
    return events.front().sequence;
}

std::string collected_text(const std::vector<ObservedEvent>& events) {
    std::string text;
    for (const auto& observed : events) {
        text += observed.event.text;
    }
    return text;
}

std::string escape_for_debug(const std::string& text) {
    std::string escaped;
    escaped.reserve(text.size());
    for (char ch : text) {
        switch (ch) {
            case '\n':
                escaped += "\\n";
                break;
            case '\r':
                escaped += "\\r";
                break;
            case '\t':
                escaped += "\\t";
                break;
            default:
                escaped.push_back(ch);
                break;
        }
    }
    return escaped;
}

ModelRuntime* make_runtime_or_skip(std::unique_ptr<ModelRuntime>* runtime) {
    const std::string model_path = NANO_LLM_TEST_MODEL_PATH;
    if (model_path.empty()) {
        std::cerr << "[SKIP] unit_generation_batching: NANO_LLM_TEST_MODEL is empty\n";
        return nullptr;
    }

    try {
        RuntimeConfig config;
        config.model_path = model_path;
        config.compute_threads = 1;
        config.max_seq_len = 256;
        config.max_new_tokens = 16;
        config.prefill_ctx_bytes = 256ull * 1024 * 1024;
        config.decode_ctx_bytes = 128ull * 1024 * 1024;
        *runtime = std::make_unique<ModelRuntime>(config);
        return runtime->get();
    } catch (const std::exception& e) {
        std::cerr << "[SKIP] unit_generation_batching: failed to load model at '" << model_path
                  << "' (" << e.what() << ")\n";
        return nullptr;
    }
}

void test_two_active_requests_complete(ModelRuntime* runtime) {
    GenerationService service(runtime, 2, 2);
    service.start();
    // Log wall-clock time so batching changes can be compared during local runs.
    auto start_time = std::chrono::high_resolution_clock::now();

    auto request_a = make_request("Who are you?", 8);
    auto stream_a = service.submit(request_a);
    
    auto request_b = make_request("How are you?", 8);
    auto stream_b = service.submit(request_b);

    std::atomic<int> sequence_counter{0};
    std::vector<ObservedEvent> events_a;
    std::vector<ObservedEvent> events_b;

    std::thread thread_a([&] { events_a = collect_stream(stream_a, &sequence_counter); });
    std::thread thread_b([&] { events_b = collect_stream(stream_b, &sequence_counter); });

    thread_a.join();
    thread_b.join();
    auto end_time = std::chrono::high_resolution_clock::now();
    service.stop();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cerr << "test_two_active_requests_complete duration: " << duration.count() << " ms\n";

    EXPECT_TRUE(!events_a.empty());
    EXPECT_TRUE(!events_b.empty());
    const std::string text_a = collected_text(events_a);
    std::cerr << "event a prompt: [" << request_a.messages[0].content << "]\n";
    std::cerr << "event a response: [" << escape_for_debug(text_a) << "]\n";
    const std::string text_b = collected_text(events_b);
    std::cerr << "event b prompt: [" << request_b.messages[0].content << "]\n";
    std::cerr << "event b response: [" << escape_for_debug(text_b) << "]\n";
    EXPECT_TRUE(!has_error_event(events_a));
    EXPECT_TRUE(!has_error_event(events_b));
    EXPECT_TRUE(has_finish_event(events_a));
    EXPECT_TRUE(has_finish_event(events_b));
    EXPECT_TRUE(!text_a.empty());
    EXPECT_TRUE(!text_b.empty());
}

void test_cancelled_request_unblocks_next(ModelRuntime* runtime) {
    GenerationService service(runtime, 1, 1);
    service.start();

    auto stream_a = service.submit(make_request(
        "Hi", 8));
    auto stream_b = service.submit(make_request(
        "Hello", 8));

    stream_a->cancel();

    std::atomic<int> sequence_counter{0};
    std::vector<ObservedEvent> events_a;
    std::vector<ObservedEvent> events_b;

    std::thread thread_a([&] { events_a = collect_stream(stream_a, &sequence_counter); });
    std::thread thread_b([&] { events_b = collect_stream(stream_b, &sequence_counter); });

    thread_a.join();
    thread_b.join();
    service.stop();

    EXPECT_TRUE(events_a.empty());
    EXPECT_TRUE(!events_b.empty());
    EXPECT_TRUE(!has_error_event(events_b));
    EXPECT_TRUE(has_finish_event(events_b));
}

void test_decode_batch_limit_still_completes(ModelRuntime* runtime) {
    GenerationService service(runtime, 2, 1);
    service.start();

    auto start_time = std::chrono::high_resolution_clock::now();

    auto stream_a = service.submit(make_request("Who are you?", 8));
    auto stream_b = service.submit(make_request("How are you?", 8));

    std::atomic<int> sequence_counter{0};
    std::vector<ObservedEvent> events_a;
    std::vector<ObservedEvent> events_b;

    std::thread thread_a([&] { events_a = collect_stream(stream_a, &sequence_counter); });
    std::thread thread_b([&] { events_b = collect_stream(stream_b, &sequence_counter); });

    thread_a.join();
    thread_b.join();

    auto end_time = std::chrono::high_resolution_clock::now();

    service.stop();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cerr << "test_decode_batch_limit_still_completes duration: " << duration.count() << " ms\n";

    const std::string text_a = collected_text(events_a);
    std::cerr << "event a response: [" << escape_for_debug(text_a) << "]\n";
    const std::string text_b = collected_text(events_b);
    std::cerr << "event b response: [" << escape_for_debug(text_b) << "]\n";

    EXPECT_TRUE(!events_a.empty());
    EXPECT_TRUE(!events_b.empty());
    EXPECT_TRUE(!has_error_event(events_a));
    EXPECT_TRUE(!has_error_event(events_b));
    EXPECT_TRUE(has_finish_event(events_a));
    EXPECT_TRUE(has_finish_event(events_b));
}

void test_decode_batch_size_zero_defaults_and_completes(ModelRuntime* runtime) {
    GenerationService service(runtime, 2, 0);
    service.start();

    auto stream_a = service.submit(make_request("One short reply.", 8));
    auto stream_b = service.submit(make_request("Give me a short greeting.", 8));

    std::atomic<int> sequence_counter{0};
    std::vector<ObservedEvent> events_a;
    std::vector<ObservedEvent> events_b;

    std::thread thread_a([&] { events_a = collect_stream(stream_a, &sequence_counter); });
    std::thread thread_b([&] { events_b = collect_stream(stream_b, &sequence_counter); });

    thread_a.join();
    thread_b.join();
    service.stop();

    EXPECT_TRUE(!events_a.empty());
    EXPECT_TRUE(!events_b.empty());
    EXPECT_TRUE(!has_error_event(events_a));
    EXPECT_TRUE(!has_error_event(events_b));
    EXPECT_TRUE(has_finish_event(events_a));
    EXPECT_TRUE(has_finish_event(events_b));
}

void test_mixed_prompt_lengths_batched(ModelRuntime* runtime) {
    GenerationService service(runtime, 2, 2);
    service.start();

    auto stream_a = service.submit(make_request("Hi", 8));
    auto stream_b = service.submit(make_request(
        "Please write a slightly longer but still concise response about what you are.",
        8));

    std::atomic<int> sequence_counter{0};
    std::vector<ObservedEvent> events_a;
    std::vector<ObservedEvent> events_b;

    std::thread thread_a([&] { events_a = collect_stream(stream_a, &sequence_counter); });
    std::thread thread_b([&] { events_b = collect_stream(stream_b, &sequence_counter); });

    thread_a.join();
    thread_b.join();
    service.stop();

    EXPECT_TRUE(!events_a.empty());
    EXPECT_TRUE(!events_b.empty());
    EXPECT_TRUE(!has_error_event(events_a));
    EXPECT_TRUE(!has_error_event(events_b));
    EXPECT_TRUE(has_finish_event(events_a));
    EXPECT_TRUE(has_finish_event(events_b));

    const int first_a = first_sequence(events_a);
    const int first_b = first_sequence(events_b);
    const int finish_a = finish_sequence(events_a);
    const int finish_b = finish_sequence(events_b);
    EXPECT_TRUE(first_a > 0);
    EXPECT_TRUE(first_b > 0);
    EXPECT_TRUE(finish_a > 0);
    EXPECT_TRUE(finish_b > 0);
    EXPECT_TRUE(first_a < finish_b);
    EXPECT_TRUE(first_b < finish_a);
}

void test_three_way_fairness_with_batch_of_two(ModelRuntime* runtime) {
    GenerationService service(runtime, 3, 2);
    service.start();

    auto stream_a = service.submit(make_request("Give a short sentence about yourself.", 8));
    auto stream_b = service.submit(make_request("Say hello in one sentence.", 8));
    auto stream_c = service.submit(make_request("Provide one concise status update.", 8));

    std::atomic<int> sequence_counter{0};
    std::vector<ObservedEvent> events_a;
    std::vector<ObservedEvent> events_b;
    std::vector<ObservedEvent> events_c;

    std::thread thread_a([&] { events_a = collect_stream(stream_a, &sequence_counter); });
    std::thread thread_b([&] { events_b = collect_stream(stream_b, &sequence_counter); });
    std::thread thread_c([&] { events_c = collect_stream(stream_c, &sequence_counter); });

    thread_a.join();
    thread_b.join();
    thread_c.join();
    service.stop();

    EXPECT_TRUE(!events_a.empty());
    EXPECT_TRUE(!events_b.empty());
    EXPECT_TRUE(!events_c.empty());
    EXPECT_TRUE(!has_error_event(events_a));
    EXPECT_TRUE(!has_error_event(events_b));
    EXPECT_TRUE(!has_error_event(events_c));
    EXPECT_TRUE(has_finish_event(events_a));
    EXPECT_TRUE(has_finish_event(events_b));
    EXPECT_TRUE(has_finish_event(events_c));

    const int first_a = first_sequence(events_a);
    const int first_b = first_sequence(events_b);
    const int first_c = first_sequence(events_c);
    const int finish_a = finish_sequence(events_a);
    const int finish_b = finish_sequence(events_b);
    const int finish_c = finish_sequence(events_c);

    EXPECT_TRUE(first_a > 0);
    EXPECT_TRUE(first_b > 0);
    EXPECT_TRUE(first_c > 0);
    EXPECT_TRUE(finish_a > 0);
    EXPECT_TRUE(finish_b > 0);
    EXPECT_TRUE(finish_c > 0);

    EXPECT_TRUE(first_a < finish_b || first_a < finish_c);
    EXPECT_TRUE(first_b < finish_a || first_b < finish_c);
    EXPECT_TRUE(first_c < finish_a || first_c < finish_b);
}

}  // namespace

int main() {
    std::unique_ptr<ModelRuntime> runtime;
    ModelRuntime* runtime_ptr = make_runtime_or_skip(&runtime);
    if (runtime_ptr == nullptr) {
        return 0;
    }

    test_two_active_requests_complete(runtime_ptr);
    test_cancelled_request_unblocks_next(runtime_ptr);
    test_decode_batch_limit_still_completes(runtime_ptr);
    test_decode_batch_size_zero_defaults_and_completes(runtime_ptr);
    test_mixed_prompt_lengths_batched(runtime_ptr);
    test_three_way_fairness_with_batch_of_two(runtime_ptr);

    return test_util::finalize();
}
