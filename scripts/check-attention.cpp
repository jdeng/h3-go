#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <utility>
#include <vector>

static bool check_attention(ggml_backend_t backend, int tokens, int heads, int batches,
                            const std::string & output_prefix, const std::string & reference_prefix, int channels = 128) {
    auto context = ggml_init({16 * 1024 * 1024, nullptr, true});
    auto query = ggml_new_tensor_4d(context, GGML_TYPE_F32, channels, tokens, heads, batches);
    auto keys = ggml_new_tensor_4d(context, GGML_TYPE_F16, channels, tokens, heads, batches);
    auto values = ggml_new_tensor_4d(context, GGML_TYPE_F16, channels, tokens, heads, batches);
    auto output = ggml_flash_attn_ext(context, query, keys, values, nullptr, float(channels) / std::sqrt(float(channels)), 0.f, 0.f);
    ggml_prec_set_acc(output, GGML_PREC_F32);
    auto graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, output);
    auto buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    if (buffer == nullptr) {
        ggml_free(context);
        return false;
    }
    std::vector<float> queries(ggml_nelements(query));
    std::vector<ggml_fp16_t> key_data(queries.size()), value_data(queries.size());
    std::mt19937 random(42);
    for (size_t index = 0; index < queries.size(); ++index) {
        queries[index] = (float(random() & 65535) / 32768.f - 1.f) * 2.f;
        key_data[index] = ggml_fp32_to_fp16((float(random() & 65535) / 32768.f - 1.f) / 64.f);
        value_data[index] = ggml_fp32_to_fp16((float(random() & 65535) / 32768.f - 1.f) / float(channels));
    }
    ggml_backend_tensor_set(query, queries.data(), 0, ggml_nbytes(query));
    ggml_backend_tensor_set(keys, key_data.data(), 0, ggml_nbytes(keys));
    ggml_backend_tensor_set(values, value_data.data(), 0, ggml_nbytes(values));
    bool success = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    std::vector<double> timings;
    for (int repetition = 0; repetition < 7; ++repetition) {
        const auto start = std::chrono::steady_clock::now();
        success = (ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS) && success;
        const auto stop = std::chrono::steady_clock::now();
        timings.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
    }
    std::sort(timings.begin(), timings.end());
    std::vector<float> actual(ggml_nelements(output));
    ggml_backend_tensor_get(output, actual.data(), 0, ggml_nbytes(output));
    for (float value : actual) {
        success = std::isfinite(value) && success;
    }
    double maximum_error = 0;
    std::vector<std::pair<size_t, double>> cpu_samples;
    for (int batch = 0; batch < batches; ++batch) {
        for (int head : {0, heads - 1}) {
            for (int token : {0, tokens / 2, tokens - 1}) {
                const size_t source_base = (size_t(batch) * heads + head) * tokens * channels;
                std::vector<double> scores(tokens);
                for (int key = 0; key < tokens; ++key) {
                    double dot = 0;
                    for (int channel = 0; channel < channels; ++channel) {
                        const float query_value = ggml_fp16_to_fp32(ggml_fp32_to_fp16(queries[source_base + token * channels + channel]));
                        dot += double(query_value) * ggml_fp16_to_fp32(key_data[source_base + key * channels + channel]);
                    }
                    scores[key] = dot * (float(channels) / std::sqrt(float(channels)));
                }
                const double maximum = *std::max_element(scores.begin(), scores.end());
                double denominator = 0;
                for (double & score : scores) {
                    score = std::exp(score - maximum);
                    denominator += score;
                }
                for (int channel = 0; channel < channels; ++channel) {
                    double expected = 0;
                    for (int key = 0; key < tokens; ++key) {
                        expected += scores[key] * ggml_fp16_to_fp32(value_data[source_base + key * channels + channel]);
                    }
                    expected /= denominator;
                    const size_t destination = ((size_t(batch) * tokens + token) * heads + head) * channels + channel;
                    cpu_samples.emplace_back(destination, expected);
                    maximum_error = std::max(maximum_error, std::abs(actual[destination] - expected));
                }
            }
        }
    }
    success = maximum_error < 5e-6 && success;
    const std::string suffix = (channels == 128 ? std::string() : "-d" + std::to_string(channels)) + "-" + std::to_string(tokens) + "-" + std::to_string(heads) + "-" + std::to_string(batches) + ".bin";
    double relative_error = 0;
    double reference_max_error = 0;
    double reference_cpu_error = 0;
    if (!reference_prefix.empty()) {
        std::ifstream reference(reference_prefix + suffix, std::ios::binary);
        std::vector<float> expected(actual.size());
        reference.read(reinterpret_cast<char *>(expected.data()), expected.size() * sizeof(float));
        success = reference.good() && reference.peek() == std::char_traits<char>::eof() && success;
        double square_error = 0;
        double square_reference = 0;
        for (size_t index = 0; index < actual.size(); ++index) {
            const double difference = double(actual[index]) - expected[index];
            square_error += difference * difference;
            square_reference += double(expected[index]) * expected[index];
            reference_max_error = std::max(reference_max_error, std::abs(difference));
        }
        relative_error = std::sqrt(square_error / std::max(square_reference, 1e-30));
        for (const auto & sample : cpu_samples) {
            reference_cpu_error = std::max(reference_cpu_error, std::abs(expected[sample.first] - sample.second));
        }
        success = reference_max_error < 2e-5 && relative_error < 0.006 &&
                  maximum_error <= 1.25 * reference_cpu_error + 2e-7 && success;
    }
    if (!output_prefix.empty()) {
        std::ofstream destination(output_prefix + suffix, std::ios::binary);
        destination.write(reinterpret_cast<const char *>(actual.data()), actual.size() * sizeof(float));
        success = destination.good() && success;
    }
    std::printf("channels=%d tokens=%d heads=%d batches=%d median_ms=%.3f cpu_max_error=%.8g reference_cpu_error=%.8g reference_max_error=%.8g relative_rmse=%.8g %s\n",
                channels, tokens, heads, batches, timings[timings.size() / 2], maximum_error, reference_cpu_error, reference_max_error, relative_error,
                success ? "PASS" : "FAIL");
    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    return success;
}

int main(int argc, char ** argv) {
    if (argc > 3) {
        std::fprintf(stderr, "Usage: check-attention [output-prefix|-] [reference-prefix]\n");
        return 2;
    }
    const std::string output_prefix = argc > 1 && std::string(argv[1]) != "-" ? argv[1] : "";
    const std::string reference_prefix = argc > 2 ? argv[2] : "";
    auto backend = ggml_backend_cuda_init(0);
    if (backend == nullptr) {
        return 1;
    }
    bool success = check_attention(backend, 129, 2, 2, output_prefix, reference_prefix);
    success = check_attention(backend, 1024, 2, 2, output_prefix, reference_prefix) && success;
    success = check_attention(backend, 4350, 56, 1, output_prefix, reference_prefix) && success;
    success = check_attention(backend, 16800, 56, 1, output_prefix, reference_prefix) && success;
    success = check_attention(backend, 129, 2, 4, output_prefix, reference_prefix, 64) && success;
    success = check_attention(backend, 1797, 32, 4, output_prefix, reference_prefix, 64) && success;
    ggml_backend_free(backend);
    return success ? 0 : 1;
}
