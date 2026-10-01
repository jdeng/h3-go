#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

static bool check_rope(ggml_backend_t backend, int heads, int tokens, int batches, int pairs, int channels = 128, bool affine = true) {
    auto context = ggml_init({16 * 1024 * 1024, nullptr, true});
    auto storage = ggml_new_tensor_4d(context, GGML_TYPE_F32, channels, heads * 3, tokens, batches);
    auto input = ggml_view_4d(context, storage, channels, heads, tokens, batches,
                             storage->nb[1] * (channels == 64 ? 3 : 1), storage->nb[2], storage->nb[3],
                             size_t(channels == 64 ? 1 : heads) * channels * sizeof(float));
    auto weight = ggml_new_tensor_1d(context, GGML_TYPE_F32, channels);
    auto position = ggml_new_tensor_4d(context, GGML_TYPE_F32, 2, 2, pairs, tokens);
    const float epsilon = 1e-5f;
    auto output = ggml_h3_rms_rope(context, input, affine ? weight : nullptr, position, epsilon);
    auto keys = ggml_h3_rms_rope_ext(context, input, affine ? weight : nullptr, position, epsilon, GGML_TYPE_F16, 1.f / float(channels));
    auto values = ggml_h3_value_layout(context, input, 1.f / float(channels));
    auto reference_keys = ggml_cast(context, ggml_scale(context, output, 1.f / float(channels)), GGML_TYPE_F16);
    auto reference_values = ggml_cont(context, ggml_permute(context, input, 0, 2, 1, 3));
    reference_values = ggml_reshape_3d(context, reference_values, channels, tokens, heads * batches);
    reference_values = ggml_cast(context, ggml_scale(context, reference_values, 1.f / float(channels)), GGML_TYPE_F16);
    auto attention = ggml_flash_attn_ext(context, output, keys, values, nullptr, float(channels) / std::sqrt(float(channels)), 0.f, 0.f);
    auto reference_attention = ggml_flash_attn_ext(context, output, reference_keys, reference_values, nullptr,
                                                  float(channels) / std::sqrt(float(channels)), 0.f, 0.f);
    ggml_prec_set_acc(attention, GGML_PREC_F32);
    ggml_prec_set_acc(reference_attention, GGML_PREC_F32);
    auto graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, output);
    ggml_build_forward_expand(graph, attention);
    ggml_build_forward_expand(graph, reference_attention);
    auto buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    if (buffer == nullptr) {
        ggml_free(context);
        return false;
    }
    std::vector<float> source(ggml_nelements(storage)), weights(channels), positions(ggml_nelements(position));
    for (size_t index = 0; index < source.size(); ++index) {
        source[index] = std::sin(float(index % 7919) * 0.03f) * 17.0f;
    }
    for (int channel = 0; channel < channels; ++channel) {
        weights[channel] = affine ? 0.5f + channel / 256.0f : 1.f;
    }
    for (int token = 0; token < tokens; ++token) {
        for (int pair = 0; pair < pairs; ++pair) {
            const float angle = (token * 0.13f + pair * 0.27f);
            const size_t offset = (size_t(token) * pairs + pair) * 4;
            positions[offset] = std::cos(angle);
            positions[offset + 1] = -std::sin(angle);
            positions[offset + 2] = std::sin(angle);
            positions[offset + 3] = std::cos(angle);
        }
    }
    ggml_backend_tensor_set(storage, source.data(), 0, source.size() * sizeof(float));
    ggml_backend_tensor_set(weight, weights.data(), 0, weights.size() * sizeof(float));
    ggml_backend_tensor_set(position, positions.data(), 0, positions.size() * sizeof(float));
    bool success = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    std::vector<float> actual(ggml_nelements(output));
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    float maximum_error = 0;
    for (int batch = 0; batch < batches; ++batch) {
        for (int token = 0; token < tokens; ++token) {
            for (int head = 0; head < heads; ++head) {
                const size_t source_offset = ((size_t(batch) * tokens + token) * heads * 3 + (channels == 64 ? head * 3 + 1 : heads + head)) * channels;
                double sum = 0;
                for (int channel = 0; channel < channels; ++channel) {
                    const double value = source[source_offset + channel];
                    sum += value * value;
                }
                const float scale = 1.0f / std::sqrt(float(sum / channels) + epsilon);
                for (int channel = 0; channel < channels; ++channel) {
                    float expected = source[source_offset + channel] * scale * weights[channel];
                    if (channel < pairs * 2) {
                        const int pair = channel % pairs;
                        const size_t position_offset = (size_t(token) * pairs + pair) * 4 + (channel / pairs) * 2;
                        const float first = source[source_offset + pair] * scale * weights[pair];
                        const float second = source[source_offset + pair + pairs] * scale * weights[pair + pairs];
                        expected = first * positions[position_offset] + second * positions[position_offset + 1];
                    }
                    const size_t output_offset = ((size_t(batch) * heads + head) * tokens + token) * channels + channel;
                    const float error = std::abs(actual[output_offset] - expected);
                    maximum_error = std::max(maximum_error, error);
                    success = success && std::isfinite(actual[output_offset]) && error < 3e-6f;
                }
            }
        }
    }
    for (auto pair : {std::make_pair(keys, reference_keys), std::make_pair(values, reference_values)}) {
        std::vector<ggml_fp16_t> actual_half(ggml_nelements(pair.first)), expected_half(actual_half.size());
        ggml_backend_tensor_get(pair.first, actual_half.data(), 0, actual_half.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_get(pair.second, expected_half.data(), 0, expected_half.size() * sizeof(ggml_fp16_t));
        success = success && actual_half == expected_half;
    }
    std::vector<float> fused_attention(ggml_nelements(attention)), expected_attention(fused_attention.size());
    ggml_backend_tensor_get(attention, fused_attention.data(), 0, fused_attention.size() * sizeof(float));
    ggml_backend_tensor_get(reference_attention, expected_attention.data(), 0, expected_attention.size() * sizeof(float));
    for (size_t index = 0; index < fused_attention.size(); ++index) {
        success = success && std::isfinite(fused_attention[index]) && fused_attention[index] == expected_attention[index];
    }
    std::printf("channels=%d affine=%d heads=%d tokens=%d batches=%d rotary_dim=%d max_error=%g exact_kv_attention=%s\n",
                channels, affine, heads, tokens, batches, pairs * 2, maximum_error, success ? "PASS" : "FAIL");
    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    return success;
}

int main() {
    auto backend = ggml_backend_cuda_init(0);
    if (backend == nullptr) {
        return 1;
    }
    bool success = true;
    for (int pairs : {1, 48, 64}) {
        success = check_rope(backend, 56, 129, 2, pairs) && success;
    }
    success = check_rope(backend, 1, 1, 1, 48) && success;
    success = check_rope(backend, 32, 129, 4, 24, 64, false) && success;
    success = check_rope(backend, 2, 1025, 2, 32, 64, true) && success;
    ggml_backend_free(backend);
    return success ? 0 : 1;
}
