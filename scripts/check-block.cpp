#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

static ggml_tensor* span_view(ggml_context* context, ggml_tensor* input, int begin, int end) {
    return ggml_view_4d(context, input, input->ne[0], end - begin, input->ne[2], input->ne[3],
                        input->nb[1], input->nb[2], input->nb[3], size_t(begin) * input->nb[1]);
}

static bool check_block(ggml_backend_t backend, int channels, int tokens, int batches,
                        const std::vector<int32_t>& ends, const std::vector<int32_t>& rows, int shift) {
    auto context = ggml_init({16 * 1024 * 1024, nullptr, true});
    auto input = ggml_new_tensor_3d(context, GGML_TYPE_F32, channels, tokens, batches);
    auto weight = ggml_new_tensor_1d(context, GGML_TYPE_F32, channels);
    auto update = ggml_dup_tensor(context, input);
    auto projection = ggml_new_tensor_2d(context, GGML_TYPE_F32, channels * 18, 2);
    auto normalized = ggml_mul(context, ggml_rms_norm(context, input, 1e-5f), weight);
    auto norm_input = ggml_dup(context, input);
    auto residual_update = ggml_dup(context, update);
    auto fused_norm = ggml_h3_rms_modulate(context, norm_input, weight, projection, ends.data(), rows.data(),
                                          ends.size(), 6, shift, 1e-5f);
    auto fused_gate = ggml_h3_gated_residual(context, input, residual_update, projection, ends.data(), rows.data(),
                                            ends.size(), 6, shift + 2);
    ggml_tensor* reference_norm = nullptr;
    ggml_tensor* reference_gate = nullptr;
    int begin = 0;
    for (size_t span = 0; span < ends.size(); ++span) {
        const size_t offset = size_t(rows[span] * 6 + shift) * channels * sizeof(float);
        auto bias = ggml_view_1d(context, projection, channels, offset);
        auto scale = ggml_view_1d(context, projection, channels, offset + channels * sizeof(float));
        auto gate = ggml_view_1d(context, projection, channels, offset + channels * sizeof(float) * 2);
        auto part = span_view(context, normalized, begin, ends[span]);
        part = ggml_add(context, ggml_add(context, part, ggml_mul(context, part, scale)), bias);
        auto residual = ggml_add(context, span_view(context, input, begin, ends[span]),
                                 ggml_mul(context, span_view(context, update, begin, ends[span]), gate));
        reference_norm = reference_norm == nullptr ? part : ggml_concat(context, reference_norm, part, 1);
        reference_gate = reference_gate == nullptr ? residual : ggml_concat(context, reference_gate, residual, 1);
        begin = ends[span];
    }
    auto graph = ggml_new_graph(context);
    for (auto* source : {input, weight, update, projection}) {
        ggml_set_input(source);
    }
    for (auto* result : {fused_norm, fused_gate, reference_norm, reference_gate}) {
        ggml_set_output(result);
        ggml_build_forward_expand(graph, result);
    }
    auto allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(allocator, graph)) {
        ggml_gallocr_free(allocator);
        ggml_free(context);
        return false;
    }
    for (auto* tensor : {input, weight, update, projection}) {
        std::vector<float> values(ggml_nelements(tensor));
        for (size_t index = 0; index < values.size(); ++index) {
            values[index] = std::sin(float(index % 7919) * 0.07f) * (tensor == input ? 17.0f : 0.8f);
        }
        ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(float));
    }
    bool success = ggml_backend_supports_op(backend, fused_norm) && ggml_backend_supports_op(backend, fused_gate) &&
                   ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    success = success && fused_norm->data == norm_input->data && fused_gate->data == residual_update->data;
    float maximum_error = 0.0f;
    for (auto pair : {std::make_pair(fused_norm, reference_norm), std::make_pair(fused_gate, reference_gate)}) {
        std::vector<float> actual(ggml_nelements(pair.first)), expected(actual.size());
        ggml_backend_tensor_get(pair.first, actual.data(), 0, actual.size() * sizeof(float));
        ggml_backend_tensor_get(pair.second, expected.data(), 0, expected.size() * sizeof(float));
        for (size_t index = 0; index < actual.size(); ++index) {
            const float error = std::abs(actual[index] - expected[index]);
            maximum_error = std::max(maximum_error, error);
            success = success && std::isfinite(actual[index]) && error < 5e-6f * (1.0f + std::abs(expected[index]));
        }
    }
    std::printf("channels=%d tokens=%d batches=%d spans=%zu shift=%d max_error=%g %s\n",
                channels, tokens, batches, ends.size(), shift, maximum_error, success ? "PASS" : "FAIL");
    ggml_gallocr_free(allocator);
    ggml_free(context);
    return success;
}

static bool check_swiglu(ggml_backend_t backend, int channels, int tokens, float base_scale, float adapter_scale) {
    auto context = ggml_init({16 * 1024 * 1024, nullptr, true});
    auto base = ggml_new_tensor_2d(context, GGML_TYPE_F32, channels * 2, tokens);
    auto adapter = ggml_dup_tensor(context, base);
    auto fused = ggml_h3_swiglu(context, base, adapter, base_scale, adapter_scale);
    auto expected = ggml_swiglu(context, ggml_add(context, ggml_scale(context, base, base_scale),
                                                  ggml_scale(context, adapter, adapter_scale)));
    auto graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, fused);
    ggml_build_forward_expand(graph, expected);
    auto buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    if (buffer == nullptr) {
        ggml_free(context);
        return false;
    }
    for (auto* source : {base, adapter}) {
        std::vector<float> values(ggml_nelements(source));
        for (size_t index = 0; index < values.size(); ++index) {
            values[index] = std::sin(float(index % 7919) * (source == base ? 0.07f : 0.09f)) * 0.3f;
        }
        ggml_backend_tensor_set(source, values.data(), 0, values.size() * sizeof(float));
    }
    bool success = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    std::vector<float> actual_values(ggml_nelements(fused)), expected_values(actual_values.size());
    ggml_backend_tensor_get(fused, actual_values.data(), 0, actual_values.size() * sizeof(float));
    ggml_backend_tensor_get(expected, expected_values.data(), 0, expected_values.size() * sizeof(float));
    float maximum_error = 0.f;
    for (size_t index = 0; index < actual_values.size(); ++index) {
        maximum_error = std::max(maximum_error, std::abs(actual_values[index] - expected_values[index]));
        success = success && std::isfinite(actual_values[index]) && actual_values[index] == expected_values[index];
    }
    std::printf("swiglu channels=%d tokens=%d scales=%g,%g max_error=%g %s\n",
                channels, tokens, base_scale, adapter_scale, maximum_error, success ? "PASS" : "FAIL");
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
    for (int shift : {0, 3}) {
        success = check_block(backend, 5376, 129, 2, {1, 84, 129}, {0, 3, 5}, shift) && success;
        success = check_block(backend, 33, 17, 3, {1, 2, 3, 7, 16, 17}, {5, 1, 5, 0, 2, 3}, shift) && success;
        success = check_block(backend, 5376, 1, 1, {1}, {4}, shift) && success;
        success = check_block(backend, 5376, 4149, 1, {84, 309, 4149}, {0, 1, 2}, shift) && success;
    }
    success = check_swiglu(backend, 14336, 257, 128.f, 128.f) && success;
    success = check_swiglu(backend, 14336, 1, 128.f, 1.f) && success;
    success = check_swiglu(backend, 33, 17, 0.37f, -1.25f) && success;
    ggml_backend_free(backend);
    return success ? 0 : 1;
}
