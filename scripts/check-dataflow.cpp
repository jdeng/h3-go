#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static bool equal_tensors(ggml_tensor* actual, ggml_tensor* expected) {
    if (actual->type != expected->type || !ggml_are_same_shape(actual, expected)) {
        return false;
    }
    std::vector<unsigned char> actual_data(ggml_nbytes(actual)), expected_data(ggml_nbytes(expected));
    ggml_backend_tensor_get(actual, actual_data.data(), 0, actual_data.size());
    ggml_backend_tensor_get(expected, expected_data.data(), 0, expected_data.size());
    return actual_data == expected_data;
}

static bool check_activation(ggml_backend_t backend, int channels, int tokens, int batches, bool swiglu) {
    auto context = ggml_init({16 * 1024 * 1024, nullptr, true});
    auto input = ggml_new_tensor_3d(context, GGML_TYPE_F32, channels * (swiglu ? 2 : 1), tokens, batches);
    auto adapter = ggml_dup_tensor(context, input);
    auto weight = ggml_new_tensor_1d(context, GGML_TYPE_F32, channels);
    auto projection = ggml_new_tensor_2d(context, GGML_TYPE_F32, channels * 18, 1);
    const int32_t ends[] = {1, tokens};
    const int32_t rows[] = {2, 0};
    auto actual = swiglu ? ggml_h3_swiglu_ext(context, input, adapter, 1.f, 0.5f, GGML_TYPE_BF16)
                        : ggml_h3_rms_modulate_ext(context, input, weight, projection, ends, rows, 2, 6, 3, 1e-5f, GGML_TYPE_BF16);
    auto full = swiglu ? ggml_h3_swiglu(context, input, adapter, 1.f, 0.5f)
                      : ggml_h3_rms_modulate(context, input, weight, projection, ends, rows, 2, 6, 3, 1e-5f);
    auto expected = ggml_cast(context, full, GGML_TYPE_BF16);
    auto packed = ggml_quantize_i8_convrot(context, actual, 256);
    auto packed_reference = ggml_quantize_i8_convrot(context, ggml_cast(context, expected, GGML_TYPE_F32), 256);
    auto graph = ggml_new_graph(context);
    for (auto output : {actual, expected, packed, packed_reference}) {
        ggml_build_forward_expand(graph, output);
    }
    auto buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    if (buffer == nullptr) {
        ggml_free(context);
        return false;
    }
    for (auto tensor : {input, adapter, weight, projection}) {
        std::vector<float> data(ggml_nelements(tensor));
        for (size_t index = 0; index < data.size(); ++index) {
            data[index] = std::sin(float(index % 7919) * 0.013f) * (tensor == input ? 37.f : 0.2f);
        }
        ggml_backend_tensor_set(tensor, data.data(), 0, ggml_nbytes(tensor));
    }
    ggml_backend_tensor_memset(packed, 0, 0, ggml_nbytes(packed));
    ggml_backend_tensor_memset(packed_reference, 0, 0, ggml_nbytes(packed_reference));
    const bool success = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS &&
                         equal_tensors(actual, expected) && equal_tensors(packed, packed_reference) &&
                         ggml_nbytes(actual) * 2 == ggml_nbytes(full);
    std::printf("%s channels=%d tokens=%d batches=%d BF16-and-INT8-exact=%s activation_bytes=%zu/%zu\n",
                swiglu ? "swiglu" : "rms-modulate", channels, tokens, batches,
                success ? "PASS" : "FAIL", ggml_nbytes(actual), ggml_nbytes(full));
    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    return success;
}

static bool check_quantization(ggml_backend_t backend, int channels, int rows, int group, bool bf16) {
    auto context = ggml_init({16 * 1024 * 1024, nullptr, true});
    auto input = ggml_new_tensor_2d(context, bf16 ? GGML_TYPE_BF16 : GGML_TYPE_F32, channels, rows);
    auto output = ggml_quantize_i8_convrot(context, input, group);
    auto graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, output);
    auto buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    if (buffer == nullptr) {
        ggml_free(context);
        return false;
    }
    std::vector<float> values(ggml_nelements(input));
    std::vector<ggml_bf16_t> reduced(values.size());
    for (size_t index = 0; index < values.size(); ++index) {
        const int row = index / channels;
        values[index] = row == 0 ? 0.f : std::sin(float(index % 7919) * 0.017f) * (row % 2 ? 80000.f : 0.1f);
        if (bf16) {
            reduced[index] = ggml_fp32_to_bf16(values[index]);
            values[index] = ggml_bf16_to_fp32(reduced[index]);
        }
    }
    ggml_backend_tensor_set(input, bf16 ? static_cast<const void*>(reduced.data()) : values.data(), 0, ggml_nbytes(input));
    bool success = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    std::vector<int8_t> quantized(size_t(channels) * ((rows + 3) / 4 * 4));
    std::vector<float> scales(rows);
    ggml_backend_tensor_get(output, quantized.data(), 0, quantized.size());
    ggml_backend_tensor_get(output, scales.data(), quantized.size(), scales.size() * sizeof(float));
    for (int row = 0; row < rows; ++row) {
        std::vector<float> rotated(channels);
        for (int channel = 0; channel < channels; ++channel) {
            rotated[channel] = values[size_t(row) * channels + channel] / std::sqrt(float(group));
        }
        for (int begin = 0; begin < channels; begin += group) {
            for (int stride = 1; stride < group; stride *= 4) {
                for (int lane = 0; lane < group / 4; ++lane) {
                    const int offset = begin + (lane / stride) * 4 * stride + lane % stride;
                    const float first = rotated[offset];
                    const float second = rotated[offset + stride];
                    const float third = rotated[offset + stride * 2];
                    const float fourth = rotated[offset + stride * 3];
                    rotated[offset] = first + second + third - fourth;
                    rotated[offset + stride] = first + second - third + fourth;
                    rotated[offset + stride * 2] = first - second + third + fourth;
                    rotated[offset + stride * 3] = -first + second + third + fourth;
                }
            }
        }
        float maximum = 0.f;
        for (float value : rotated) {
            maximum = std::max(maximum, std::abs(value));
        }
        const float scale = maximum / 127.f;
        success = scales[row] == scale && success;
        const float inverse = scale == 0.f ? 0.f : 1.f / scale;
        for (int channel = 0; channel < channels; ++channel) {
            const int expected = std::clamp(int(std::nearbyint(rotated[channel] * inverse)), -127, 127);
            success = quantized[size_t(row) * channels + channel] == expected && success;
        }
    }
    std::printf("quant channels=%d rows=%d group=%d bf16=%d CPU-exact=%s\n", channels, rows, group, bf16, success ? "PASS" : "FAIL");
    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    return success;
}

static bool check_bf16_matmul(ggml_backend_t backend, int tokens) {
    constexpr int inner = 256;
    constexpr int channels = 28672;
    auto context = ggml_init({16 * 1024 * 1024, nullptr, true});
    auto weight = ggml_new_tensor_2d(context, GGML_TYPE_BF16, inner, channels);
    auto input = ggml_new_tensor_2d(context, GGML_TYPE_BF16, inner, tokens);
    auto output = ggml_mul_mat(context, weight, input);
    auto promoted = ggml_mul_mat(context, weight, input);
    ggml_prec_set_acc(promoted, GGML_PREC_F32);
    auto graph = ggml_new_graph(context);
    auto reference_graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, output);
    ggml_build_forward_expand(reference_graph, promoted);
    auto buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    if (buffer == nullptr) {
        ggml_free(context);
        return false;
    }
    std::vector<ggml_bf16_t> weights(ggml_nelements(weight)), inputs(ggml_nelements(input));
    for (size_t index = 0; index < weights.size(); ++index) {
        weights[index] = ggml_fp32_to_bf16(std::sin(float(index % 7919) * 0.017f) * 0.03f);
    }
    for (size_t index = 0; index < inputs.size(); ++index) {
        inputs[index] = ggml_fp32_to_bf16(std::cos(float(index % 7879) * 0.013f));
    }
    ggml_backend_tensor_set(weight, weights.data(), 0, ggml_nbytes(weight));
    ggml_backend_tensor_set(input, inputs.data(), 0, ggml_nbytes(input));
    bool success = true;
    double times[2] = {};
    int variant = 0;
    for (auto current : {reference_graph, graph}) {
        std::vector<double> samples;
        for (int repetition = 0; repetition < 5; ++repetition) {
            const auto start = std::chrono::steady_clock::now();
            success = ggml_backend_graph_compute(backend, current) == GGML_STATUS_SUCCESS && success;
            if (repetition > 0) {
                samples.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
            }
        }
        std::sort(samples.begin(), samples.end());
        times[variant++] = samples[samples.size() / 2];
    }
    float maximum_error = 0.f;
    for (int sample = 0; sample < 128; ++sample) {
        const int token = (sample * 97) % tokens;
        const int channel = (sample * 7919) % channels;
        double expected = 0;
        for (int index = 0; index < inner; ++index) {
            expected += double(ggml_bf16_to_fp32(weights[channel * inner + index])) * ggml_bf16_to_fp32(inputs[token * inner + index]);
        }
        float actual;
        ggml_backend_tensor_get(output, &actual, (size_t(token) * channels + channel) * sizeof(float), sizeof(float));
        maximum_error = std::max(maximum_error, float(std::abs(actual - expected)));
        success = std::isfinite(actual) && std::abs(actual - expected) < 1e-5 && success;
    }
    std::printf("LoRA-up tokens=%d promoted-F32-ms=%.3f BF16-operands-F32-acc-ms=%.3f CPU-max-error=%g %s\n",
                tokens, times[0], times[1], maximum_error, success ? "PASS" : "FAIL");
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
    for (bool swiglu : {false, true}) {
        success = check_activation(backend, 5376, 129, 2, swiglu) && success;
        success = check_activation(backend, 14336, 17, 1, swiglu) && success;
    }
    for (int channels : {5376, 14336, 32768}) {
        for (int group : {64, 256}) {
            for (bool bf16 : {false, true}) {
                success = check_quantization(backend, channels, 5, group, bf16) && success;
            }
        }
    }
    success = check_bf16_matmul(backend, 4350) && success;
    success = check_bf16_matmul(backend, 16800) && success;
    ggml_backend_free(backend);
    return success ? 0 : 1;
}
