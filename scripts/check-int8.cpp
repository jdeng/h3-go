#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

static bool check_shape(ggml_backend_t backend, int rows, int columns, int inner, bool with_bias) {
    auto context = ggml_init({16 * 1024 * 1024, nullptr, true});
    auto weights = ggml_new_tensor_2d(context, GGML_TYPE_I8, inner, columns);
    auto inputs = ggml_new_tensor_2d(context, GGML_TYPE_F32, inner, rows);
    auto scales = ggml_new_tensor_1d(context, GGML_TYPE_F32, columns);
    auto bias = with_bias ? ggml_new_tensor_1d(context, GGML_TYPE_F32, columns) : nullptr;
    auto output = ggml_mul_mat_i8_tensorwise(context, weights, inputs, scales, bias, 0);
    auto graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, output);
    auto buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    if (buffer == nullptr) {
        ggml_free(context);
        return false;
    }
    std::vector<int8_t> weight_data(size_t(columns) * inner);
    std::vector<float> input_data(size_t(rows) * inner);
    std::vector<float> scale_data(columns), bias_data(columns);
    for (size_t index = 0; index < weight_data.size(); ++index) {
        weight_data[index] = int((index * 31 + index / 43) % 255) - 127;
    }
    for (int row = 0; row < rows; ++row) {
        const float row_scale = (row % 3 + 1) * 0.25f;
        for (int column = 0; column < inner; ++column) {
            input_data[size_t(row) * inner + column] =
                (column == 0 ? 127 : int((column * 13 + row * 17) % 255) - 127) * row_scale;
        }
    }
    for (int column = 0; column < columns; ++column) {
        scale_data[column] = (column % 7 + 1) / 1024.0f;
        bias_data[column] = (column % 13 - 6) / 32.0f;
    }
    ggml_backend_tensor_set(weights, weight_data.data(), 0, weight_data.size());
    ggml_backend_tensor_set(inputs, input_data.data(), 0, input_data.size() * sizeof(float));
    ggml_backend_tensor_set(scales, scale_data.data(), 0, scale_data.size() * sizeof(float));
    if (bias != nullptr) {
        ggml_backend_tensor_set(bias, bias_data.data(), 0, bias_data.size() * sizeof(float));
    }
    bool success = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    float maximum_error = 0;
    for (int sample = 0; success && sample < 128; ++sample) {
        const int row = sample == 0 ? rows - 1 : (sample * 97) % rows;
        const int column = sample == 0 ? columns - 1 : (sample * 7919) % columns;
        const float row_scale = (row % 3 + 1) * 0.25f;
        int64_t accumulator = 0;
        for (int inner_index = 0; inner_index < inner; ++inner_index) {
            const int activation = int(input_data[size_t(row) * inner + inner_index] / row_scale);
            accumulator += activation * weight_data[size_t(column) * inner + inner_index];
        }
        const float expected = float(accumulator) * row_scale * scale_data[column] +
                               (with_bias ? bias_data[column] : 0);
        float actual = 0;
        ggml_backend_tensor_get(output, &actual, (size_t(row) * columns + column) * sizeof(float), sizeof(float));
        maximum_error = std::max(maximum_error, std::abs(actual - expected));
        success = std::isfinite(actual) && std::abs(actual - expected) <= 1e-3f + 1e-5f * std::abs(expected);
    }
    std::printf("M=%d N=%d K=%d bias=%d max_error=%g %s\n", rows, columns, inner,
                with_bias, maximum_error, success ? "PASS" : "FAIL");
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
    for (const auto &shape : std::vector<std::pair<int, int>>{{21504, 5376}, {28672, 5376}, {5376, 7168}, {5376, 14336}}) {
        for (int rows : {127, 129, 4149}) {
            success = check_shape(backend, rows, shape.first, shape.second, rows != 129) && success;
        }
    }
    ggml_backend_free(backend);
    return success ? 0 : 1;
}
