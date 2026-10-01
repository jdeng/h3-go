#include "model/vae/minimax_h3_vae.hpp"
#include "runtime/tiling.h"
#include <chrono>
#include <cstdio>

sd::Tensor<float> reference_process_tiles_2d(const sd::Tensor<float>&, int, int, int, int, int, float,
                                             bool, bool, const TileProcessCallback&, bool);

static bool check_tiles(int width, int height, int scale, int tile_width, int tile_height,
                        float overlap, bool circular_x, bool circular_y, int batches) {
    sd::Tensor<float> input({width, height, 7, 3, batches});
    for (int64_t index = 0; index < input.numel(); ++index) {
        input[index] = float(index % 997) / 997.f;
    }
    auto resize = [scale](const sd::Tensor<float>& tile) {
        auto shape = tile.shape();
        shape[0] *= scale;
        shape[1] *= scale;
        sd::Tensor<float> output(shape);
        const int64_t input_plane = tile.shape()[0] * tile.shape()[1];
        const int64_t output_plane = shape[0] * shape[1];
        for (int64_t plane = 0; plane < tile.numel() / input_plane; ++plane) {
            for (int64_t row = 0; row < shape[1]; ++row) {
                for (int64_t column = 0; column < shape[0]; ++column) {
                    output[plane * output_plane + row * shape[0] + column] =
                        tile[plane * input_plane + (row / scale) * tile.shape()[0] + column / scale];
                }
            }
        }
        return output;
    };
    const auto started = std::chrono::steady_clock::now();
    auto expected = reference_process_tiles_2d(input, width * scale, height * scale, scale,
                                                tile_width, tile_height, overlap, circular_x, circular_y, resize, true);
    const auto middle = std::chrono::steady_clock::now();
    auto actual = process_tiles_2d(input, width * scale, height * scale, scale,
                                   tile_width, tile_height, overlap, circular_x, circular_y, resize, true);
    const auto finished = std::chrono::steady_clock::now();
    const bool success = actual.shape() == expected.shape() && actual.values() == expected.values();
    std::printf("tiles=%dx%d scale=%d batch=%d overlap=%.2f circular=%d/%d old_ms=%.2f new_ms=%.2f exact=%s\n",
                width, height, scale, batches, overlap, circular_x, circular_y,
                std::chrono::duration<double, std::milli>(middle - started).count(),
                std::chrono::duration<double, std::milli>(finished - middle).count(), success ? "PASS" : "FAIL");
    return success;
}

static bool check_temporal(int batches, int extent) {
    sd::Tensor<float> previous({13, 9, 5, 3, batches});
    sd::Tensor<float> current({13, 9, 17, 3, batches});
    for (int64_t index = 0; index < previous.numel(); ++index) {
        previous[index] = float(index % 71) / 71.f;
    }
    for (int64_t index = 0; index < current.numel(); ++index) {
        current[index] = float(index % 53) / 53.f;
    }
    auto expected = current;
    const int64_t count = std::min<int64_t>(extent, previous.shape()[2]);
    for (int64_t batch = 0; batch < batches; ++batch) {
        for (int64_t channel = 0; channel < 3; ++channel) {
            for (int64_t frame = 0; frame < count; ++frame) {
                const float weight = float(frame) / count;
                for (int64_t row = 0; row < 9; ++row) {
                    for (int64_t column = 0; column < 13; ++column) {
                        expected.index(column, row, frame, channel, batch) =
                            previous.index(column, row, 5 - count + frame, channel, batch) * (1.f - weight) +
                            current.index(column, row, frame, channel, batch) * weight;
                    }
                }
            }
        }
    }
    auto actual = MiniMaxH3VAE::MiniMaxH3VideoVAERunner::blend_temporal(previous, current, extent);
    const bool success = actual.shape() == expected.shape() && actual.values() == expected.values();
    std::printf("temporal batch=%d extent=%d exact=%s\n", batches, extent, success ? "PASS" : "FAIL");
    return success;
}

int main() {
    bool success = true;
    for (bool circular_x : {false, true}) {
        for (bool circular_y : {false, true}) {
            success = check_tiles(23, 19, 2, 12, 10, 0.25f, circular_x, circular_y, 2) && success;
        }
    }
    success = check_tiles(7, 5, 2, 16, 16, 0.f, false, false, 1) && success;
    success = check_tiles(32, 16, 1, 16, 16, 0.f, false, false, 2) && success;
    success = check_tiles(40, 24, 16, 16, 16, 0.25f, false, false, 4) && success;
    for (int extent : {0, 1, 5, 9}) {
        success = check_temporal(1, extent) && success;
        success = check_temporal(4, extent) && success;
    }
    return success ? 0 : 1;
}
