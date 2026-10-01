#include "core/ggml_graph_cut.h"
#include "core/runner_cache.h"
#include "ggml-alloc.h"
#include "ggml-cuda.h"
#include <cstdio>
#include <set>
#include <vector>

static bool check_cache(ggml_backend_t backend, int columns, int rows, bool reuse) {
    sd::GraphCutTensorCache cuts(backend);
    cuts.set_reuse(reuse);
    std::set<void*> addresses;
    bool success = true;
    for (int iteration = 0; iteration < 8; ++iteration) {
        auto context = ggml_init({2 * 1024 * 1024, nullptr, true});
        auto input = ggml_new_tensor_2d(context, GGML_TYPE_F32, columns, rows);
        auto output = input;
        for (int layer = 0; layer < 12; ++layer) {
            output = ggml_scale(context, output, 2.f);
        }
        sd::ggml_graph_cut::mark_graph_cut(output, "test." + std::to_string(iteration), "hidden");
        auto graph = ggml_new_graph(context);
        ggml_build_forward_expand(graph, output);
        auto buffer = ggml_backend_alloc_ctx_tensors(context, backend);
        if (buffer == nullptr) {
            ggml_free(context);
            return false;
        }
        std::vector<float> values(columns * rows, float(iteration + 1));
        ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
        sd::ggml_graph_cut::Segment segment;
        segment.output_node_indices.push_back(ggml_graph_n_nodes(graph) - 1);
        segment.future_cut_names.insert(output->name);
        if (reuse && !cuts.prepare(graph, segment)) {
            success = false;
            break;
        }
        success = ggml_backend_graph_compute_async(backend, graph) == GGML_STATUS_SUCCESS && success;
        success = cuts.capture(graph, segment, "pipeline-test") == GGML_STATUS_SUCCESS && success;
        ggml_backend_synchronize(backend);
        auto cached = cuts.get(output->name);
        if (cached == nullptr) {
            success = false;
            break;
        }
        addresses.insert(cached->data);
        ggml_backend_tensor_get(cached, values.data(), 0, values.size() * sizeof(float));
        for (float value : values) {
            success = value == float(iteration + 1) * 4096.f && success;
        }
        cuts.prune(segment.future_cut_names);
        ggml_backend_buffer_free(buffer);
        ggml_free(context);
    }
    if (reuse) {
        success = addresses.size() == 2 && success;
    }
    cuts.clear();
    success = cuts.resident_bytes(ggml_backend_get_device(backend)) == 0 && success;
    std::printf("%dx%d cache reuse=%d distinct_buffers=%zu: %s\n",
                columns, rows, reuse, addresses.size(), success ? "PASS" : "FAIL");
    return success;
}

static bool check_request_cache(ggml_backend_t backend) {
    auto context = ggml_init({2 * 1024 * 1024, nullptr, true});
    auto input = ggml_new_tensor_2d(context, GGML_TYPE_F32, 5376, 85);
    auto output = input;
    for (int layer = 0; layer < 24; ++layer) {
        output = ggml_scale(context, output, layer % 2 == 0 ? 2.f : 0.5f);
    }
    auto graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, output);
    auto buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    std::vector<float> values(ggml_nelements(input), 7.f);
    ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
    bool success;
    {
        sd::RunnerCache cache(backend);
        cache.stage("refined", output);
        success = ggml_backend_graph_compute_async(backend, graph) == GGML_STATUS_SUCCESS;
        success = cache.capture(graph) == GGML_STATUS_SUCCESS && success;
        cache.graph_end(true);
        ggml_backend_tensor_get(cache.get("refined"), values.data(), 0, values.size() * sizeof(float));
        for (float value : values) {
            success = value == 7.f && success;
        }
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    std::printf("request cache compute-stream ordering: %s\n", success ? "PASS" : "FAIL");
    return success;
}

int main() {
    auto backend = ggml_backend_cuda_init(0);
    if (backend == nullptr) {
        return 1;
    }
    bool success = check_request_cache(backend);
    for (bool reuse : {false, true}) {
        success = check_cache(backend, 5376, 85, reuse) && success;
        success = check_cache(backend, 128, 37, reuse) && success;
    }
    ggml_backend_free(backend);
    return success ? 0 : 1;
}
