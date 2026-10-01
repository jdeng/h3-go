#include "core/ggml_graph_cut.h"
#include "core/segment_weight_pipeline.h"
#include "device_residency_manager.h"
#include <algorithm>
#include <cstdio>
#include <memory>
#include <unordered_map>

struct ResidencyProbe : DeviceResidencyManager {
    std::vector<std::vector<ggml_tensor*>> order;
    std::vector<ggml_tensor*> protected_weights;

    bool segmented_compute_enabled() const override { return true; }
    bool prefetch_enabled() const override { return false; }
    void set_workspace_reclaimer(uintptr_t, std::function<bool()>) override {}
    void remove_runtime_owner(uintptr_t) override {}
    bool fits_compute_backend_capacity(const DeviceMemoryRequest&, const std::vector<ggml_tensor*>&) const override { return true; }
    bool assign_compute_backend(const std::vector<ggml_tensor*>&, ggml_backend_t) override { return true; }
    bool prepare_params(const std::vector<ggml_tensor*>&) override { return true; }
    void release_compute_backend_params(const std::vector<ggml_tensor*>&) override {}
    void evict_compute_backend_params(const std::vector<ggml_tensor*>&) override {}
    WeightResidencyInfo inspect_compute_backend_params(const std::vector<ggml_tensor*>&) const override { return {}; }
    void update_runtime_residency(uintptr_t, ggml_backend_t, size_t) override {}
    bool ensure_compute_backend_capacity(const DeviceMemoryRequest&, const std::vector<ggml_tensor*>&,
                                         const std::vector<std::vector<ggml_tensor*>>& preferred,
                                         const std::vector<ggml_tensor*>& protected_params) override {
        order = preferred;
        protected_weights = protected_params;
        return true;
    }
    WeightPrefetchResult prefetch_params(uintptr_t, const std::vector<ggml_tensor*>&) override { return WeightPrefetchResult::Unsupported; }
    bool activate_prefetched_params(uintptr_t, const std::vector<ggml_tensor*>&) override { return false; }
    void clear_prefetched_params(uintptr_t) override {}
};

int main() {
    auto context = ggml_init({1024 * 1024, nullptr, true});
    auto graph = ggml_new_graph(context);
    std::vector<ggml_tensor*> weights;
    std::unordered_set<const ggml_tensor*> parameters;
    for (int index = 0; index < 5; ++index) {
        weights.push_back(ggml_new_tensor_1d(context, GGML_TYPE_F32, 1));
        parameters.insert(weights.back());
        ggml_build_forward_expand(graph, weights.back());
    }
    sd::ggml_graph_cut::Plan plan;
    plan.segments.resize(4);
    for (size_t segment = 0; segment < plan.segments.size(); ++segment) {
        for (int leaf = 0; leaf < sd::ggml_graph_cut::leaf_count(graph); ++leaf) {
            auto* parameter = sd::ggml_graph_cut::leaf_tensor(graph, leaf);
            if (parameter == weights[segment] || (segment % 2 == 0 && parameter == weights[4])) {
                sd::ggml_graph_cut::Segment::InputRef reference;
                reference.type = sd::ggml_graph_cut::Segment::INPUT_PARAM;
                reference.leaf_index = leaf;
                plan.segments[segment].input_refs.push_back(reference);
            }
        }
    }
    auto probe = std::make_shared<ResidencyProbe>();
    bool success = true;
    for (bool cyclic : {false, true}) {
        sd::SegmentWeightPipeline pipeline(probe, nullptr, 1, graph, plan, parameters, false, cyclic);
        for (size_t current = 0; current < plan.segments.size(); ++current) {
            success = pipeline.ensure_segment_capacity(current, {DeviceMemoryRequest{}}) && success;
            success = success && probe->protected_weights == pipeline.params(current);
            if (!cyclic) {
                for (size_t index = 0; index < plan.segments.size(); ++index) {
                    success = success && probe->order[index] == pipeline.params(plan.segments.size() - index - 1);
                }
                continue;
            }
            std::unordered_map<ggml_tensor*, size_t> next_use;
            for (size_t distance = 1; distance <= plan.segments.size(); ++distance) {
                for (auto* parameter : pipeline.params((current + distance) % plan.segments.size())) {
                    next_use.emplace(parameter, distance);
                }
            }
            std::unordered_set<ggml_tensor*> seen;
            size_t previous_distance = plan.segments.size();
            for (const auto& group : probe->order) {
                for (auto* parameter : group) {
                    success = success && seen.insert(parameter).second && next_use.at(parameter) <= previous_distance;
                    previous_distance = next_use.at(parameter);
                }
            }
            success = success && seen.size() == weights.size();
        }
    }
    ggml_free(context);
    std::printf("cyclic eviction, shared weights, wraparound, protected weights, legacy fallback: %s\n", success ? "PASS" : "FAIL");
    return success ? 0 : 1;
}
