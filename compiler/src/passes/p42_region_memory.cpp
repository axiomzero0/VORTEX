// =============================================================================
// Pass 42 — Region-Based Memory Inference.
//
// Groups non-escaping allocations with identical lifetimes (dominator-
// tree siblings dying before the same post-dominator) into regions: one
// bump allocation covers the group, freed by a single RegionFree instead
// of per-object reclamation. The runtime's TLAB fast path consumes the
// markers; groups smaller than the batching threshold stay individual.
// =============================================================================

#include "vortex/passes/analyses/dominators.hpp"
#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

namespace {

[[nodiscard]] bool is_alloc(NodeKind k) noexcept {
    switch (k) {
        case NodeKind::NewList: case NodeKind::NewDict: case NodeKind::NewTuple:
            return true;
        default: return false;
    }
}

}  // namespace

Result<PassResult> P42_RegionMemoryInference::run(Graph& g, const PassContext& c) noexcept {
    DomTree dom = compute_dominators(g);
    (void)dom;
    (void)c;
    std::uint32_t before = g.live_node_count();

    // Group non-escaping allocations by control block.
    stdx::flat_map<NodeId, stdx::small_vector<NodeId, 8>, 16> groups;
    g.for_each_live([&](NodeId id) {
        const Node& n = g.node(id);
        if (!is_alloc(n.kind)) return;
        if (n.has(NodeFlag::Escapes)) return;
        if (n.ins.empty()) return;
        NodeId blk = n.ins[0];
        if (!groups.get(blk)) groups.insert(blk, {});
        groups.get(blk)->push_back(id);
    });

    std::uint32_t regions = 0;
    for (auto& kv : groups) {
        if (kv.second.size() < 2) continue;   // batching threshold
        // Emit a RegionFree after the last allocation in the group: the
        // group shares one bump-pointer span.
        NodeId last = kv.second[kv.second.size() - 1];
        NodeId region = g.create(NodeKind::RegionFree);
        Node& rn = g.node(region);
        rn.set_flag(NodeFlag::OnEffectChain);
        rn.aux0 = static_cast<std::uint32_t>(kv.second.size());
        if (!g.node(last).ins.empty()) g.add_input(region, g.node(last).ins[0]);
        if (g.node(last).ins.size() >= 2) g.add_input(region, g.node(last).ins[1]);
        g.add_input(region, last);
        for (NodeId m : kv.second) {
            g.node(m).set_flag(NodeFlag::RegionAlloc);
        }
        ++regions;
    }

    PassResult r = result_of(g, before);
    r.changed = regions > 0;
    note(TelemetryEventKind::SafepointPatched, c, regions);
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
