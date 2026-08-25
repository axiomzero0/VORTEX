// =============================================================================
// Pass 38 — GIL Hoisting & Batching.
//
// Identifies regions of pure native computation (no MayCall, no Python
// object allocation, no effect ops beyond unboxed arithmetic) and marks
// them NoGIL: the runtime acquires the interpreter lock once per region
// instead of per op. For fully-unboxed loops (post pass-47 chains) the
// GIL acquisition is removed entirely — pure math on machine registers
// cannot touch interpreter state.
// =============================================================================

#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

Result<PassResult> P38_GILHoisting::run(Graph& g, const PassContext& c) noexcept {
    (void)c;
    std::uint32_t before = g.live_node_count();
    std::uint32_t regions = 0;

    // A maximal run of consecutive (node-id order) unboxed, effect-free
    // operations forms a NoGIL region.
    std::uint32_t run_length = 0;
    stdx::small_vector<NodeId, 16> run_members;
    g.for_each_live([&](NodeId id) {
        const Node& n = g.node(id);
        bool pure_native = !n.has(NodeFlag::OnEffectChain) && !n.has(NodeFlag::MayCall) &&
                           (n.has(NodeFlag::Unboxed) || n.kind == NodeKind::ConstInt ||
                            n.kind == NodeKind::ConstFloat);
        if (pure_native) {
            ++run_length;
            run_members.push_back(id);
        } else {
            if (run_length >= 4) {   // batching threshold: short runs pay
                                    // more in boundary bookkeeping
                for (NodeId m : run_members) {
                    g.node(m).set_flag(NodeFlag::NoGIL);
                }
                ++regions;
            }
            run_length = 0;
            run_members.clear();
        }
    });
    if (run_length >= 4) {
        for (NodeId m : run_members) {
            g.node(m).set_flag(NodeFlag::NoGIL);
        }
        ++regions;
    }

    PassResult r = result_of(g, before);
    r.changed = false;
    note(TelemetryEventKind::SafepointPatched, c, regions);
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
