// =============================================================================
// Pass 26 — Interprocedural Constant Propagation (summary-based).
//
// For each direct call whose arguments are all constants, the call node's
// FrameState records the constant argument values; the runtime's direct-
// call path binds them, letting the next SCCP round fold the callee body.
// Summary form avoids exponential cloning.
// =============================================================================

#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

Result<PassResult> P26_IPCP::run(Graph& g, const PassContext& c) noexcept {
    if (c.tier == TierMode::Tier1) return PassResult{};
    std::uint32_t before = g.live_node_count();
    std::uint32_t summaries = 0;

    g.for_each_live([&](NodeId id) {
        Node& n = g.node(id);
        if (n.kind != NodeKind::GuardedDirectCall && n.kind != NodeKind::CallDirect) return;
        if (n.ins.size() < 4) return;

        bool all_const = true;
        for (std::uint32_t i = 3; i < n.ins.size() && all_const; ++i) {
            NodeKind k = g.node(n.ins[i]).kind;
            all_const = k == NodeKind::ConstInt || k == NodeKind::ConstFloat ||
                        k == NodeKind::ConstPy;
        }
        if (!all_const) return;

        FrameState fs;
        fs.code_unit_id = c.code_unit_id;
        fs.bytecode_offset = 0;
        for (std::uint32_t i = 3; i < n.ins.size(); ++i) {
            fs.values.push_back(n.ins[i]);
            fs.kinds.push_back(0);
        }
        n.aux1 = g.add_frame_state(fs);
        n.set_flag(NodeFlag::Speculative);
        ++summaries;
    });

    PassResult r = result_of(g, before);
    r.changed = false;
    note(TelemetryEventKind::SafepointPatched, c, summaries);
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
