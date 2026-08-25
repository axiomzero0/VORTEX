// =============================================================================
// Pass 30 — Loop Unrolling with PGO-Directed Trip Counts.
//
// Unrolls loops with provably-constant trip counts (range-loops with
// constant bounds). Factor capped by cfg::unroll_max_factor, trip count
// and the node budget (Rule 10: growing passes run in guarded fixpoints).
// The decision is recorded on the loop header (aux0 = factor) for the
// scheduler's bytecode-level body duplication — IR-level duplication
// would double phis without register context.
// =============================================================================

#include "vortex/passes/analyses/dominators.hpp"
#include "vortex/passes/analyses/loops.hpp"
#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

namespace {

[[nodiscard]] bool constant_trip_count(const Graph& g, NodeId loop_header,
                                       std::int64_t& trips) noexcept {
    NodeId check = invalid_node;
    g.for_each_live([&](NodeId id) {
        const Node& n = g.node(id);
        if (n.kind == NodeKind::GetIterCheck && !n.ins.empty() && n.ins[0] == loop_header) {
            check = id;
        }
    });
    if (check == invalid_node || g.node(check).ins.size() < 3) return false;
    NodeId it = g.node(check).ins[2];
    const Node& iter = g.node(it);
    if (iter.kind != NodeKind::Iter || iter.ins.size() < 3) return false;
    NodeId range_call = iter.ins[2];
    const Node& rc = g.node(range_call);
    if (rc.kind != NodeKind::CallPy || rc.ins.size() < 4) return false;
    const Node& callee = g.node(rc.ins[2]);
    if (callee.kind != NodeKind::LoadGlobal) return false;
    const Node& lo = g.node(rc.ins[3]);
    if (rc.ins.size() < 5) {
        if (lo.kind != NodeKind::ConstInt) return false;
        trips = lo.const_value.as.i;
        return trips >= 0 && trips <= 1024;
    }
    const Node& hi = g.node(rc.ins[4]);
    if (lo.kind != NodeKind::ConstInt || hi.kind != NodeKind::ConstInt) return false;
    trips = hi.const_value.as.i - lo.const_value.as.i;
    return trips >= 0 && trips <= 1024;
}

}  // namespace

Result<PassResult> P30_LoopUnrolling::run(Graph& g, const PassContext& c) noexcept {
    if (c.tier == TierMode::Tier1) return PassResult{};
    DomTree dom = compute_dominators(g);
    LoopInfo loops = compute_loops(g, dom);
    if (loops.loops.empty()) return PassResult{};

    std::uint32_t before = g.live_node_count();
    std::uint32_t unrolled = 0;

    for (const LoopInfo::Loop& loop : loops.loops) {
        if (budget_exceeded(g, c)) {
            note(TelemetryEventKind::BudgetExceeded, c);
            break;
        }
        std::int64_t trips = 0;
        if (!constant_trip_count(g, loop.header, trips)) continue;

        std::uint32_t factor = cfg::unroll_max_factor;
        if (static_cast<std::int64_t>(factor) > trips) factor = static_cast<std::uint32_t>(trips);
        if (factor < 2) continue;

        Node& header = g.node(loop.header);
        header.aux0 = factor;
        header.set_flag(NodeFlag::Hot);
        ++unrolled;
    }

    PassResult r = result_of(g, before);
    r.changed = false;
    note(TelemetryEventKind::SafepointPatched, c, unrolled);
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
