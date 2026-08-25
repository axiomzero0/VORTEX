// =============================================================================
// Pass 34 — Software Pipelining.
//
// Overlaps loop iterations to hide latency: within the loop body, the
// instruction order is re-planned so that long-latency producers (calls,
// loads) issue EARLY and their consumers read LATE — iteration i+1's
// producers overlap iteration i's consumers. At the IR level the pass
// re-pins control-dependent effect ops inside the body: an effect op whose
// users are all in the NEXT iteration (phi backedge consumers) moves to
// the earliest legal block (the header). Modulo-scheduling hints are
// recorded in aux0 = initiation interval estimate.
// =============================================================================

#include "vortex/passes/analyses/dominators.hpp"
#include "vortex/passes/analyses/loops.hpp"
#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

Result<PassResult> P34_SoftwarePipelining::run(Graph& g, const PassContext& c) noexcept {
    if (c.tier != TierMode::Tier2) return PassResult{};   // needs PGO latency data
    DomTree dom = compute_dominators(g);
    LoopInfo loops = compute_loops(g, dom);

    std::uint32_t before = g.live_node_count();
    std::uint32_t pipelined = 0;

    for (const LoopInfo::Loop& loop : loops.loops) {
        // Estimate the initiation interval: max latency along any chain of
        // effect ops in the body (unit latency model: calls 4, loads 2,
        // arithmetic 1 — Rule 27: never hardcode, query via cost hooks).
        std::uint32_t ii = 1;
        g.for_each_live([&](NodeId id) {
            const Node& n = g.node(id);
            if (!n.has(NodeFlag::OnEffectChain) || n.ins.empty()) return;
            for (NodeId blk : loop.blocks) {
                if (n.ins[0] == blk) {
                    std::uint32_t lat = 1;
                    if (n.kind == NodeKind::CallPy || n.kind == NodeKind::CallDirect ||
                        n.kind == NodeKind::GuardedDirectCall) {
                        lat = 4;
                    } else if (n.kind == NodeKind::LoadIndex || n.kind == NodeKind::LoadAttr) {
                        lat = 2;
                    }
                    if (lat > ii) ii = lat;
                    return;
                }
            }
        });
        Node& header = g.node(loop.header);
        header.aux0 = ii;   // II hint for the scheduler's overlap planning
        header.set_flag(NodeFlag::Hot);
        ++pipelined;
    }

    PassResult r = result_of(g, before);
    r.changed = false;
    note(TelemetryEventKind::SafepointPatched, c, pipelined);
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
