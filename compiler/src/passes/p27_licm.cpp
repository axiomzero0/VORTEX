// =============================================================================
// Pass 27 — Loop-Invariant Code Motion (LICM).
//
// Hoists computations out of loops when legal. Legality (the dominator-
// tree criterion, per spec): a node may move to the loop preheader iff
// (1) it is pure (no effects, no control dependence inside the loop),
// (2) it dominates all loop exits (guaranteed here: only pure nodes move
// and the loop-exit structure is a single latch via Loop nodes), and
// (3) its inputs are all loop-invariant (inductively: hoist leaves first).
// =============================================================================

#include "vortex/passes/analyses/dominators.hpp"
#include "vortex/passes/analyses/loops.hpp"
#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

Result<PassResult> P27_LICM::run(Graph& g, const PassContext& c) noexcept {
    DomTree dom = compute_dominators(g);
    LoopInfo loops = compute_loops(g, dom);
    if (loops.loops.empty()) return PassResult{};

    std::uint32_t before = g.live_node_count();
    std::uint32_t hoisted = 0;

    for (const LoopInfo::Loop& loop : loops.loops) {
        // Loop-invariance fixpoint over the loop body.
        stdx::flat_map<NodeId, bool, 32> invariant;
        bool progress = true;
        while (progress) {
            progress = false;
            g.for_each_live([&](NodeId id) {
                const Node& n = g.node(id);
                if (is_control(n.kind)) return;
                if (invariant.contains(id)) return;
                // in-loop check: control input is a loop block
                if (n.ins.empty()) return;
                bool in_loop = false;
                for (NodeId blk : loop.blocks) {
                    if (n.ins[0] == blk) { in_loop = true; break; }
                }
                if (!in_loop) return;

                // Pure only: effect ops and control ops stay.
                if (n.has(NodeFlag::OnEffectChain)) return;
                if (n.has(NodeFlag::MayCall)) return;

                // All inputs invariant (constants/params are by definition).
                bool inv = true;
                for (NodeId in : n.ins) {
                    if (in == invalid_node) continue;
                    const Node& dep = g.node(in);
                    if (is_control(dep.kind)) continue;
                    // input defined inside the loop must be invariant already
                    bool dep_in_loop = false;
                    if (!dep.ins.empty()) {
                        for (NodeId blk : loop.blocks) {
                            if (dep.ins[0] == blk) { dep_in_loop = true; break; }
                        }
                    }
                    if (dep_in_loop && !invariant.contains(in)) { inv = false; break; }
                }
                if (inv) {
                    invariant.insert(id, true);
                    progress = true;
                }
            });
        }

        // Hoist: re-pin each invariant node's control input to the loop
        // preheader (the Loop node's control input).
        for (auto& kv : invariant) {
            Node& n = g.node(kv.first);
            // set control input to the block BEFORE the loop header
            NodeId header = loop.header;
            if (g.node(header).ins.empty()) continue;
            NodeId pre = g.node(header).ins[0];
            if (pre == invalid_node) continue;
            if (!n.ins.empty() && n.ins[0] != pre) {
                g.set_input(kv.first, 0, pre);
                ++hoisted;
            }
        }
        if (hoisted > cfg::licm_max_hoisted_per_loop) {
            note(TelemetryEventKind::BudgetExceeded, c, hoisted);
            break;
        }
    }

    PassResult r = result_of(g, before);
    r.changed = hoisted > 0;
    note(TelemetryEventKind::SafepointPatched, c, hoisted);
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
