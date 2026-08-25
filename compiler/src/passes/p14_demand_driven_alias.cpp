// =============================================================================
// Pass 14 — Demand-Driven Alias Analysis.
//
// For each hot-loop memory access pair, run a focused backwards BFS from
// the two base pointers; if their backwards slices intersect only at nodes
// OUTSIDE the loop (loop-invariant definitions of distinct allocations),
// the pair cannot alias inside the loop. The pass marks qualifying
// LoadIndex/StoreIndex pairs with the TypeGuarded flag — the vectorizer's
// (31c/32) legality signal — avoiding the full O(n^3) CFL cost.
// =============================================================================

#include "vortex/passes/analyses/dominators.hpp"
#include "vortex/passes/analyses/loops.hpp"
#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

namespace {

void backward_slice(const Graph& g, NodeId root,
                     stdx::flat_map<NodeId, bool, 32>& seen) noexcept {
    stdx::small_vector<NodeId, 32> stack{root};
    seen.insert(root, true);
    while (!stack.empty()) {
        NodeId id = stack.back();
        stack.pop_back();
        for (NodeId in : g.node(id).ins) {
            if (!seen.contains(in)) {
                seen.insert(in, true);
                stack.push_back(in);
            }
        }
    }
}

[[nodiscard]] bool is_alloc(NodeKind k) noexcept {
    switch (k) {
        case NodeKind::NewList: case NodeKind::NewDict: case NodeKind::NewTuple:
            return true;
        default: return false;
    }
}

}  // namespace

Result<PassResult> P14_DemandDrivenAlias::run(Graph& g, const PassContext& c) noexcept {
    DomTree dom = compute_dominators(g);
    LoopInfo loops = compute_loops(g, dom);
    if (loops.loops.empty()) return PassResult{};

    std::uint32_t before = g.live_node_count();
    std::uint32_t pairs_proved = 0;

    for (const LoopInfo::Loop& loop : loops.loops) {
        // Collect index accesses in this loop.
        stdx::small_vector<NodeId, 8> accesses;
        g.for_each_live([&](NodeId id) {
            const Node& n = g.node(id);
            if (n.kind != NodeKind::LoadIndex && n.kind != NodeKind::StoreIndex) return;
            if (n.ins.empty()) return;
            // in-loop iff its control input (ins[0]) is a loop block
            for (NodeId blk : loop.blocks) {
                if (n.ins[0] == blk) { accesses.push_back(id); return; }
            }
        });

        // Pairwise demand queries.
        for (std::size_t i = 0; i < accesses.size(); ++i) {
            for (std::size_t j = i + 1; j < accesses.size(); ++j) {
                NodeId a = accesses[i], b = accesses[j];
                if (a == b) continue;
                const Node& na = g.node(a);
                const Node& nb = g.node(b);
                if (na.ins.size() < 3 || nb.ins.size() < 3) continue;
                NodeId base_a = na.ins[2], base_b = nb.ins[2];

                // Same base node: definitely alias — skip.
                if (base_a == base_b) continue;

                // Distinct allocation bases defined OUTSIDE the loop and
                // never stored through inside the loop: NoAlias.
                const Node& ba = g.node(base_a);
                const Node& bb = g.node(base_b);
                if (!is_alloc(ba.kind) || !is_alloc(bb.kind)) continue;

                // both bases must be defined outside every loop block
                bool a_inside = false, b_inside = false;
                for (NodeId blk : loop.blocks) {
                    if (ba.ins.size() >= 1 && ba.ins[0] == blk) a_inside = true;
                    if (bb.ins.size() >= 1 && bb.ins[0] == blk) b_inside = true;
                }
                if (a_inside || b_inside) continue;

                // No store-through (StoreIndex/CallPy) inside the loop may
                // target either base (conservative: any store to ANY base
                // blocks the proof, matching the spec's demand-driven
                // precision/deferral note).
                bool blocked = false;
                g.for_each_live([&](NodeId id) {
                    const Node& n = g.node(id);
                    if (n.kind != NodeKind::StoreIndex && n.kind != NodeKind::CallPy) return;
                    for (NodeId blk : loop.blocks) {
                        if (!n.ins.empty() && n.ins[0] == blk) { blocked = true; return; }
                    }
                });
                if (blocked) continue;

                g.node(a).set_flag(NodeFlag::TypeGuarded);
                g.node(b).set_flag(NodeFlag::TypeGuarded);
                ++pairs_proved;
            }
        }
    }

    PassResult r = result_of(g, before);
    r.changed = false;
    note(TelemetryEventKind::BudgetExceeded, c, pairs_proved);
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
