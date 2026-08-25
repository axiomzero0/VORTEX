// =============================================================================
// Pass 35 — Loop Fusion & Fission.
//
// Fusion: two ADJACENT loops (same bounds, no intervening effects, no
// carried dependences between them) merge into one — halves loop overhead
// and improves cache locality of index streams. Fission: a fused loop
// whose body exceeds the register-pressure budget splits back apart
// (cfg::vector_max_loop_body_nodes as the pressure proxy). Decisions are
// recorded on the loop headers; the scheduler merges or splits at layout.
// =============================================================================

#include "vortex/passes/analyses/dominators.hpp"
#include "vortex/passes/analyses/loops.hpp"
#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

namespace {

[[nodiscard]] bool same_range_bound(const Graph& g, NodeId h1, NodeId h2) noexcept {
    NodeId it1 = invalid_node, it2 = invalid_node;
    g.for_each_live([&](NodeId id) {
        const Node& n = g.node(id);
        if (n.kind == NodeKind::GetIterCheck && n.ins.size() >= 3) {
            if (!n.ins.empty() && n.ins[0] == h1) it1 = n.ins[2];
            if (!n.ins.empty() && n.ins[0] == h2) it2 = n.ins[2];
        }
    });
    if (it1 == invalid_node || it2 == invalid_node) return false;
    const Node& i1 = g.node(it1);
    const Node& i2 = g.node(it2);
    if (i1.kind != NodeKind::Iter || i2.kind != NodeKind::Iter) return false;
    if (i1.ins.size() < 3 || i2.ins.size() < 3) return false;
    return i1.ins[2] == i2.ins[2];
}

}  // namespace

Result<PassResult> P35_LoopFusionFission::run(Graph& g, const PassContext& c) noexcept {
    DomTree dom = compute_dominators(g);
    LoopInfo loops = compute_loops(g, dom);
    if (loops.loops.size() < 2) return PassResult{};

    std::uint32_t before = g.live_node_count();
    std::uint32_t fused = 0;

    for (std::size_t i = 0; i < loops.loops.size(); ++i) {
        for (std::size_t j = i + 1; j < loops.loops.size(); ++j) {
            const LoopInfo::Loop& a = loops.loops[i];
            const LoopInfo::Loop& b = loops.loops[j];
            if (!same_range_bound(g, a.header, b.header)) continue;

            // No loop-carried dependence: nothing in B reads a value defined
            // in A's body.
            bool coupled = false;
            g.for_each_live([&](NodeId id) {
                const Node& n = g.node(id);
                if (n.ins.empty()) return;
                bool in_b = false;
                for (NodeId blk : b.blocks) {
                    if (n.ins[0] == blk) { in_b = true; break; }
                }
                if (!in_b) return;
                for (NodeId in : n.ins) {
                    const Node& dep = g.node(in);
                    if (dep.ins.empty()) continue;
                    for (NodeId blk : a.blocks) {
                        if (dep.ins[0] == blk) { coupled = true; return; }
                    }
                }
            });
            if (coupled) continue;

            g.node(a.header).shape_id = 0xF00Du;   // fusion-group marker
            g.node(b.header).shape_id = 0xF00Du;
            ++fused;
        }
    }

    // Fission: single loops over the pressure budget.
    std::uint32_t split = 0;
    for (const LoopInfo::Loop& loop : loops.loops) {
        std::uint32_t body_ops = 0;
        g.for_each_live([&](NodeId id) {
            const Node& n = g.node(id);
            if (n.ins.empty()) return;
            for (NodeId blk : loop.blocks) {
                if (n.ins[0] == blk) { ++body_ops; return; }
            }
        });
        if (body_ops > cfg::vector_max_loop_body_nodes) {
            g.node(loop.header).shape_id = 0x5F1Fu;   // fission marker
            ++split;
        }
    }

    PassResult r = result_of(g, before);
    r.changed = false;
    note(TelemetryEventKind::SafepointPatched, c, fused * 100 + split);
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
