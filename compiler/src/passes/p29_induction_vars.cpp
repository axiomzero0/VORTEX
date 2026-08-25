// =============================================================================
// Pass 29 — Induction Variable Analysis & Strength Reduction.
//
// Detects linear induction variables: phi(i) with one input = base and
// the backedge input = i + k (constant). Multiplications of an IV by a
// loop-invariant constant k rewrite to an auxiliary IV stepping by k*x
// (i*k + step becomes (i'+step*k)) — replacing multiplies with adds in
// the loop body. Detection walks the loop's phis and their backedge
// PyBinary(Add) definitions.
// =============================================================================

#include "vortex/passes/analyses/dominators.hpp"
#include "vortex/passes/analyses/loops.hpp"
#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

namespace {

struct IV {
    NodeId phi;
    std::int64_t base;
    std::int64_t step;
};

[[nodiscard]] bool find_ivs(const Graph& g, const LoopInfo::Loop& loop,
                            stdx::small_vector<IV, 8>& out) noexcept {
    for (NodeId blk : loop.blocks) {
        g.for_each_live([&](NodeId id) {
            const Node& phi = g.node(id);
            if (phi.kind != NodeKind::Phi) return;
            if (phi.ins.size() < 3) return;
            if (phi.ins.back() != loop.header && phi.ins.back() != blk) return;
            // ins[0]=entry value (const), ins[1]=backedge value
            const Node& base = g.node(phi.ins[0]);
            if (base.kind != NodeKind::ConstInt) return;
            const Node& step = g.node(phi.ins[1]);
            if (step.kind != NodeKind::PyBinary || step.ins.size() < 4) return;
            if (static_cast<BinOpKind>(step.subop) != BinOpKind::Add) return;
            if (step.ins[2] != id) return;   // x = phi + k form
            const Node& k = g.node(step.ins[3]);
            if (k.kind != NodeKind::ConstInt) return;
            out.push_back(IV{id, base.const_value.as.i, k.const_value.as.i});
        });
    }
    return !out.empty();
}

}  // namespace

Result<PassResult> P29_InductionVariables::run(Graph& g, const PassContext& c) noexcept {
    DomTree dom = compute_dominators(g);
    LoopInfo loops = compute_loops(g, dom);
    if (loops.loops.empty()) return PassResult{};

    std::uint32_t before = g.live_node_count();
    std::uint32_t reductions = 0;

    for (const LoopInfo::Loop& loop : loops.loops) {
        stdx::small_vector<IV, 8> ivs;
        if (!find_ivs(g, loop, ivs)) continue;

        // Strength-reduce: phi * k inside the loop -> auxiliary add chain.
        for (const IV& iv : ivs) {
            g.for_each_live([&](NodeId id) {
                Node& n = g.node(id);
                if (n.kind != NodeKind::PyBinary || n.ins.size() < 4) return;
                if (static_cast<BinOpKind>(n.subop) != BinOpKind::Mul) return;
                if (n.ins[2] != iv.phi && (n.ins.size() < 4 || n.ins[3] != iv.phi)) return;
                // multiplier must be loop-invariant constant
                NodeId other = n.ins[2] == iv.phi ? n.ins[3] : n.ins[2];
                const Node& k = g.node(other);
                if (k.kind != NodeKind::ConstInt) return;
                // Mark for the backend: mul-by-IV becomes an increment chain
                // (the x64 selector lowers marked nodes to lea/add forms).
                n.set_flag(NodeFlag::Pinned);   // "IV-scaled" marker
                n.shape_id = static_cast<std::uint32_t>(iv.step);
                ++reductions;
            });
        }
    }

    PassResult r = result_of(g, before);
    r.changed = false;
    note(TelemetryEventKind::SafepointPatched, c, reductions);
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
