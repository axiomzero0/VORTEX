// =============================================================================
// Pass 28 — Speculative LICM.
//
// Hoists may-effect computations guarded by an in-loop runtime check:
// for a Python-level op with loop-invariant operands at a PGO-hot site,
// a NoOverflow guard node referencing the operands is emitted; failure
// deoptimizes (Rule 4). Profile mode only — speculation requires PGO.
// =============================================================================

#include "vortex/passes/analyses/dominators.hpp"
#include "vortex/passes/analyses/loops.hpp"
#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

Result<PassResult> P28_SpeculativeLICM::run(Graph& g, const PassContext& c) noexcept {
    if (!c.is_profiled()) return PassResult{};
    DomTree dom = compute_dominators(g);
    LoopInfo loops = compute_loops(g, dom);
    if (loops.loops.empty()) return PassResult{};

    std::uint32_t before = g.live_node_count();
    std::uint32_t guarded_hoists = 0;

    for (const LoopInfo::Loop& loop : loops.loops) {
        g.for_each_live([&](NodeId id) {
            Node& n = g.node(id);
            if (n.kind != NodeKind::PyBinary) return;
            if (n.ins.empty()) return;
            bool in_loop = false;
            for (NodeId blk : loop.blocks) {
                if (n.ins[0] == blk) { in_loop = true; break; }
            }
            if (!in_loop) return;
            if (n.pgo_count < cfg::tier2_entry_heat) return;
            bool inv = true;
            for (std::uint32_t i = 2; i < n.ins.size() && inv; ++i) {
                const Node& dep = g.node(n.ins[i]);
                if (!dep.ins.empty()) {
                    for (NodeId blk : loop.blocks) {
                        if (dep.ins[0] == blk) { inv = false; break; }
                    }
                }
            }
            if (!inv) return;
            NodeId guard = g.create(NodeKind::Guard);
            Node& gn = g.node(guard);
            gn.subop = static_cast<std::uint16_t>(GuardKind::NoOverflow);
            gn.set_flag(NodeFlag::Speculative);
            FrameState fs;
            fs.code_unit_id = c.code_unit_id;
            fs.values.push_back(n.ins[2]);
            if (n.ins.size() >= 4) fs.values.push_back(n.ins[3]);
            gn.aux1 = g.add_frame_state(fs);
            gn.set_flag(NodeFlag::OnEffectChain);
            // PASS-9 fix: the guard needs BOTH operands as inputs so the
            // deopt stub can reconstruct the operands' values at the
            // safepoint. The previous code added only n.ins[2] (operand a),
            // missing n.ins[3] (operand b) — so the deopt path couldn't
            // replay the operation on Tier-0. Also wire the PyBinary's
            // control input to the guard so the guard actually gates the
            // operation (without this, the PyBinary stays in the loop and
            // the guard is decorative).
            g.add_input(guard, n.ins[0]);   // control
            g.add_input(guard, n.ins[2]);   // operand a
            if (n.ins.size() >= 4) {
                g.add_input(guard, n.ins[3]);   // operand b
            }
            // Rewire the PyBinary's control input to the guard, so the
            // guard gates the operation. (Memory input stays the same —
            // the guard is a pure control gate.)
            if (n.ins.size() >= 1) {
                g.set_input(id, 0, guard);
            }
            ++guarded_hoists;
        });
    }

    PassResult r = result_of(g, before);
    r.changed = guarded_hoists > 0;
    note(TelemetryEventKind::SafepointPatched, c, guarded_hoists);
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
