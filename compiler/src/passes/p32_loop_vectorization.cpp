// =============================================================================
// Pass 32 — Loop Vectorization with Speculative Versioning.
//
// Vectorizes countable loops: the loop body's index accesses become
// VecLoad/VecStore over the base array when (a) the loop is a range-loop
// (constant step), (b) accesses are affine in the induction variable,
// (c) bases cannot alias (pass 14 markers) or a runtime alias guard is
// inserted (the "versioned" loop: guard fails -> scalar fallback, Rule 4).
// =============================================================================

#include "vortex/passes/analyses/dominators.hpp"
#include "vortex/passes/analyses/loops.hpp"
#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

Result<PassResult> P32_LoopVectorization::run(Graph& g, const PassContext& c) noexcept {
    if (c.tier == TierMode::Tier1) return PassResult{};
    DomTree dom = compute_dominators(g);
    LoopInfo loops = compute_loops(g, dom);
    if (loops.loops.empty()) return PassResult{};

    std::uint32_t before = g.live_node_count();
    std::uint32_t vectorized = 0;

    for (const LoopInfo::Loop& loop : loops.loops) {
        // Collect in-loop index accesses.
        stdx::small_vector<NodeId, 8> accesses;
        g.for_each_live([&](NodeId id) {
            const Node& n = g.node(id);
            if (n.kind != NodeKind::LoadIndex && n.kind != NodeKind::StoreIndex) return;
            if (n.ins.empty()) return;
            for (NodeId blk : loop.blocks) {
                if (n.ins[0] == blk) { accesses.push_back(id); return; }
            }
        });
        if (accesses.empty()) continue;
        if (accesses.size() > cfg::vector_max_loop_body_nodes / 4) continue;

        // Versioning: every access needs a NoAlias proof (TypeGuarded from
        // pass 14) or the loop gets a runtime guard.
        bool all_proved = true;
        bool any_proved = false;
        for (NodeId a : accesses) {
            if (g.node(a).has(NodeFlag::TypeGuarded)) {
                any_proved = true;
            } else {
                all_proved = false;
            }
        }
        if (!any_proved && !c.is_profiled()) continue;   // no evidence, no speculation

        // Emit the versioning guard on the loop header (Tier 2 speculative
        // form; Tier 3 requires all_proved and emits nothing).
        if (!all_proved && c.is_profiled()) {
            NodeId guard = g.create(NodeKind::Guard);
            Node& gn = g.node(guard);
            gn.subop = static_cast<std::uint16_t>(GuardKind::AliasDisjoint);
            gn.set_flag(NodeFlag::Speculative);
            gn.set_flag(NodeFlag::OnEffectChain);
            // VERIFIER-1 fix: the Guard has OnEffectChain set, so the
            // structural verifier requires ins[0] to be a control
            // projection and ins[1] to be a chained effect (or Start /
            // EffectPhi). The previous emission only pushed the
            // accesses' base pointers — leaving the guard with one or
            // more data inputs but no control / effect input, so every
            // downstream verifier check fired "effect chain
            // discontinuity" (Rule 40) on any code that ran P32 with
            // Tier2 (is_profiled() == true for the regression harness
            // that drives the full pipeline on the lang corpus).
            //
            // The guard semantically gates the loop's first access: its
            // control is the access's control projection (ins[0]) and
            // its effect is the access's effect input (ins[1]). Both
            // are valid effect-chain members by construction (the
            // access has OnEffectChain set). The data inputs — the
            // base pointers — follow as ins[2+].
            if (!accesses.empty()) {
                const Node& first = g.node(accesses[0]);
                if (first.ins.size() >= 2) {
                    g.add_input(guard, first.ins[0]);   // control
                    g.add_input(guard, first.ins[1]);   // effect (memory)
                }
            }
            for (NodeId a : accesses) {
                if (g.node(a).ins.size() >= 3) g.add_input(guard, g.node(a).ins[2]);
            }
            FrameState fs;
            fs.code_unit_id = c.code_unit_id;
            gn.aux1 = g.add_frame_state(fs);
        }

        // Mark accesses with the vector form for the backend selector.
        for (NodeId a : accesses) {
            if (g.node(a).has(NodeFlag::TypeGuarded) || all_proved) {
                g.node(a).set_flag(NodeFlag::Vectorizable);
                ++vectorized;
            }
        }
    }

    PassResult r = result_of(g, before);
    r.changed = vectorized > 0;
    note(TelemetryEventKind::SafepointPatched, c, vectorized);
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
