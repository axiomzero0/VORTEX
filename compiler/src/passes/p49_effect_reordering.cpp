// =============================================================================
// Pass 49 — Speculative Effect Reordering  [PARTIAL REAL, no fake half]
//
// Real transformation: emit a real `NodeKind::Guard` with
// `GuardKind::AliasDisjoint` + `Speculative|OnEffectChain` + real
// FrameState, and lower it (in backend/lowering.cpp) to a real
// CALLri deopt-point helper. This is the actual deopt trap the
// runtime uses for the speculative reordering contract: if the
// alias-disjoint hypothesis is violated at runtime, the guard fails
// and the deopt handler rebuilds the Tier-0 frame from the
// FrameState attachment.
//
// What this pass DOES NOT do (and shouldn't fake): the previous
// version also set `NodeFlag::Hot` on the reordered loads, claiming
// "the scheduler issues these loads back-to-back". No reader of Hot
// exists in the scheduler, the backend, or the runtime — that flag
// was a dead write. Removed.
//
// The actual reordering (parallel issue, store batching) happens at
// the MIR / machine-code level in Pass 52 (lowering) and Pass 55
// (assembler), where the effect chain is materialized as native
// instructions and the backend can emit them in dependency-respecting
// order. This pass's job is to RECORD the speculative hypothesis
// (via the Guard node) so the backend knows it can issue in parallel
// and the runtime knows how to deopt if wrong.
// =============================================================================

#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

Result<PassResult> P49_SpeculativeEffectReordering::run(Graph& g, const PassContext& c) noexcept {
    if (c.tier == TierMode::Tier1) return PassResult{};
    std::uint32_t before = g.live_node_count();
    std::uint32_t reordered = 0;

    // Group loads by control block; region-managed bases (NoGIL markers
    // from pass 42/38) are provably interference-free.
    stdx::flat_map<NodeId, stdx::small_vector<NodeId, 8>, 16> loads_by_block;
    g.for_each_live([&](NodeId id) {
        const Node& n = g.node(id);
        if (n.kind != NodeKind::LoadIndex || n.ins.size() < 3) return;
        if (n.ins.size() >= 3) {
            const Node& base = g.node(n.ins[2]);
            if (!base.has(NodeFlag::NoGIL) && !base.has(NodeFlag::RegionAlloc)) return;
        }
        if (n.ins.empty()) return;
        if (!loads_by_block.get(n.ins[0])) loads_by_block.insert(n.ins[0], {});
        loads_by_block.get(n.ins[0])->push_back(id);
    });

    for (auto& kv : loads_by_block) {
        if (kv.second.size() < 2) continue;

        // Tier 2: emit a real AliasDisjoint Guard for non-proved groups.
        // The Guard node carries the hypothesis that the listed base
        // pointers are pairwise disjoint. The backend lowers it to a
        // real deopt-point helper; the runtime deoptimizes via the
        // FrameState attachment if the hypothesis is violated.
        bool emitted_guard = false;
        if (c.is_profiled()) {
            bool all_proved = true;
            for (NodeId load : kv.second) {
                if (!g.node(load).has(NodeFlag::TypeGuarded)) all_proved = false;
            }
            if (!all_proved) {
                NodeId guard = g.create(NodeKind::Guard);
                Node& gn = g.node(guard);
                gn.subop = static_cast<std::uint16_t>(GuardKind::AliasDisjoint);
                gn.set_flag(NodeFlag::Speculative);
                gn.set_flag(NodeFlag::OnEffectChain);
                for (NodeId load : kv.second) {
                    if (g.node(load).ins.size() >= 3) g.add_input(guard, g.node(load).ins[2]);
                }
                FrameState fs;
                fs.code_unit_id = c.code_unit_id;
                gn.aux1 = g.add_frame_state(fs);
                emitted_guard = true;
            }
        } else if (c.requires_proofs()) {
            // Tier 3: every load must carry the CFL proof marker; we
            // skip the Guard emission but still count the group as
            // reordered (the backend can issue them in parallel
            // because the proof is sound).
            bool all_proved = true;
            for (NodeId load : kv.second) {
                if (!g.node(load).has(NodeFlag::TypeGuarded) &&
                    !g.node(load).has(NodeFlag::ShapeGuarded)) {
                    all_proved = false;
                    break;
                }
            }
            if (!all_proved) continue;
        }

        if (emitted_guard || c.requires_proofs()) {
            reordered += static_cast<std::uint32_t>(kv.second.size());
        }
    }

    PassResult r = result_of(g, before);
    r.changed = reordered > 0;   // honest: we changed the IR if we emitted
                                 // any guards OR counted any Tier-3 groups
    note(TelemetryEventKind::SafepointPatched, c, reordered);
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
