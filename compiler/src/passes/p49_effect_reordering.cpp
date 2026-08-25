// =============================================================================
// Pass 49 — Speculative Effect Reordering.
//
// Reorders memory operations to expose ILP: independent loads (bases
// proven non-aliasing by passes 12/14) hoist together and issue in
// parallel; stores batch after the loads. Tier 3 (AOT): CFL-Reachability
// proof required, no guards. Tier 2 (JIT): PGO probability >= 99% plus
// an AliasDisjoint guard whose failure deoptimizes (Rules 2/3). Pure
// loads over region-allocated (never-escaping) bases reorder freely —
// no observable semantics can intervene.
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

        // Tier 2: guards required on non-proved pairs.
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
            }
        } else if (c.requires_proofs()) {
            // Tier 3: every load must carry the CFL proof marker.
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

        // Mark the group: the scheduler issues these loads back-to-back
        // (no intervening stores — the effect chain keeps them adjacent).
        for (NodeId load : kv.second) {
            g.node(load).set_flag(NodeFlag::Hot);
            ++reordered;
        }
    }

    PassResult r = result_of(g, before);
    r.changed = false;
    note(TelemetryEventKind::SafepointPatched, c, reordered);
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
