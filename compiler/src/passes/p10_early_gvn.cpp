// =============================================================================
// Pass 10 — Early Global Value Numbering.
//
// Herbrand equivalence: value-number every pure node by its structural
// key (kind + subop + payload + input ids). Two nodes with equal numbers
// compute the same value; the later is replaced by the earlier. Runs
// globally (not per-block, unlike pass 07) and re-runs after an internal
// DCE sweep exposes further equivalences through forwarded inputs.
// =============================================================================

#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

namespace {

bool gvn_round(Graph& g) noexcept {
    bool changed = false;
    stdx::flat_map<std::uint64_t, NodeId, 64> table;
    g.for_each_live([&](NodeId id) {
        Node& n = g.node(id);
        if (is_control(n.kind)) return;
        if (!n.has(NodeFlag::Pure)) return;
        if (n.has(NodeFlag::MayCall) || n.has(NodeFlag::MayThrow)) return;
        std::uint64_t h = g.node_hash(id);
        for (std::uint32_t attempt = 0; attempt < 2; ++attempt) {
            std::uint64_t key = h + attempt;
            if (NodeId* existing = table.get(key)) {
                if (g.node(*existing).structurally_equal(n)) {
                    g.replace_all_uses(id, *existing);
                    g.kill(id);
                    changed = true;
                    return;
                }
            } else {
                table.insert(key, id);
                return;
            }
        }
        table.insert(h ^ 0x9e3779b97f4a7c15ull, id);
    });
    return changed;
}

void sweep(Graph& g) noexcept {
    bool changed = true;
    while (changed) {
        changed = false;
        g.for_each_live([&](NodeId id) {
            Node& n = g.node(id);
            if (is_control(n.kind)) return;
            if (n.use_count == 0 && !n.has(NodeFlag::OnEffectChain)) {
                g.kill(id);
                changed = true;
            }
        });
    }
}

}  // namespace

Result<PassResult> P10_EarlyGVN::run(Graph& g, const PassContext& c) noexcept {
    if (c.tier == TierMode::Tier1) {
        note(TelemetryEventKind::BudgetExceeded, c);
        return PassResult{};   // deferred to Tier 2/3 (budget)
    }
    std::uint32_t before = g.live_node_count();
    bool changed = gvn_round(g);
    sweep(g);
    if (gvn_round(g)) changed = true;
    PassResult r = result_of(g, before);
    r.changed = changed;
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
