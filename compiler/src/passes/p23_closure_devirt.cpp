// =============================================================================
// Pass 23 — Closure & Lambda Devirtualization.
//
// Converts closure-cell traffic into direct SSA flow when the cell's
// producer is visible in the same unit: a CellGet whose input cell is a
// MakeCell whose initial value is a known node becomes a plain reference
// to that node (the cell layer is a pure indirection for read-only
// closures). CellSet chains to the same cell collapse to the final value
// when no intervening CellGet reads the intermediate state.
// =============================================================================

#include "vortex/frontend/lowering.hpp"
#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;
using vortex::fe::NativeHelper;

Result<PassResult> P23_ClosureDevirtualization::run(Graph& g, const PassContext& c) noexcept {
    (void)c;
    std::uint32_t before = g.live_node_count();
    bool changed = false;

    // map: cell node -> initial value node (from MakeCell)
    stdx::flat_map<NodeId, NodeId, 16> cell_init;
    // PASS-5 fix: track which cells are EVER written via CellSet. A cell
    // that's written can't have its CellGet forwarded to the initial
    // value — the actual value at CellGet time depends on the program
    // order of CellSet/CellGet. The previous code forwarded unconditionally,
    // miscompiling any case where CellSet changes the cell value before
    // CellGet reads it.
    stdx::flat_map<NodeId, bool, 16> cell_has_set;
    g.for_each_live([&](NodeId id) {
        const Node& n = g.node(id);
        if (n.kind != NodeKind::CallNative) return;
        const auto helper = static_cast<NativeHelper>(n.subop);
        if (helper == NativeHelper::MakeCell) {
            if (n.ins.size() < 3) return;
            cell_init.insert(n.id, n.ins[2]);
        } else if (helper == NativeHelper::CellSet) {
            if (n.ins.size() >= 3) {
                cell_has_set.insert_or_assign(n.ins[2], true);
            }
        }
    });

    // CellGet(cell) where cell is in-unit MakeCell -> forward initial value
    // ONLY if there are NO CellSets to that cell anywhere (otherwise the
    // cell's value at CellGet time depends on intervening CellSets).
    g.for_each_live([&](NodeId id) {
        Node& n = g.node(id);
        if (n.kind != NodeKind::CallNative) return;
        if (static_cast<NativeHelper>(n.subop) != NativeHelper::CellGet) return;
        if (n.ins.size() < 3) return;
        NodeId cell = n.ins[2];
        // PASS-5: skip forwarding if any CellSet writes to this cell.
        if (const bool* has_set = cell_has_set.get(cell)) {
            (void)has_set;
            return;
        }
        if (const NodeId* init = cell_init.get(cell)) {
            g.replace_all_uses(id, *init);
            g.kill(id);
            changed = true;
        }
    });

    return result_of(g, before);
}

}  // namespace abi_v1
}  // namespace vortex::passes
