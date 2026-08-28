// =============================================================================
// Pass 06 — Control Flow Simplification.
//
// Folds constant-condition branches and collapses the resulting merge
// structures: when an If folds, both projections route through the If's
// control input; downstream phis resolve to the surviving arm's value
// slot (matched by region-input position); single-input regions (and
// regions whose live inputs are all one node) forward to that input.
// Unreachable Region/If subtrees sweep via DCE.
// =============================================================================

#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

namespace {

bool fold_if(Graph& g, NodeId id) noexcept {
    Node& n = g.node(id);
    if (n.kind != NodeKind::If || n.ins.size() < 2) return false;
    const Node& cond = g.node(n.ins[1]);
    bool truth = false;
    if (cond.kind == NodeKind::ConstPy && cond.const_value.tag() == Tag::Bool) {
        truth = cond.const_value.as_i() != 0;
    } else if (cond.kind == NodeKind::ConstInt) {
        truth = cond.const_value.as_i() != 0;
    } else {
        return false;
    }

    NodeId tproj = invalid_node, fproj = invalid_node;
    g.for_each_live([&](NodeId p) {
        const Node& proj = g.node(p);
        if (proj.kind == NodeKind::IfTrue && !proj.ins.empty() && proj.ins[0] == id) tproj = p;
        if (proj.kind == NodeKind::IfFalse && !proj.ins.empty() && proj.ins[0] == id) fproj = p;
    });
    NodeId kept = truth ? tproj : fproj;
    NodeId dead = truth ? fproj : tproj;
    NodeId through = n.ins[0];

    // Resolve merge phis first: region inputs referencing a projection match
    // phi value slots by position.
    g.for_each_live([&](NodeId maybe_region) {
        Node& reg = g.node(maybe_region);
        if (reg.kind != NodeKind::Region) return;
        std::uint32_t kept_slot = 0;
        bool found_slot = false;
        for (std::uint32_t ri = 0; ri < reg.ins.size(); ++ri) {
            if (reg.ins[ri] == kept) { kept_slot = ri; found_slot = true; break; }
        }
        if (!found_slot) return;
        g.for_each_live([&](NodeId maybe_phi) {
            Node& p = g.node(maybe_phi);
            if (p.kind != NodeKind::Phi && p.kind != NodeKind::EffectPhi) return;
            if (p.ins.empty() || p.ins.back() != maybe_region) return;
            if (kept_slot < p.ins.size() - 1) {
                g.replace_all_uses(maybe_phi, p.ins[kept_slot]);
                g.kill(maybe_phi);
            }
        });
    });

    if (dead != invalid_node) {
        g.replace_all_uses(dead, through);
        g.kill(dead);
    }
    if (kept != invalid_node) {
        g.replace_all_uses(kept, through);
        g.kill(kept);
    }
    g.replace_all_uses(id, through);
    g.kill(id);
    return true;
}

bool fold_single_input_regions(Graph& g) noexcept {
    bool changed = false;
    g.for_each_live([&](NodeId id) {
        Node& n = g.node(id);
        if (n.kind != NodeKind::Region) return;
        NodeId only = invalid_node;
        bool single = true;
        for (NodeId in : n.ins) {
            if (in == invalid_node || g.node(in).has(NodeFlag::Dead)) continue;
            if (only == invalid_node) only = in;
            else if (in != only) single = false;
        }
        if (single && only != invalid_node) {
            // Phis of this region resolve to the surviving predecessor's slot.
            g.for_each_live([&](NodeId maybe_phi) {
                Node& p = g.node(maybe_phi);
                if (p.kind != NodeKind::Phi && p.kind != NodeKind::EffectPhi) return;
                if (p.ins.empty() || p.ins.back() != id) return;
                for (std::uint32_t pi = 0; pi + 1 < p.ins.size(); ++pi) {
                    if (p.ins[pi] == only) {
                        g.replace_all_uses(maybe_phi, p.ins[pi]);
                        g.kill(maybe_phi);
                        break;
                    }
                }
            });
            g.replace_all_uses(id, only);
            g.kill(id);
            changed = true;
        }
    });
    return changed;
}

}  // namespace

Result<PassResult> P06_ControlFlowSimplification::run(Graph& g, const PassContext& c) noexcept {
    (void)c;
    std::uint32_t before = g.live_node_count();
    bool changed = true;
    while (changed) {
        changed = false;
        g.for_each_live([&](NodeId id) {
            if (fold_if(g, id)) changed = true;
        });
        if (fold_single_input_regions(g)) changed = true;
    }
    return result_of(g, before);
}

}  // namespace abi_v1
}  // namespace vortex::passes
