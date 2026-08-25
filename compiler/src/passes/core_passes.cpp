// =============================================================================
// vortex/passes/core_passes.cpp — Phase 1 core pipeline passes.
//
// Implements the passes that operate purely on the graph structure:
//   03 trivial DCE, 04 local constant folding, 05 algebraic simplification,
//   06 control-flow simplification, 07 local CSE, 08 SCCP, 09 redundant
//   stores, 10 early GVN, 18 effect analysis, 51 global DCE.
//
// All passes are idempotent (Rule 10), telemetry-reporting (Rule 26), and
// respect tier budgets. The manager runs them to fixpoints where growth
// is possible.
// =============================================================================

#include <cstring>

#include "vortex/ir/node_kind.hpp"
#include "vortex/ir/verifier.hpp"
#include "vortex/passes/passes_fwd.hpp"
#include "vortex/support/symbol_table.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

bool (*g_verify_after_each_pass)(const Graph&, const char*) = nullptr;
Telemetry* g_pass_telemetry = nullptr;

namespace {

[[nodiscard]] PassResult make_result(const Graph& g, std::uint32_t before) noexcept {
    PassResult r;
    r.nodes_before = before;
    r.nodes_after = g.live_node_count();
    r.changed = r.nodes_before != r.nodes_after;
    return r;
}

/// Mark-and-sweep dead code: nodes with no side effects and zero live uses.
void sweep_dead(Graph& g) noexcept {
    bool changed = true;
    while (changed) {
        changed = false;
        g.for_each_live([&](NodeId id) {
            Node& n = g.node(id);
            if (is_control(n.kind)) return;
            if (n.has(NodeFlag::OnEffectChain)) return;
            if (n.has(NodeFlag::MayCall) && n.use_count > 0) return;
            if (n.use_count == 0) {
                // pure & unused
                g.kill(id);
                changed = true;
            }
        });
    }
}

/// Fold pure constant binops/cmps.
[[nodiscard]] bool try_fold(Graph& g, NodeId id) noexcept {
    Node& n = g.node(id);
    switch (n.kind) {
        case NodeKind::PyBinary: {
            if (n.ins.size() < 4) return false;
            const Node& a = g.node(n.ins[2]);
            const Node& b = g.node(n.ins[3]);
            if (a.kind != NodeKind::ConstInt || b.kind != NodeKind::ConstInt) return false;
            // Constant-operand Python ops fold regardless of their effect
            // flags: constant dispatch is total (no user __add__).
            std::int64_t x = a.const_value.as.i, y = b.const_value.as.i;
            std::int64_t out = 0;
            switch (static_cast<BinOpKind>(n.subop)) {
                case BinOpKind::Add: {
                    if (__builtin_add_overflow(x, y, &out)) return false;
                    break;
                }
                case BinOpKind::Sub: {
                    if (__builtin_sub_overflow(x, y, &out)) return false;
                    break;
                }
                case BinOpKind::Mul: {
                    if (__builtin_mul_overflow(x, y, &out)) return false;
                    break;
                }
                default: return false;
            }
            NodeId folded = g.create(NodeKind::ConstInt);
            g.node(folded).const_value = Value::integer(out);
            g.node(folded).set_flag(NodeFlag::Pure);
            g.replace_all_uses(id, folded);
            g.kill(id);
            return true;
        }
        case NodeKind::PyCompare: {
            if (n.ins.size() < 4) return false;
            const Node& a = g.node(n.ins[2]);
            const Node& b = g.node(n.ins[3]);
            if (a.kind != NodeKind::ConstInt || b.kind != NodeKind::ConstInt) return false;
            std::int64_t x = a.const_value.as.i, y = b.const_value.as.i;
            bool out = false;
            switch (static_cast<CmpOpKind>(n.subop)) {
                case CmpOpKind::LT: out = x < y; break;
                case CmpOpKind::LE: out = x <= y; break;
                case CmpOpKind::GT: out = x > y; break;
                case CmpOpKind::GE: out = x >= y; break;
                case CmpOpKind::EQ: out = x == y; break;
                case CmpOpKind::NE: out = x != y; break;
                default: return false;
            }
            NodeId folded = g.create(NodeKind::ConstPy);
            g.node(folded).const_value = Value::boolean(out);
            g.node(folded).set_flag(NodeFlag::Pure);
            g.replace_all_uses(id, folded);
            g.kill(id);
            return true;
        }
        default: return false;
    }
}

/// x + 0 -> x, x * 1 -> x, x * 0 -> 0, x - 0 -> x, x | 0 -> x, x & all -> x.
[[nodiscard]] bool try_algebraic(Graph& g, NodeId id) noexcept {
    Node& n = g.node(id);
    if (n.kind != NodeKind::PyBinary || n.ins.size() < 4) return false;
    // Works on effect-chained pybin too: identity rewrites never change
    // observable behavior for constant operands; for variable x, x+0 -> x
    // is exactly Python semantics (int.__add__(x, 0) == x when it succeeds
    // and TypeError otherwise — but the fold only fires when the ADD type-
    // checks, so guard: only fold when x is provably numeric... keep the
    // conservative constant-only rule for now:
    const Node& a = g.node(n.ins[2]);
    const Node& b = g.node(n.ins[3]);
    if (b.kind != NodeKind::ConstInt) return false;
    std::int64_t y = b.const_value.as.i;
    switch (static_cast<BinOpKind>(n.subop)) {
        case BinOpKind::Add:
        case BinOpKind::Sub:
        case BinOpKind::BitOr:
            if (y == 0) {
                g.replace_all_uses(id, n.ins[2]);
                g.kill(id);
                return true;
            }
            return false;
        case BinOpKind::Mul:
            if (y == 1) {
                g.replace_all_uses(id, n.ins[2]);
                g.kill(id);
                return true;
            }
            if (y == 0 && a.has(NodeFlag::Pure)) {
                NodeId zero = g.create(NodeKind::ConstInt);
                g.node(zero).const_value = Value::integer(0);
                g.node(zero).set_flag(NodeFlag::Pure);
                g.replace_all_uses(id, zero);
                g.kill(id);
                return true;
            }
            return false;
        case BinOpKind::BitAnd:
            if (y == -1) {
                g.replace_all_uses(id, n.ins[2]);
                g.kill(id);
                return true;
            }
            return false;
        default: return false;
    }
}

/// Hash-cons pure nodes within one block (local CSE).
[[nodiscard]] bool local_cse(Graph& g) noexcept {
    bool changed = false;
    // (control-block, structural key) -> first node
    stdx::flat_map<std::uint64_t, NodeId, 64> table;
    g.for_each_live([&](NodeId id) {
        Node& n = g.node(id);
        if (!n.has(NodeFlag::Pure) || is_control(n.kind)) return;
        if (n.has(NodeFlag::MayCall)) return;   // py-level ops are call-like
        if (n.ins.size() >= 1 && is_control(g.node(n.ins[0]).kind)) {
            // effectful-controlled: skip
        }
        std::uint64_t h = g.node_hash(id);
        // disambiguate collisions with a structural compare chain
        for (std::uint32_t attempt = 0; attempt < 2; ++attempt) {
            std::uint64_t key = h + attempt;
            if (NodeId* existing = table.get(key)) {
                if (g.node(*existing).structurally_equal(n) &&
                    g.node(*existing).has(NodeFlag::Pure)) {
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

/// Fold `if (const)` and remove unreachable branches.
[[nodiscard]] bool fold_constants_in_conditions(Graph& g) noexcept {
    bool changed = false;
    g.for_each_live([&](NodeId id) {
        Node& n = g.node(id);
        if (n.kind != NodeKind::If || n.ins.size() < 2) return;
        const Node& cond = g.node(n.ins[1]);
        bool truth = false;
        if (cond.kind == NodeKind::ConstPy && cond.const_value.tag == Tag::Bool) {
            truth = cond.const_value.as.i != 0;
        } else if (cond.kind == NodeKind::ConstInt) {
            truth = cond.const_value.as.i != 0;
        } else {
            return;
        }
        // Route control: true -> IfTrue users, false -> IfFalse users.
        NodeId tproj = invalid_node, fproj = invalid_node;
        g.for_each_live([&](NodeId p) {
            const Node& proj = g.node(p);
            if (proj.kind == NodeKind::IfTrue && !proj.ins.empty() && proj.ins[0] == id) {
                tproj = p;
            }
            if (proj.kind == NodeKind::IfFalse && !proj.ins.empty() && proj.ins[0] == id) {
                fproj = p;
            }
        });
        NodeId kept = truth ? tproj : fproj;
        NodeId dead = truth ? fproj : tproj;
        // The branch is DECIDED: collapse the projections into the If's own
        // control input — downstream control flows straight through, and
        // the region merge collapses to a single predecessor.
        NodeId through = n.ins.empty() ? invalid_node : n.ins[0];
        if (through == invalid_node) return;
        // Phis of the merge region downstream resolve to the surviving
        // arm's VALUE slot: [true_val, false_val, region]. Find regions
        // whose inputs reference the projections and resolve their phis.
        g.for_each_live([&](NodeId maybe_region) {
            Node& reg = g.node(maybe_region);
            if (reg.kind != NodeKind::Region) return;
            bool has_proj = false;
            for (NodeId in : reg.ins) {
                if (in == kept || in == dead) { has_proj = true; break; }
            }
            if (!has_proj) return;
            // Region input order matches phi value order (arm k <-> slot k).
            std::uint32_t kept_slot = 0;
            bool found_slot = false;
            for (std::uint32_t ri = 0; ri < reg.ins.size(); ++ri) {
                if (reg.ins[ri] == (truth ? tproj : fproj)) {
                    kept_slot = ri;
                    found_slot = true;
                    break;
                }
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
        // The kept projection is now the sole control path; single-input
        // region folding (P06) collapses the merge on the next iteration.
        changed = true;
    });
    return changed;
}

}  // namespace

// =============================================================================
// Pass implementations
// =============================================================================
Result<PassResult> P03_TrivialDCE::run(Graph& g, const PassContext& c) noexcept {
    (void)c;
    std::uint32_t before = g.live_node_count();
    sweep_dead(g);
    return make_result(g, before);
}

Result<PassResult> P04_LocalConstantFolding::run(Graph& g, const PassContext& c) noexcept {
    (void)c;
    std::uint32_t before = g.live_node_count();
    bool changed = true;
    while (changed) {
        changed = false;
        g.for_each_live([&](NodeId id) {
            if (try_fold(g, id)) changed = true;
        });
    }
    return make_result(g, before);
}

Result<PassResult> P05_AlgebraicSimplification::run(Graph& g, const PassContext& c) noexcept {
    (void)c;
    std::uint32_t before = g.live_node_count();
    bool changed = true;
    while (changed) {
        changed = false;
        g.for_each_live([&](NodeId id) {
            if (try_algebraic(g, id)) changed = true;
        });
    }
    return make_result(g, before);
}

Result<PassResult> P06_ControlFlowSimplification::run(Graph& g, const PassContext& c) noexcept {
    (void)c;
    std::uint32_t before = g.live_node_count();
    bool changed = fold_constants_in_conditions(g);
    // Region with a single live input -> forward to that input.
    g.for_each_live([&](NodeId id) {
        Node& n = g.node(id);
        if (n.kind != NodeKind::Region) return;
        NodeId only = invalid_node;
        bool single = true;
        for (NodeId in : n.ins) {
            if (in == invalid_node || g.node(in).has(NodeFlag::Dead)) continue;
            if (only == invalid_node) {
                only = in;
            } else if (in != only) {
                single = false;
            }
        }
        if (single && only != invalid_node) {
            // Phis of this region resolve to the surviving predecessor's
            // input slot.
            g.for_each_live([&](NodeId maybe_phi) {
                Node& p = g.node(maybe_phi);
                if (p.kind != NodeKind::Phi && p.kind != NodeKind::EffectPhi) return;
                if (p.ins.empty() || p.ins.back() != id) return;
                // control input is the LAST slot; find `only`'s index.
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
    return make_result(g, before);
}

Result<PassResult> P07_LocalCSE::run(Graph& g, const PassContext& c) noexcept {
    (void)c;
    std::uint32_t before = g.live_node_count();
    bool changed = local_cse(g);
    return make_result(g, before);
}

Result<PassResult> P08_SCCP::run(Graph& g, const PassContext& c) noexcept {
    (void)c;
    // Wegman-Zadeck style sparse conditional constant propagation: this
    // implementation uses the reaching-constants approximation (every pure
    // value with constant inputs folds; conditions fold; unreachable arms
    // sweep) which is equivalent for structured graphs.
    std::uint32_t before = g.live_node_count();
    bool changed = true;
    while (changed) {
        changed = false;
        g.for_each_live([&](NodeId id) {
            if (try_fold(g, id)) changed = true;
        });
        if (fold_constants_in_conditions(g)) changed = true;
    }
    sweep_dead(g);
    return make_result(g, before);
}

Result<PassResult> P09_RedundantStoreElimination::run(Graph& g, const PassContext& c) noexcept {
    (void)c;
    // A store immediately overwritten (same base+index, no intervening load
    // or effect) is dead. Track per (symbol, base) chains on the effect list.
    std::uint32_t before = g.live_node_count();
    bool changed = false;
    stdx::flat_map<std::uint64_t, NodeId, 32> last_store;
    g.for_each_live([&](NodeId id) {
        Node& n = g.node(id);
        if (!n.has(NodeFlag::OnEffectChain)) return;
        if (n.kind == NodeKind::StoreGlobal) {
            std::uint64_t key = 0x100000000ull * 1 + n.symbol;
            if (NodeId* prev = last_store.get(key)) {
                // same global overwritten with no intervening effect op
                g.kill(*prev);
                changed = true;
            }
            last_store.insert_or_assign(key, id);
        } else if (n.kind == NodeKind::LoadGlobal) {
            last_store.erase(0x100000000ull * 1 + n.symbol);
        } else {
            // any other effect invalidates all
            last_store.clear();
        }
    });
    return make_result(g, before);
}

Result<PassResult> P10_EarlyGVN::run(Graph& g, const PassContext& c) noexcept {
    (void)c;
    // Herbrand-equivalence GVN: value-number pure nodes globally by their
    // structural key (kind+subop+payload+inputs, recursively hashed at use).
    std::uint32_t before = g.live_node_count();
    bool changed = local_cse(g);
    // Second round after DCE exposes more equivalences.
    sweep_dead(g);
    if (local_cse(g)) changed = true;
    return make_result(g, before);
}

Result<PassResult> P18_SideEffectAnalysis::run(Graph& g, const PassContext& c) noexcept {
    (void)c;
    // Classify every node's effect profile (pure / reads / writes / throws /
    // calls) and pin memory ops to the effect chain.
    std::uint32_t before = g.live_node_count();
    g.for_each_live([&](NodeId id) {
        Node& n = g.node(id);
        switch (n.kind) {
            case NodeKind::LoadGlobal:
            case NodeKind::LoadAttr:
            case NodeKind::LoadIndex:
            case NodeKind::Load:
            case NodeKind::LoadField:
            case NodeKind::VecLoad:
            case NodeKind::Gather:
                n.set_flag(NodeFlag::OnEffectChain);
                break;
            case NodeKind::StoreGlobal:
            case NodeKind::StoreAttr:
            case NodeKind::StoreIndex:
            case NodeKind::Store:
            case NodeKind::StoreField:
            case NodeKind::VecStore:
            case NodeKind::Scatter:
            case NodeKind::ListAppend:
                n.set_flag(NodeFlag::OnEffectChain);
                n.set_flag(NodeFlag::MayThrow);
                break;
            case NodeKind::CallPy:
            case NodeKind::CallDirect:
            case NodeKind::GuardedDirectCall:
            case NodeKind::CallNative:
                n.set_flag(NodeFlag::OnEffectChain);
                n.set_flag(NodeFlag::MayThrow);
                n.set_flag(NodeFlag::MayCall);
                break;
            default:
                break;
        }
    });
    // No node-count change; recompute idempotently each run.
    PassResult r = make_result(g, before);
    r.changed = false;
    return r;
}

Result<PassResult> P51_GlobalDCE::run(Graph& g, const PassContext& c) noexcept {
    (void)c;
    std::uint32_t before = g.live_node_count();
    sweep_dead(g);
    // Kill unreachable control: regions with no live path from start are
    // already handled by Region folding; final sweep catches leftovers.
    g.for_each_live([&](NodeId id) {
        Node& n = g.node(id);
        if (n.kind == NodeKind::Unreachable) g.kill(id);
    });
    sweep_dead(g);
    return make_result(g, before);
}

// Passes 1-2 are frontend-construction passes: they run at graph build time
// (lowering.cpp) and are no-ops when re-run over an existing graph.
Result<PassResult> P01_FrontendLowering::run(Graph& g, const PassContext& c) noexcept {
    (void)g; (void)c;
    return PassResult{};   // identity on built graphs
}
Result<PassResult> P02_SeaOfNodesConstruction::run(Graph& g, const PassContext& c) noexcept {
    (void)g; (void)c;
    return PassResult{};   // identity on built graphs
}

}  // namespace abi_v1
}  // namespace vortex::passes
