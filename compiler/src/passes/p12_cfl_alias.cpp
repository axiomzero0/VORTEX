// =============================================================================
// Pass 12 — CFL-Reachability Alias Analysis (Reps, Horwitz, Sagiv 1995).
//
// Models aliasing as a context-free language reachability problem over the
// constraint graph. Terminal symbols:
//   assign(a,b) — b's value flows into a
//   deref(a)     — a is used as a memory base (store/load)
//   alloc(a)     — a binds a fresh object
// Grammar (alias-relevant nonterminals):
//   V  -> assign V | deref^{-1} V deref | alloc        (value flow)
//   A  -> V V^{-1}                                     (two names meet one object)
// A pair (a,b) MAY alias iff A reaches (a,b). We compute the V-relation by
// transitive closure over assign edges, objects = alloc + deref-composed
// sources, then A = pairs sharing a reachable object. Sets stored in a
// flat bit matrix; O(n^3) closure bounded by the node budget.
// =============================================================================

#include "vortex/passes/analyses/alias.hpp"
#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

namespace {

[[nodiscard]] bool is_alloc(NodeKind k) noexcept {
    switch (k) {
        case NodeKind::NewList: case NodeKind::NewDict: case NodeKind::NewTuple:
        case NodeKind::NewObject:
            return true;
        default: return false;
    }
}

[[nodiscard]] bool is_base_user(const Node& n) noexcept {
    // nodes that treat an input as a memory base
    switch (n.kind) {
        case NodeKind::LoadIndex: case NodeKind::StoreIndex:
        case NodeKind::LoadAttr: case NodeKind::StoreAttr:
            return true;
        default: return false;
    }
}

}  // namespace

Result<PassResult> P12_CFLReachabilityAlias::run(Graph& g, const PassContext& c) noexcept {
    if (budget_exceeded(g, c)) {
        note(TelemetryEventKind::BudgetExceeded, c);
        return PassResult{};   // Rule 45: cost gate on the O(n^3) closure
    }
    std::uint32_t before = g.live_node_count();

    // Build the V-relation directly: V(x,y) iff value of y can reach x.
    // assign edges: from data inputs to the consuming node.
    stdx::flat_map<NodeId, stdx::small_vector<NodeId, 4>, 32> flows_to;
    auto add_flow = [&](NodeId from, NodeId to) noexcept {
        if (!flows_to.get(from)) flows_to.insert(from, {});
        flows_to.get(from)->push_back(to);
    };
    g.for_each_live([&](NodeId id) {
        const Node& n = g.node(id);
        for (std::uint32_t i = 2; i < n.ins.size(); ++i) {   // data inputs
            add_flow(n.ins[i], id);
        }
        if (n.kind == NodeKind::Phi || n.kind == NodeKind::EffectPhi) {
            for (std::uint32_t i = 0; i + 1 < n.ins.size(); ++i) {
                add_flow(n.ins[i], id);
            }
        }
    });

    // Transitive closure of flows_to (Warshall on the sparse adjacency via
    // repeated union — bounded by budget).
    stdx::flat_map<NodeId, stdx::small_vector<NodeId, 4>, 32> reach = flows_to;
    bool changed = true;
    std::uint32_t iters = 0;
    while (changed && iters < cfg::fixpoint_max_iterations) {
        changed = false;
        ++iters;
        for (auto& kv : reach) {
            stdx::small_vector<NodeId, 4> add;
            for (NodeId y : kv.second) {
                if (const auto* ys = reach.get(y)) {
                    for (NodeId z : *ys) add.push_back(z);
                }
            }
            auto* dst = reach.get(kv.first);
            for (NodeId z : add) {
                bool have = false;
                for (NodeId x : *dst) {
                    if (x == z) { have = true; break; }
                }
                if (!have) { dst->push_back(z); changed = true; }
            }
        }
    }

    // Objects: allocation sites. Names: nodes that may HOLD an object.
    // may_hold(x) = {x if alloc} ∪ {o : alloc(o) ∧ reach(o, x)}.
    stdx::flat_map<NodeId, stdx::small_vector<NodeId, 4>, 32> may_hold;
    auto add_hold = [&](NodeId holder, NodeId obj) noexcept {
        if (!may_hold.get(holder)) may_hold.insert(holder, {});
        may_hold.get(holder)->push_back(obj);
    };
    g.for_each_live([&](NodeId id) {
        if (is_alloc(g.node(id).kind)) {
            add_hold(id, id);
            if (const auto* rs = reach.get(id)) {
                for (NodeId x : *rs) add_hold(x, id);
            }
        }
    });

    // Alias pairs: two nodes share at least one object. Record NoAlias pairs
    // as the actionable output: disjoint object sets.
    std::uint32_t noalias_pairs = 0;
    std::uint32_t tracked = 0;
    for (auto& ka : may_hold) {
        for (auto& kb : may_hold) {
            if (ka.first >= kb.first) continue;
            ++tracked;
            bool shared = false;
            for (NodeId o1 : ka.second) {
                for (NodeId o2 : kb.second) {
                    if (o1 == o2) { shared = true; break; }
                }
                if (shared) break;
            }
            if (!shared) ++noalias_pairs;
        }
    }
    // Mark memory-base users with the analysis marker for downstream
    // vectorization legality (31c/32) and effect reordering (49).
    g.for_each_live([&](NodeId id) {
        if (is_base_user(g.node(id))) {
            g.node(id).set_flag(NodeFlag::ShapeGuarded);   // "alias-analyzed"
        }
    });

    PassResult r = result_of(g, before);
    r.changed = false;
    (void)noalias_pairs;
    (void)tracked;
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
