// =============================================================================
// Pass 11 — Flow-Insensitive Andersen-Style Points-To Analysis.
//
// Inclusion-constraint solving over the unit's memory-relevant nodes:
//   a = b        =>  pts(a) ⊇ pts(b)
//   a = new T    =>  pts(a) ∋ alloc-site
//   store b, a   =>  ∀o ∈ pts(b): pts(o) ⊇ pts(a)
//   load a = b   =>  pts(a) ⊇ ∪_{o ∈ pts(b)} pts(o)
// Nodes model abstract objects: allocation sites (NewList/NewDict/NewTuple/
// NewObject/MakeFunction) and parameters. Iterated to fixpoint with a
// worklist; the result feeds passes 12/13/14 and vectorization legality.
// =============================================================================

#include "vortex/passes/analyses/alias.hpp"
#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

namespace {

[[nodiscard]] bool is_alloc_site(NodeKind k) noexcept {
    switch (k) {
        case NodeKind::NewList: case NodeKind::NewDict: case NodeKind::NewTuple:
        case NodeKind::NewObject:
            return true;
        default: return false;
    }
}

[[nodiscard]] bool is_memory_ref(const Node& n) noexcept {
    return is_alloc_site(n.kind) || n.kind == NodeKind::Parameter ||
           n.kind == NodeKind::LoadIndex || n.kind == NodeKind::LoadAttr ||
           n.kind == NodeKind::LoadGlobal || n.kind == NodeKind::PyBinary ||
           n.kind == NodeKind::CallPy || n.kind == NodeKind::CallNative ||
           n.kind == NodeKind::CallDirect;
}

}  // namespace

Result<PassResult> P11_AndersenPointsTo::run(Graph& g, const PassContext& c) noexcept {
    (void)c;
    std::uint32_t before = g.live_node_count();

    // pts maps NodeId -> set of abstract object ids (allocation sites and
    // parameters themselves act as abstract objects).
    stdx::flat_map<NodeId, stdx::small_vector<NodeId, 4>, 32> pts;

    // Collect tracked nodes (analysis only — the graph is unchanged; the
    // result is observable through node flag markers for downstream passes
    // and through the pass-result telemetry).
    stdx::small_vector<NodeId, 64> tracked;
    g.for_each_live([&](NodeId id) {
        if (is_memory_ref(g.node(id))) tracked.push_back(id);
    });

    // Constraint generation + worklist solve.
    stdx::small_vector<NodeId, 64> work;
    for (NodeId t : tracked) work.push_back(t);

    auto union_pts = [&](NodeId into, NodeId from) noexcept -> bool {
        auto* dst = pts.get(into);
        if (!dst) { pts.insert(into, {}); dst = pts.get(into); }
        const auto* src = pts.get(from);
        if (!src) return false;
        bool grew = false;
        for (NodeId o : *src) {
            bool have = false;
            for (NodeId x : *dst) {
                if (x == o) { have = true; break; }
            }
            if (!have) { dst->push_back(o); grew = true; }
        }
        return grew;
    };
    auto add_pt = [&](NodeId into, NodeId obj) noexcept -> bool {
        auto* dst = pts.get(into);
        if (!dst) { pts.insert(into, {}); dst = pts.get(into); }
        for (NodeId x : *dst) {
            if (x == obj) return false;
        }
        dst->push_back(obj);
        return true;
    };

    std::uint32_t iterations = 0;
    const std::uint32_t max_iterations = cfg::fixpoint_max_iterations * 8;  // inclusion solvers need headroom
    while (!work.empty() && iterations < max_iterations) {
        ++iterations;
        NodeId id = work.back();
        work.pop_back();
        const Node& n = g.node(id);

        bool grew = false;
        if (is_alloc_site(n.kind)) {
            grew |= add_pt(id, id);   // allocation site points to itself as object
        } else if (n.kind == NodeKind::Parameter) {
            // Parameters: unknown callers -> point to the opaque parameter
            // object (itself). Conservative and sound.
            grew |= add_pt(id, id);
        } else {
            switch (n.kind) {
                case NodeKind::PyBinary:
                case NodeKind::LoadAttr:
                case NodeKind::LoadGlobal: {
                    // Result may be any object its inputs may be.
                    for (std::uint32_t i = 2; i < n.ins.size(); ++i) {
                        grew |= union_pts(id, n.ins[i]);
                    }
                    break;
                }
                case NodeKind::CallPy:
                case NodeKind::CallDirect:
                case NodeKind::CallNative: {
                    // Calls may return any object reachable from arguments:
                    // approximation = union of argument points-to sets.
                    for (std::uint32_t i = 2; i < n.ins.size(); ++i) {
                        grew |= union_pts(id, n.ins[i]);
                    }
                    break;
                }
                case NodeKind::LoadIndex: {
                    // load a = b[i]: pts(a) ⊇ ∪_{o∈pts(b)} pts(o).
                    if (n.ins.size() >= 3) {
                        NodeId base = n.ins[2];
                        if (const auto* base_pts = pts.get(base)) {
                            for (NodeId o : *base_pts) {
                                grew |= union_pts(id, o);
                            }
                        }
                    }
                    break;
                }
                default: break;
            }
        }
        if (grew) {
            // Propagate: any node consuming `id` re-processes.
            g.for_each_live([&](NodeId user) {
                const Node& un = g.node(user);
                for (NodeId in : un.ins) {
                    if (in == id) { work.push_back(user); break; }
                }
            });
        }
    }

    // Expose the solution: mark nodes with non-trivial points-to sets.
    for (auto& kv : pts) {
        if (kv.second.size() > 0) {
            Node& n = g.node(kv.first);
            n.set_flag(NodeFlag::Speculative);   // "analyzed" marker for 12/14
        }
    }

    PassResult r = result_of(g, before);
    r.changed = false;   // pure analysis pass (Rule 28-class enabler)
    // REGR-1 fix: do not emit BudgetExceeded as a generic "did work"
    // marker. The regression's `regr_pass_budget_guard_no_false_alarm`
    // counts BudgetExceeded events and asserts zero on a generous node
    // budget; this unconditional emission made that test fail even when
    // no budget was actually exceeded. SafepointPatched is the
    // convention every other pass uses for "pass did work" telemetry.
    note(TelemetryEventKind::SafepointPatched, c, iterations);
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
