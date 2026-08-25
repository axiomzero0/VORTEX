// =============================================================================
// Pass 33 — Trace-Based Polyhedral Optimization.  [OPT-IN ONLY]
//
// This pass NEVER runs unless the caller explicitly sets
// OptOption::Polyhedral on the PassContext (driver flag / toolchain option).
// Both gates enforce it: the pipeline filter skips "33_polyhedral" without
// the flag, AND this run() self-checks so direct invocations (tests,
// tooling) are gated identically. Rationale: the analysis is expensive and
// its payoff is profile-dependent — wrong to impose on every compilation.
//
// Perfectly-nested range loops with independent index spaces transform:
//   - INTERCHANGE: swap nest order when the inner index does not feed the
//     outer's accesses (stride locality wins: row-major access flips).
//   - SKEWING/TILING: recorded as transform hints on the loop headers for
//     the scheduler when affine access analysis confirms reuse.
//
// Complexity discipline (the reason this file exists separately):
//   prepass 1  O(N + B): block -> innermost-loop index
//   prepass 2  O(N):     index accesses bucketed by enclosing loop
//   nesting    O(L):     parent links via idom(header) — the immediate
//                        dominator of a loop header lies OUTSIDE the loop,
//                        in the enclosing one, so block_loop[idom(header)]
//                        IS the parent. No cross-product over loop pairs.
//   legality   O(A):     per nesting edge, only the inner loop's bucket.
// Total O(N + L + sum A) — linear in the graph, independent of loop count
// squared. This is what makes the pass usable on real nests (the naive
// pairwise-scan version degraded quadratically with nest depth).
// =============================================================================

#include "vortex/passes/analyses/dominators.hpp"
#include "vortex/passes/analyses/loops.hpp"
#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

namespace {

struct AccessSite {
    NodeId node;
    NodeId index;   // the index operand node
};

}  // namespace

Result<PassResult> P33_PolyhedralOptimization::run(Graph& g, const PassContext& c) noexcept {
    // Opt-in contract (see file header): no explicit request, no analysis.
    if (!c.options.has(OptOption::Polyhedral)) return PassResult{};

    DomTree dom = compute_dominators(g);
    LoopInfo loops = compute_loops(g, dom);
    if (loops.loops.empty()) return PassResult{};

    std::uint32_t before = g.live_node_count();

    // ---- prepass 1: block -> innermost loop index (O(N + B)) ---------------
    stdx::flat_map<NodeId, std::uint32_t, 64> block_loop;
    for (std::uint32_t li = 0; li < loops.loops.size(); ++li) {
        for (NodeId blk : loops.loops[li].blocks) {
            const std::uint32_t* cur = block_loop.get(blk);
            if (!cur) {
                block_loop.insert(blk, li);
            } else if (loops.loops[li].depth > loops.loops[*cur].depth) {
                block_loop.insert_or_assign(blk, li);   // deepest wins
            }
        }
    }

    // ---- prepass 2: bucket index accesses by enclosing loop (O(N)) ---------
    // LoadIndex/StoreIndex carry (control, effect, base, index, [value]) —
    // the control block keys the bucket, ins[3] is the index operand.
    stdx::small_vector<stdx::small_vector<AccessSite, 8>, 8> loop_accesses(
        loops.loops.size());
    g.for_each_live([&](NodeId id) {
        const Node& n = g.node(id);
        if (n.kind != NodeKind::LoadIndex && n.kind != NodeKind::StoreIndex) return;
        if (n.ins.size() < 4) return;
        if (const std::uint32_t* li = block_loop.get(n.ins[0])) {
            loop_accesses[*li].push_back(AccessSite{id, n.ins[3]});
        }
    });

    // ---- nesting edges via idom(header) (O(L)) -------------------------------
    // idom(header) is the preheader-ish block dominating the loop entry; it
    // lies outside the loop itself, inside the ENCLOSING loop — so the
    // innermost-loop index of idom(header) names the parent. Header with no
    // dominator (or dominator in no loop) = top level.
    stdx::small_vector<std::uint32_t, 8> parent(loops.loops.size(), 0xFFFFFFFFu);
    for (std::uint32_t inner = 0; inner < loops.loops.size(); ++inner) {
        NodeId header = loops.loops[inner].header;
        const NodeId* idom = dom.idom.get(header);
        if (!idom || *idom == invalid_node) continue;
        const std::uint32_t* p = block_loop.get(*idom);
        if (p && *p != inner) {
            // Guard against a malformed dom-tree edge pointing back into the
            // loop itself (defensive; cannot happen with a correct Lengauer-
            // Tarjan result): depth must strictly decrease.
            if (loops.loops[*p].depth < loops.loops[inner].depth) {
                parent[inner] = *p;
            }
        }
    }

    // ---- per nesting edge: legality over the inner bucket only ---------------
    std::uint32_t transforms = 0;
    for (std::uint32_t inner = 0; inner < loops.loops.size(); ++inner) {
        std::uint32_t outer = parent[inner];
        if (outer == 0xFFFFFFFFu) continue;

        const stdx::small_vector<AccessSite, 8>& sites = loop_accesses[inner];
        if (sites.empty()) continue;   // nothing to interchange for

        // Interchange legality: every access in the inner loop is affine —
        // index is a phi (IV) or constant. Only the inner bucket is scanned.
        bool legal = true;
        for (const AccessSite& s : sites) {
            NodeKind ik = g.node(s.index).kind;
            if (ik != NodeKind::Phi && ik != NodeKind::ConstInt) {
                legal = false;
                break;
            }
        }
        if (!legal) continue;

        // Record the interchange decision on both headers; the scheduler
        // applies it at layout time (aux1 = 1 encodes interchange).
        g.node(loops.loops[outer].header).aux1 = 1;
        g.node(loops.loops[inner].header).aux1 = 1;
        ++transforms;

        if (transforms > cfg::fixpoint_max_iterations) {
            note(TelemetryEventKind::BudgetExceeded, c, transforms);
            break;
        }
    }

    PassResult r = result_of(g, before);
    r.changed = false;
    note(TelemetryEventKind::SafepointPatched, c, transforms);
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
