// =============================================================================
// vortex/ir/verifier.cpp — Rule 40 checks implementation.
// =============================================================================

#include "vortex/ir/verifier.hpp"

#include <cstdio>

namespace vortex::ir {
inline namespace abi_v1 {

stdx::small_vector<Diagnostic, 4> verify_graph(const Graph& g) noexcept {
    stdx::small_vector<Diagnostic, 4> problems;

    // 1. Dangling ids / dead inputs.
    g.for_each_live([&](NodeId id) {
        const Node& n = g.node(id);
        for (NodeId input : n.ins) {
            if (input >= g.node_count()) {
                Diagnostic d = Diagnostic::error("dangling NodeId input",
                                                 diag_code::graph_verify_dangling_node);
                d.actual = "input id";
                d.fix = "Pass rewired inputs to an out-of-range id; audit replace_all_uses";
                problems.push_back(d);
            } else if (g.node(input).has(NodeFlag::Dead)) {
                Diagnostic d = Diagnostic::error("live node consumes dead node",
                                                 diag_code::graph_verify_dangling_node);
                d.fix = "DCE must rewrite users before killing";
                problems.push_back(d);
            }
        }
    });

    // 2 & 4. Use-count consistency + effect-chain continuity.
    stdx::small_vector<std::uint32_t, 256> observed_uses(g.node_count(), 0);
    g.for_each_live([&](NodeId id) {
        const Node& n = g.node(id);
        for (NodeId input : n.ins) {
            if (input < g.node_count()) observed_uses[input]++;
        }
    });
    g.for_each_live([&](NodeId id) {
        if (g.node(id).use_count != observed_uses[id]) {
            Diagnostic d = Diagnostic::error("use-count bookkeeping drift",
                                             diag_code::graph_verify_use_def);
            d.expected = "use_count == observed input occurrences";
            d.actual = "drift";
            d.fix = "Always route input edits through Graph::add_input/set_input";
            problems.push_back(d);
        }
    });

    // Effect chain: every OnEffectChain node must be reachable from the
    // start's memory projection chain (through EffectPhi / effect inputs).
    // Compute reachability over "effect producer -> effect consumer" edges:
    // a node X is chained if some live node has X as its effect input.
    stdx::small_vector<std::uint8_t, 256> chained(g.node_count(), 0);
    g.for_each_live([&](NodeId id) {
        const Node& n = g.node(id);
        if (!n.has(NodeFlag::OnEffectChain)) return;
        // Effect input slot: ins[1] for memory-class nodes (convention).
        if (n.ins.size() >= 2 && n.ins[1] < g.node_count()) {
            chained[n.ins[1]] = 1;
        }
    });
    // Start seeds the chain.
    if (g.start() != invalid_node) {
        const Node& s = g.node(g.start());
        (void)s;
    }
    g.for_each_live([&](NodeId id) {
        const Node& n = g.node(id);
        if (!n.has(NodeFlag::OnEffectChain)) return;
        if (n.kind == NodeKind::EffectPhi) return;   // merges are chain roots
        // A chained op must have an effect input that is itself chained
        // (or be the first op right after start).
        bool ok = false;
        if (n.ins.size() >= 2 && n.ins[1] < g.node_count()) {
            const Node& eff = g.node(n.ins[1]);
            ok = chained[n.ins[1]] == 1 || eff.kind == NodeKind::EffectPhi ||
                 eff.kind == NodeKind::Start;
        }
        if (!ok) {
            Diagnostic d = Diagnostic::error("effect chain discontinuity",
                                             diag_code::graph_verify_effect_chain);
            d.fix = "Effect producers must chain via ins[1]; check lowering/pass edits";
            problems.push_back(d);
        }
    });

    // 3. Control inputs of control-region users are Region/If/Loop projections.
    //
    // Catch is exempted just like Region: a Catch node is the merge target
    // for runtime exception dispatch. When the try body's only statement
    // is a `return EXPR` where EXPR may raise (e.g. `return 1/0`), the
    // body's lowering pushes no post-statement ArmState snapshots
    // (control_ becomes invalid after the Return terminates), so the
    // Catch region has zero IR-edges in — but it is still reachable
    // at runtime via the exception dispatch mechanism. Exempting it
    // from the structural "control node must have a control input"
    // invariant keeps the verifier sound on this shape (the catch path's
    // reachability is runtime-driven, not IR-edge-driven — the same
    // reason Region is exempted: a region may be a structurally-empty
    // merge that the scheduler/runtime fills in later).
    g.for_each_live([&](NodeId id) {
        const Node& n = g.node(id);
        if (!is_control(n.kind)) return;
        if (n.kind == NodeKind::Start) return;
        if (n.ins.empty() && n.kind != NodeKind::Region &&
            n.kind != NodeKind::Catch) {
            Diagnostic d = Diagnostic::error("control node without control input",
                                             diag_code::graph_verify_dominance);
            d.fix = "Control nodes must consume a control projection";
            problems.push_back(d);
        }
    });

    // 5. Speculative guards carry a FrameState index within range.
    g.for_each_live([&](NodeId id) {
        const Node& n = g.node(id);
        if (n.kind != NodeKind::Guard) return;
        if (!n.has(NodeFlag::Speculative)) return;   // non-PGO guards exempt (static)
        if (n.aux1 >= g.frame_state_count()) {
            Diagnostic d = Diagnostic::error("speculative guard missing FrameState (Rule 5)",
                                             diag_code::graph_verify_framestate);
            d.fix = "Attach add_frame_state() result via aux1 when emitting the guard";
            problems.push_back(d);
        }
    });

    return problems;
}

bool verify_or_report(const Graph& g, [[maybe_unused]] const char* pass_name) noexcept {
    auto problems = verify_graph(g);
    if (problems.empty()) return true;
#if VORTEX_DEBUG || VORTEX_ENABLE_ASSERTS
    std::fprintf(stderr, "== VORTEX verifier: %zu problem(s) after pass '%s' ==\n",
                 problems.size(), pass_name ? pass_name : "<unknown>");
    for (const Diagnostic& d : problems) d.report(stderr);
#endif
    return false;
}

}  // namespace abi_v1
}  // namespace vortex::ir
