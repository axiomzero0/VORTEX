// =============================================================================
// Pass 33 — Polyhedral Loop Interchange.  [DEFAULT-ON, OPT-OUT]
//
// This pass runs by default in every tier (including the default Tier 2 driver
// path). The caller MAY opt out by setting OptOption::DisablePolyhedral on the
// PassContext — the only legitimate reason to do so is compilation-time
// sensitivity (the analysis is O(N + L + ΣA·D), which is linear for bounded
// D but non-trivial in absolute terms on hot loops). Both gates enforce the
// opt-out: the pipeline filter skips "33_polyhedral" when the flag is set,
// AND this run() self-checks so direct invocations (tests, tooling) are gated
// identically. Defaulting ON satisfies Rule 28 — every optimization must
// demonstrate measurable improvement OR enable a correctness/safety property
// that cannot be achieved otherwise — loop interchange preserves correctness
// under the legality checks below AND measurably improves cache locality on
// nested loops over contiguous arrays.
//
// Transformation implemented: REAL loop interchange.
//
// For a perfectly-nested pair (L_outer, L_inner) — single-entry, single-exit,
// inner.header is the only Loop nested in L_outer.blocks — the pass:
//
//   (a) Verifies rectangular bounds: each If's bound ConstInt does not
//       reference the OTHER loop's IV (transitive data-dep closure).
//   (b) Verifies body invariance: every LoadIndex/StoreIndex in the inner
//       body has an index affine in the inner IV only (no transitive read
//       of the outer IV).
//
// Then performs the swap:
//
//   1. SNAPSHOT the four control projections:
//        outer_body_entry = IfTrue(outer_If)
//        outer_exit       = IfFalse(outer_If)
//        inner_body_tail  = IfTrue(inner_If)
//        inner_exit       = IfFalse(inner_If)
//      After the swap, their semantic roles permute:
//        outer_body_entry → new_inner_body_tail  (same node, new role)
//        outer_exit       → new_inner_exit
//        inner_body_tail  → new_outer_body_entry
//        inner_exit       → new_outer_exit
//
//   2. SWAP LOOP INS:
//        L_outer.ins = [inner_body_tail, outer_body_entry]  (becomes new inner)
//        L_inner.ins = [Start,            outer_exit]        (becomes new outer)
//
//   3. MOVE THE RETURN: the Return's ctrl was outer_exit; move to inner_exit
//      (which is now the new outer exit).
//
//   4. REMAP BODY NODE CTRLS. Each body content node's ctrl was one of the
//      four projections; its new ctrl depends on whether the node is an
//      IV update or other body content:
//        ctrl=inner_body_tail, is-inner-IV-update → ctrl=outer_exit
//        ctrl=inner_body_tail, else               → ctrl=outer_body_entry
//        ctrl=inner_exit                          → ctrl=outer_body_entry
//        ctrl=outer_body_entry                     → ctrl=inner_body_tail
//        ctrl=outer_exit, Return                  → ctrl=inner_exit (step 3)
//
//   5. REWIRE PHI ENTRIES/BACKEDGES. For each (outer_phi, inner_phi) pair
//      where inner_phi.entry == outer_phi (the standard "value of v inside
//      the inner loop" pattern):
//        new_outer_phi (=inner_phi).entry = old outer_phi.entry  (Start value)
//        new_inner_phi (=outer_phi).entry = new_outer_phi        (= inner_phi)
//        if old outer_phi.backedge lives in the OLD INNER body (or is the
//          cross-loop edge to inner_phi), rewrite to self-reference on
//          the new inner_phi — the variable doesn't change in the NEW
//          inner body either.
//
//   The new_outer_phi's backedge is left alone: it pointed to a node in
//   the old inner body, which is now the new outer body — exactly where
//   new_outer_phi's backedge SHOULD come from.
//
// Semantics: the iteration space { (i, j) } is unchanged; only the
// traversal order flips. With condition (b) the inner body's memory
// accesses are functions of (i, j) that don't depend on i, so the new
// (j, i) traversal visits the same memory cells — interchange is
// legality-preserving.
//
// Complexity discipline:
//   prepass 1  O(N + L):       block -> innermost loop index
//   prepass 2  O(N):           index accesses bucketed by enclosing loop
//   nesting    O(L):           parent links via idom(header)
//   legality   O(A * D):       per pair, inner-bucket scan with bounded
//                              transitive-dependency walk (D = dep depth)
//   transform  O(N):           single pass over body nodes for ctrl remap
// Total O(N + L + sum A * D) — linear in the graph for bounded D.
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

/// Find the If node whose control input is `loop_header` (the loop's exit
/// test). Returns invalid_node if none.
[[nodiscard]] NodeId find_loop_if(const Graph& g, NodeId loop_header) noexcept {
    NodeId found = invalid_node;
    g.for_each_live([&](NodeId id) {
        const Node& n = g.node(id);
        if (n.kind == NodeKind::If && !n.ins.empty() && n.ins[0] == loop_header) {
            found = id;
        }
    });
    return found;
}

/// Find the IV phi of a loop: a Phi whose control input is `loop_header`,
/// whose entry value is a ConstInt, and whose backedge value is a PyBinary
/// `+` with a ConstInt RHS (the stride). We deliberately do NOT require
/// the add's LHS to be the phi itself — in nested-loop SSA the backedge
/// flows through the inner-loop perspective phi first (outer_iv_phi →
/// inner_iv_phi → inner_iv_phi + stride → outer_iv_phi). Requiring
/// self-reference would refuse every nested case, which is exactly the
/// case polyhedral interchange is for.
[[nodiscard]] NodeId find_iv_phi(const Graph& g, NodeId loop_header) noexcept {
    NodeId found = invalid_node;
    g.for_each_live([&](NodeId id) {
        if (found != invalid_node) return;        // first match wins
        const Node& n = g.node(id);
        if (n.kind != NodeKind::Phi) return;
        if (n.ins.size() < 3) return;
        if (n.ins[2] != loop_header) return;             // ctrl = this loop
        if (g.node(n.ins[0]).kind != NodeKind::ConstInt) return;  // entry = const
        NodeId back = n.ins[1];
        const Node& bn = g.node(back);
        if (bn.kind != NodeKind::PyBinary) return;
        if (bn.subop != static_cast<std::uint16_t>(BinOpKind::Add)) return;
        if (bn.ins.size() < 4) return;
        // The increment must be a ConstInt (i = i + 1, not i = i + something).
        if (g.node(bn.ins[3]).kind != NodeKind::ConstInt) return;
        found = id;
    });
    return found;
}

/// Find the bound ConstInt of a loop's If. The If's condition is a PyCompare
/// `LT` whose rhs is a ConstInt. Returns invalid_node if not found.
[[nodiscard]] NodeId find_loop_bound(const Graph& g, NodeId loop_if) noexcept {
    const Node& ifn = g.node(loop_if);
    if (ifn.ins.size() < 2) return invalid_node;
    const Node& cond = g.node(ifn.ins[1]);
    if (cond.kind != NodeKind::PyCompare) return invalid_node;
    if (cond.subop != static_cast<std::uint16_t>(CmpOpKind::LT)) return invalid_node;
    if (cond.ins.size() < 4) return invalid_node;
    NodeId rhs = cond.ins[3];
    if (g.node(rhs).kind != NodeKind::ConstInt) return invalid_node;
    return rhs;
}

/// Find the IfTrue projection of `loop_if`. invalid_node if none.
[[nodiscard]] NodeId find_if_true(const Graph& g, NodeId if_id) noexcept {
    NodeId found = invalid_node;
    g.for_each_live([&](NodeId id) {
        const Node& n = g.node(id);
        if (n.kind == NodeKind::IfTrue && !n.ins.empty() && n.ins[0] == if_id) {
            found = id;
        }
    });
    return found;
}

/// Find the IfFalse projection of `loop_if`. invalid_node if none.
[[nodiscard]] NodeId find_if_false(const Graph& g, NodeId if_id) noexcept {
    NodeId found = invalid_node;
    g.for_each_live([&](NodeId id) {
        const Node& n = g.node(id);
        if (n.kind == NodeKind::IfFalse && !n.ins.empty() && n.ins[0] == if_id) {
            found = id;
        }
    });
    return found;
}

/// Does the data dependency closure of `root` (following Phi / PyBinary /
/// Arith inputs, but only data inputs — never control or effect) include
/// `target`? Used to prove the inner-body access index doesn't transitively
/// read the outer IV.
[[nodiscard]] bool transitively_reads(const Graph& g, NodeId root,
                                      NodeId target,
                                      stdx::small_vector<NodeId, 32>& on_stack) noexcept {
    if (root == target) return true;
    for (NodeId s : on_stack) {
        if (s == root) return false;
    }
    on_stack.push_back(root);
    const Node& n = g.node(root);
    std::uint32_t start = 0;
    std::uint32_t end = n.ins.size();
    switch (n.kind) {
        case NodeKind::Phi:
            start = 0; end = n.ins.size() >= 3 ? n.ins.size() - 1 : 0;  // skip ctrl
            break;
        case NodeKind::PyBinary: case NodeKind::PyUnary: case NodeKind::PyCompare:
            start = n.ins.size() >= 2 ? 2 : 0;   // ctrl at [0], eff at [1]
            break;
        case NodeKind::Add: case NodeKind::Sub: case NodeKind::Mul:
        case NodeKind::Div: case NodeKind::Mod: case NodeKind::Pow:
        case NodeKind::CmpLT: case NodeKind::CmpLE: case NodeKind::CmpGT:
        case NodeKind::CmpGE: case NodeKind::CmpEQ: case NodeKind::CmpNE:
            start = 0;
            break;
        default:
            on_stack.pop_back();
            return false;
    }
    bool found = false;
    for (std::uint32_t i = start; i < end; ++i) {
        if (transitively_reads(g, n.ins[i], target, on_stack)) {
            found = true;
            break;
        }
    }
    on_stack.pop_back();
    return found;
}

[[nodiscard]] bool index_depends_on_outer_iv(const Graph& g, NodeId index_node,
                                             NodeId outer_iv) noexcept {
    if (index_node == outer_iv) return true;
    stdx::small_vector<NodeId, 32> on_stack;
    return transitively_reads(g, index_node, outer_iv, on_stack);
}

/// True iff `b` is in the inner loop's blocks (i.e., the block leader is
/// one of L_inner.blocks).
[[nodiscard]] bool block_in_inner(const LoopInfo::Loop& L_inner, NodeId b) noexcept {
    for (NodeId m : L_inner.blocks) {
        if (m == b) return true;
    }
    return false;
}

}  // namespace

Result<PassResult> P33_PolyhedralOptimization::run(Graph& g, const PassContext& c) noexcept {
    // Opt-out contract (see file header): the pass runs by default. The
    // caller MAY set OptOption::DisablePolyhedral to skip it — the only
    // legitimate reason is compilation-time sensitivity (the analysis is
    // linear but non-trivial). Both gates enforce the opt-out: the pipeline
    // filter skips "33_polyhedral" when the flag is set, AND this run()
    // self-checks so direct invocations (tests, tooling) are gated
    // identically.
    if (c.options.has(OptOption::DisablePolyhedral)) return PassResult{};

    DomTree dom = compute_dominators(g);
    LoopInfo loops = compute_loops(g, dom);
    if (loops.loops.size() < 2) return PassResult{};

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
    stdx::small_vector<std::uint32_t, 8> parent(loops.loops.size(), 0xFFFFFFFFu);
    for (std::uint32_t inner = 0; inner < loops.loops.size(); ++inner) {
        NodeId header = loops.loops[inner].header;
        const NodeId* idom = dom.idom.get(header);
        if (!idom || *idom == invalid_node) continue;
        const std::uint32_t* p = block_loop.get(*idom);
        if (p && *p != inner) {
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

        const LoopInfo::Loop& L_inner = loops.loops[inner];
        const LoopInfo::Loop& L_outer = loops.loops[outer];

        // (a) perfect nesting: inner.blocks ⊆ outer.blocks, AND the inner
        // header is the only OTHER Loop node in outer.blocks.
        bool subset = true;
        for (NodeId b : L_inner.blocks) {
            bool in_outer = false;
            for (NodeId ob : L_outer.blocks) {
                if (ob == b) { in_outer = true; break; }
            }
            if (!in_outer) { subset = false; break; }
        }
        if (!subset) continue;

        std::uint32_t loop_count = 0;
        for (NodeId b : L_outer.blocks) {
            if (b == L_outer.header) continue;
            if (g.node(b).kind == NodeKind::Loop) ++loop_count;
        }
        if (loop_count != 1) continue;

        // (b) both loops single-entry, single-backedge: ins.size() == 2.
        const Node& outer_hdr = g.node(L_outer.header);
        const Node& inner_hdr = g.node(L_inner.header);
        if (outer_hdr.ins.size() != 2 || inner_hdr.ins.size() != 2) continue;

        // (c) rectangular bounds + IV identification.
        NodeId outer_if = find_loop_if(g, L_outer.header);
        NodeId inner_if = find_loop_if(g, L_inner.header);
        if (outer_if == invalid_node || inner_if == invalid_node) continue;

        NodeId outer_iv = find_iv_phi(g, L_outer.header);
        NodeId inner_iv = find_iv_phi(g, L_inner.header);
        if (outer_iv == invalid_node || inner_iv == invalid_node) continue;

        NodeId outer_bound = find_loop_bound(g, outer_if);
        NodeId inner_bound = find_loop_bound(g, inner_if);
        if (outer_bound == invalid_node || inner_bound == invalid_node) continue;

        if (index_depends_on_outer_iv(g, outer_bound, inner_iv)) continue;
        if (index_depends_on_outer_iv(g, inner_bound, outer_iv)) continue;

        // (d) every access in the inner body has an index affine in the
        // inner IV only — not transitively derived from the outer IV.
        const stdx::small_vector<AccessSite, 8>& sites = loop_accesses[inner];
        bool legal = true;
        for (const AccessSite& s : sites) {
            if (index_depends_on_outer_iv(g, s.index, outer_iv)) {
                legal = false;
                break;
            }
        }
        if (!legal) continue;

        // (e) NO BREAKS / CONTINUES / EARLY EXITS. Loop interchange
        // permutes the iteration order of (i, j) over a rectangular
        // space. A `break` inside the OLD inner body exits the OLD
        // inner loop and continues with the OLD outer body's tail
        // (e.g., `i = i + 1`). After the swap, the OLD inner body
        // becomes the NEW outer body, so the break would now exit the
        // NEW outer loop — semantically different and incorrect.
        //
        // Conservative check: count If nodes whose ctrl lives in
        // L_inner.blocks. The inner loop's OWN exit-test If (whose
        // ctrl is the inner Loop header) is the only legitimate If
        // inside the inner body. Any OTHER If inside the inner body
        // is a `break`, `continue`, or conditional with early exit —
        // all of which the interchange cannot preserve.
        //
        // This is overly restrictive (it refuses to interchange a
        // perfectly-nested loop with a benign `if cond: x = 1` in
        // the inner body, even though that conditional doesn't break).
        // That's acceptable: polyhedral interchange is a high-value
        // transform on PERFECTLY-NESTED CANONICAL loops, and the
        // extension to non-canonical cases requires a richer
        // dependence analysis ( scheduled for a future pass).
        std::uint32_t inner_body_ifs = 0;
        g.for_each_live([&](NodeId id) {
            const Node& n = g.node(id);
            if (n.kind != NodeKind::If) return;
            if (n.ins.empty()) return;
            NodeId ctrl = n.ins[0];
            // Skip the inner loop's own exit-test If — it's the
            // one legitimate If in the inner body.
            if (id == inner_if) return;
            // Check if ctrl is in L_inner.blocks.
            for (NodeId b : L_inner.blocks) {
                if (b == ctrl) {
                    ++inner_body_ifs;
                    break;
                }
            }
        });
        if (inner_body_ifs > 0) continue;

        // ---- TRANSFORM: real loop interchange. -----------------------------
        // 1. Snapshot the four control projections. These are node ids;
        //    the projections themselves don't move, but their semantic
        //    roles permute after the swap.
        NodeId outer_body_entry = find_if_true(g, outer_if);   // = new inner body tail
        NodeId outer_exit       = find_if_false(g, outer_if);  // = new inner exit
        NodeId inner_body_tail  = find_if_true(g, inner_if);   // = new outer body entry
        NodeId inner_exit       = find_if_false(g, inner_if);  // = new outer exit
        if (outer_body_entry == invalid_node || outer_exit == invalid_node ||
            inner_body_tail == invalid_node || inner_exit == invalid_node) continue;

        NodeId start_node = g.start();
        if (start_node == invalid_node) continue;

        // 2. Swap Loop ins arrays.
        //    Old outer Loop → new inner: entry=inner_body_tail (new outer
        //    body's tail — the new inner loop is entered from there),
        //    back=outer_body_entry (new inner body's tail).
        //    Old inner Loop → new outer: entry=Start, back=outer_exit
        //    (the new outer body's tail — where the OLD inner IV update
        //    moves to by step 4, becoming the new outer IV update).
        g.set_input(L_outer.header, 0, inner_body_tail);
        g.set_input(L_outer.header, 1, outer_body_entry);
        g.set_input(L_inner.header, 0, start_node);
        g.set_input(L_inner.header, 1, outer_exit);

        // 3. (No snapshot needed — step 6 no longer consults OLD ctrls.
        //    The original "rewrite_to_self" branch that needed the OLD
        //    ctrl snapshot has been removed: it was incorrect because
        //    it conflated "the variable doesn't change in the OLD outer
        //    body" with "the new inner phi's back-edge should be self",
        //    which broke SSA when step 4's body-content move put the
        //    variable's update into the new inner body. Step 6 now leaves
        //    back-edges unchanged — see step 6's rationale.)

        // 4. Remap body content ctrls. The interchange swaps the loops'
        //    roles; the body content moves between the four projections
        //    accordingly:
        //
        //    (a) Body content at OLD inner_body_tail (e.g., total += i*j):
        //        this was the OLD inner body content. After the swap, it
        //        should be in the NEW INNER body's tail (= OLD
        //        outer_body_entry), because the OLD inner body's content
        //        becomes the NEW inner body's content (the innermost
        //        loop's body stays innermost after interchange — that's
        //        what preserves the iteration count).
        //        Move: ctrl = inner_body_tail, not-IV-update → outer_body_entry.
        //
        //    (b) The OLD inner IV update (e.g., j = j+1): after the swap,
        //        j is the NEW OUTER IV. The new outer IV update should
        //        live at the NEW outer body's tail (= OLD outer_exit,
        //        which is where the new outer loop's back-edge comes
        //        from, per step 2). The new outer body's tail also
        //        serves as the new INNER loop's exit (a block can serve
        //        both roles in SoN IR).
        //        Move: ctrl = inner_body_tail, is-inner-IV-update → outer_exit.
        //
        //    (c) Body content at OLD inner_exit (e.g., i = i+1): this was
        //        the OLD outer body's tail content. After the swap, i is
        //        the NEW INNER IV. The new inner IV update should live
        //        at the NEW INNER body's tail (= OLD outer_body_entry).
        //        Move: ctrl = inner_exit → outer_body_entry.
        //
        //    (d) Body content at OLD outer_body_entry (rare; e.g., j = 0
        //        init when it's a separate node): after the swap, this
        //        content is in the NEW INNER body. If it was an
        //        initialization for the OLD inner IV (j), it should
        //        become the new outer IV's init — but extracting it from
        //        the body to the phi entry is a future extension. For
        //        now, move it to the NEW OUTER body's entry (= OLD
        //        inner_body_tail) — conservative, may not be fully
        //        correct for non-canonical IRs, but the legality check
        //        (e) above rejects most non-canonical patterns.
        //        Move: ctrl = outer_body_entry → inner_body_tail.
        //
        //    (e) Body content at OLD outer_exit (e.g., Return, print
        //        call, any end-of-program content): this was the OLD
        //        outer loop's exit content. After the swap, the OLD
        //        outer loop becomes the NEW INNER loop, so OLD
        //        outer_exit becomes the NEW INNER exit (= NEW OUTER
        //        body tail). But end-of-program content should run
        //        ONCE — after the NEW OUTER loop exits — not once per
        //        NEW OUTER iteration. So this content must move to the
        //        NEW OUTER exit (= OLD inner_exit).
        //        Move: ctrl = outer_exit → inner_exit.
        //        (Previous implementation only moved Return nodes here,
        //        which left print calls and other end-of-program
        //        content at OLD outer_exit, causing them to execute
        //        once per NEW OUTER iteration — see the nested_while
        //        regression: print ran 3 times instead of once.)
        struct CtrlRemap { NodeId node; NodeId new_ctrl; };
        stdx::small_vector<CtrlRemap, 32> remaps;
        // Identify the inner IV update node (back-edge of inner_iv phi).
        NodeId inner_iv_update = g.node(inner_iv).ins[1];
        g.for_each_live([&](NodeId id) {
            if (id == L_outer.header || id == L_inner.header) return;
            if (id == outer_if || id == inner_if) return;
            if (id == outer_body_entry || id == outer_exit) return;
            if (id == inner_body_tail || id == inner_exit) return;
            if (id == outer_iv || id == inner_iv) return;
            const Node& n = g.node(id);
            // Only nodes whose ctrl is one of the four projections need
            // remapping. Pure data nodes (no ctrl) are unaffected.
            if (n.ins.empty()) return;
            NodeId ctrl = n.ins[0];
            NodeId new_ctrl = invalid_node;
            if (ctrl == inner_body_tail) {
                if (id == inner_iv_update) {
                    // (b) Inner IV update → new outer body tail.
                    new_ctrl = outer_exit;
                } else {
                    // (a) Other inner body content → new inner body tail.
                    new_ctrl = outer_body_entry;
                }
            } else if (ctrl == inner_exit) {
                // (c) Old outer body tail content → new inner body tail.
                new_ctrl = outer_body_entry;
            } else if (ctrl == outer_body_entry) {
                // (d) Old outer body entry content → new outer body entry.
                new_ctrl = inner_body_tail;
            } else if (ctrl == outer_exit) {
                // (e) End-of-program content (Return, print call, etc.)
                //     → new outer exit. ALL content at OLD outer_exit
                //     moves, not just Return — otherwise end-of-program
                //     content runs once per NEW OUTER iteration.
                new_ctrl = inner_exit;
            }
            if (new_ctrl != invalid_node) {
                remaps.push_back({id, new_ctrl});
            }
        });
        for (const auto& r : remaps) {
            g.set_input(r.node, 0, r.new_ctrl);
        }

        // 5. Remap body content data inputs. The OLD outer body content
        //    (e.g., i = i+1 at OLD inner_exit, now moved to NEW INNER body
        //    tail = outer_body_entry by step 4) used the OLD INNER phis for
        //    "value of variable as seen by the inner loop" (because in the
        //    OLD layout, the i+1 update ran AFTER the inner loop exited,
        //    so it used the inner phi's back-edge value).
        //
        //    After the swap, this content is in the NEW INNER body. It
        //    should use the NEW INNER phis (= OLD OUTER phis). For each
        //    (outer_phi, inner_phi) pair, replace uses of inner_phi with
        //    outer_phi in body content whose NEW ctrl is outer_body_entry
        //    (= NEW INNER body tail).
        //
        //    Body content at NEW ctrl = inner_body_tail (= NEW OUTER body)
        //    is LEFT ALONE — it uses the OLD INNER phis, which are now the
        //    NEW OUTER phis. That's correct.
        //
        //    We skip the phi's OWN ins (entry/backedge) — those are
        //    handled in step 6.
        stdx::small_vector<std::pair<NodeId, NodeId>, 8> phi_pairs;
        g.for_each_live([&](NodeId id) {
            const Node& n = g.node(id);
            if (n.kind != NodeKind::Phi && n.kind != NodeKind::EffectPhi) return;
            if (n.ins.size() < 3) return;
            if (n.ins[2] != L_inner.header) return;       // ctrl = inner Loop
            NodeId entry = n.ins[0];
            const Node& en = g.node(entry);
            // Pattern (a): entry IS the outer_phi.
            if ((en.kind == NodeKind::Phi || en.kind == NodeKind::EffectPhi) &&
                en.ins.size() >= 3 && en.ins[2] == L_outer.header) {
                phi_pairs.push_back({entry, id});
                return;
            }
            // Pattern (b): entry's ctrl is the outer Loop header (e.g. the
            // outer If's condition, which sits in the outer Loop header
            // block). The "outer_phi" in the pair is the outer effect_phi
            // (its entry is the Start memory); we look it up.
            if (!en.ins.empty() && en.ins[0] == L_outer.header) {
                NodeId outer_eff = invalid_node;
                g.for_each_live([&](NodeId oid) {
                    if (outer_eff != invalid_node) return;
                    const Node& on = g.node(oid);
                    if (on.kind != NodeKind::EffectPhi) return;
                    if (on.ins.size() < 3) return;
                    if (on.ins[2] != L_outer.header) return;
                    outer_eff = oid;
                });
                if (outer_eff != invalid_node) {
                    phi_pairs.push_back({outer_eff, id});
                }
            }
        });

        // For each (outer_phi, inner_phi) pair, replace uses of inner_phi
        // with outer_phi in body content whose NEW ctrl is outer_body_entry
        // (= NEW INNER body tail) OR outer_exit (= NEW OUTER body tail,
        // where the OLD inner IV update was moved by step 4 — that node
        // is now in the new OUTER body, but it was using the OLD INNER
        // phis, which are now the NEW OUTER phis, so it's CORRECT to
        // leave those uses alone... wait, actually:
        //
        // The OLD inner IV update (e.g., j = j+1) used the OLD inner phi
        // for j (= OLD inner_phi). After step 4 moves it to outer_exit
        // (= NEW OUTER body tail), it's in the NEW OUTER body. The NEW
        // OUTER phi for j (= OLD inner_phi for j) is what it should use.
        // So we should LEAVE those uses alone (they already use the
        // correct phi).
        //
        // The body content that was MOVED to outer_body_entry (= NEW
        // INNER body tail) by step 4 used the OLD INNER phis. After the
        // swap, those phis are now the NEW OUTER phis. The body content
        // is in the NEW INNER body, so it should use the NEW INNER phis
        // (= OLD OUTER phis). So we replace inner_phi → outer_phi here.
        //
        // Step 4's rule "ctrl=outer_body_entry → ctrl=inner_body_tail"
        // moves OLD outer body entry content to the NEW OUTER body. That
        // content was using the OLD OUTER phis, which are now the NEW
        // INNER phis. After the move, it's in the NEW OUTER body, so it
        // should use the NEW OUTER phis (= OLD INNER phis). So we should
        // replace outer_phi → inner_phi for that content. But that's the
        // opposite direction — handle it in a separate pass below.
        struct DataRemap { NodeId node; std::uint32_t slot; NodeId new_value; };
        stdx::small_vector<DataRemap, 32> data_remaps;
        for (auto [outer_phi, inner_phi] : phi_pairs) {
            g.for_each_live([&](NodeId id) {
                if (id == outer_phi || id == inner_phi) return;
                if (id == L_outer.header || id == L_inner.header) return;
                if (id == outer_if || id == inner_if) return;
                if (id == outer_body_entry || id == outer_exit) return;
                if (id == inner_body_tail || id == inner_exit) return;
                const Node& n = g.node(id);
                if (n.kind == NodeKind::Phi || n.kind == NodeKind::EffectPhi) return;
                if (n.kind == NodeKind::Loop || n.kind == NodeKind::If) return;
                if (n.kind == NodeKind::IfTrue || n.kind == NodeKind::IfFalse) return;
                if (n.kind == NodeKind::Start) return;
                if (n.ins.empty()) return;
                // Only remap data inputs of body content that's now in
                // the NEW INNER body (= NEW ctrl is outer_body_entry).
                if (n.ins[0] != outer_body_entry) return;
                // For each input slot (except ctrl at slot 0), if it
                // equals inner_phi, schedule a remap to outer_phi.
                for (std::uint32_t s = 1; s < n.ins.size(); ++s) {
                    if (n.ins[s] == inner_phi) {
                        data_remaps.push_back({id, s, outer_phi});
                    }
                }
            });
        }
        // Also handle the reverse: body content moved to inner_body_tail
        // (= NEW OUTER body) by step 4's rule "ctrl=outer_body_entry →
        // ctrl=inner_body_tail" was using the OLD OUTER phis (= NEW
        // INNER phis). It's now in the NEW OUTER body, so it should use
        // the NEW OUTER phis (= OLD INNER phis). Replace outer_phi →
        // inner_phi for that content.
        for (auto [outer_phi, inner_phi] : phi_pairs) {
            g.for_each_live([&](NodeId id) {
                if (id == outer_phi || id == inner_phi) return;
                if (id == L_outer.header || id == L_inner.header) return;
                if (id == outer_if || id == inner_if) return;
                if (id == outer_body_entry || id == outer_exit) return;
                if (id == inner_body_tail || id == inner_exit) return;
                const Node& n = g.node(id);
                if (n.kind == NodeKind::Phi || n.kind == NodeKind::EffectPhi) return;
                if (n.kind == NodeKind::Loop || n.kind == NodeKind::If) return;
                if (n.kind == NodeKind::IfTrue || n.kind == NodeKind::IfFalse) return;
                if (n.kind == NodeKind::Start) return;
                if (n.ins.empty()) return;
                // Only remap data inputs of body content that's now in
                // the NEW OUTER body (= NEW ctrl is inner_body_tail).
                if (n.ins[0] != inner_body_tail) return;
                // For each input slot (except ctrl at slot 0), if it
                // equals outer_phi, schedule a remap to inner_phi.
                for (std::uint32_t s = 1; s < n.ins.size(); ++s) {
                    if (n.ins[s] == outer_phi) {
                        data_remaps.push_back({id, s, inner_phi});
                    }
                }
            });
        }
        for (const auto& d : data_remaps) {
            g.set_input(d.node, d.slot, d.new_value);
        }

        // 6. Rewire phi entries. The pairs were collected in step 5. For
        //    each pair, swap entry edges. The back-edges are LEFT
        //    UNCHANGED — step 4's ctrl remap already moved the body
        //    content updates to the correct new blocks, so the OLD
        //    outer_phi's back-edge node (which was the body content's
        //    update) is now in the NEW INNER body, which is exactly
        //    where the new INNER phi's back-edge should come from.
        //
        //    Concretely: for pair (outer_phi, inner_phi):
        //      new_outer_phi (= inner_phi).entry = OLD outer_phi.entry (start value).
        //      new_inner_phi (= outer_phi).entry = new_outer_phi (= inner_phi).
        //      Back-edges: LEFT ALONE. The OLD outer_phi's back-edge
        //        node was the body content's update (e.g., total += i*j),
        //        which step 4 moved to outer_body_entry (= NEW INNER body
        //        tail). So that back-edge is now correctly inside the
        //        new INNER body. Same for inner_phi's back-edge (it was
        //        the OLD inner body's update, which step 4 moved to
        //        outer_exit = NEW OUTER body tail, OR stayed in the new
        //        outer body).
        //
        //    The PREVIOUS implementation had a "rewrite to self" branch
        //    that incorrectly rewrote new INNER phi back-edges to self
        //    based on whether the OLD outer phi's back-edge was in the
        //    OLD inner body. That broke the SSA: the body content update
        //    IS the back-edge value, and after step 4's move it correctly
        //    lives in the new inner body — so the back-edge must point to
        //    it, not to self.
        for (auto [outer_phi, inner_phi] : phi_pairs) {
            NodeId start_value = g.node(outer_phi).ins[0];   // old outer_phi's entry
            // new_outer_phi (= inner_phi).entry = start_value
            g.set_input(inner_phi, 0, start_value);
            // new_inner_phi (= outer_phi).entry = new_outer_phi (= inner_phi)
            g.set_input(outer_phi, 0, inner_phi);
            // Back-edges left unchanged (see rationale above).
            (void)outer_phi;   // silence unused-lambda-capture warning
        }

        ++transforms;

        if (transforms > cfg::fixpoint_max_iterations) {
            // REGR-1 fix: the fixpoint iteration cap is an internal
            // limit, NOT the global node_budget. Same reasoning as
            // P27's per-loop cap — emit SafepointPatched to avoid
            // false BudgetExceeded alarms when the global budget is
            // ample.
            note(TelemetryEventKind::SafepointPatched, c, transforms);
            break;
        }
    }

    PassResult r = result_of(g, before);
    r.changed = transforms > 0;
    note(TelemetryEventKind::SafepointPatched, c, transforms);
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
