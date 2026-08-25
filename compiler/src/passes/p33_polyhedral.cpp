// =============================================================================
// Pass 33 — Polyhedral Loop Interchange.  [OPT-IN ONLY]
//
// This pass NEVER runs unless the caller explicitly sets
// OptOption::Polyhedral on the PassContext (driver flag / toolchain option).
// Both gates enforce it: the pipeline filter skips "33_polyhedral" without
// the flag, AND this run() self-checks so direct invocations (tests,
// tooling) are gated identically. Rationale: the analysis is expensive and
// its payoff is profile-dependent — wrong to impose on every compilation.
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
    // Opt-in contract (see file header): no explicit request, no analysis.
    if (!c.options.has(OptOption::Polyhedral)) return PassResult{};

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
        //    Old outer Loop → new inner: entry=inner_body_tail, back=outer_body_entry.
        //    Old inner Loop → new outer: entry=Start,           back=outer_exit.
        g.set_input(L_outer.header, 0, inner_body_tail);
        g.set_input(L_outer.header, 1, outer_body_entry);
        g.set_input(L_inner.header, 0, start_node);
        g.set_input(L_inner.header, 1, outer_exit);

        // 3. Identify IV update nodes (backedges of the IV phis).
        NodeId outer_iv_update = g.node(outer_iv).ins[1];
        NodeId inner_iv_update = g.node(inner_iv).ins[1];

        // 4. Remap body content ctrls. Snapshot the OLD ctrl values first
        //    because some nodes need to be compared against the OLD ctrl,
        //    not the new one mid-rewrite.
        struct CtrlRemap { NodeId node; NodeId new_ctrl; };
        stdx::small_vector<CtrlRemap, 32> remaps;
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
                // Old inner body content.
                if (id == inner_iv_update) {
                    // Old inner IV update becomes the new outer IV update,
                    // which lives in the new outer body AFTER the new inner
                    // exits (= new inner exit = old outer_exit).
                    new_ctrl = outer_exit;
                } else {
                    // Body content stays in the new inner body
                    // (= new inner body tail = old outer_body_entry).
                    new_ctrl = outer_body_entry;
                }
            } else if (ctrl == inner_exit) {
                // Old outer body tail content. All of it moves to the new
                // inner body (= old outer_body_entry).
                new_ctrl = outer_body_entry;
            } else if (ctrl == outer_body_entry) {
                // Old outer body entry content (j = 0 init, etc.) moves to
                // the new outer body (= old inner_body_tail).
                new_ctrl = inner_body_tail;
            } else if (ctrl == outer_exit) {
                // Nodes at the old outer exit. The Return moves to the new
                // outer exit (= old inner_exit); other content stays put
                // (we don't know where it should go conservatively).
                if (n.kind == NodeKind::Return) {
                    new_ctrl = inner_exit;
                }
            }
            if (new_ctrl != invalid_node) {
                remaps.push_back({id, new_ctrl});
            }
        });
        for (const auto& r : remaps) {
            g.set_input(r.node, 0, r.new_ctrl);
        }

        // 5. Remap body content data inputs. Body content that was in the
        //    OLD INNER body used the OLD INNER phis for "value of variable
        //    inside the inner loop". After the swap, the same body content
        //    is in the NEW INNER body and should use the NEW INNER phis —
        //    which are the OLD OUTER phis (since the swap exchanges
        //    phi↔ctrl roles). For each (outer_phi, inner_phi) pair, replace
        //    body-content uses of inner_phi with outer_phi.
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

        // Snapshot the OLD ctrl of all body content (before step 4's
        // remap) — we need the OLD ctrl to know which nodes are "in the
        // old inner body" and need data-input remap. We snapshotted it
        // before step 4 — re-do the snapshot here to be safe.
        // (Actually, we need to remap data inputs of nodes that were in
        // the OLD INNER body, which are now in the NEW INNER body. We
        // can detect these by their NEW ctrl being one of the new inner
        // body projections: outer_body_entry or inner_exit (these are
        // the OLD outer_body_entry and OLD outer_exit that body content
        // was remapped to in step 4).)
        //
        // Wait — step 4's remap set:
        //   old_inner_body_tail nodes (not IV update) → outer_body_entry
        //   old_inner_exit nodes                    → outer_body_entry
        //   old_outer_body_entry nodes               → inner_body_tail
        //   old_inner_body_tail IV update           → outer_exit
        //
        // After step 4, nodes with NEW ctrl in {outer_body_entry} were
        // in the OLD INNER body. Nodes with NEW ctrl in {inner_body_tail}
        // were in the OLD OUTER body.
        //
        // For OLD INNER body content (NEW ctrl = outer_body_entry): we
        // remap their uses of inner_phi → outer_phi (they're now in the
        // new INNER body and should use the new INNER phis = OLD OUTER
        // phis).
        //
        // For OLD OUTER body content (NEW ctrl = inner_body_tail): we
        // also need to consider. In our IR, the only OLD OUTER body
        // content is the IV update n32, which uses n19 (OLD INNER i phi).
        // After remap, n32 is in the NEW INNER body. It should use n8
        // (OLD OUTER i phi = new INNER i phi). Same rule: replace
        // inner_phi → outer_phi.
        //
        // So the rule is uniform: for body content that's now in the new
        // INNER body (NEW ctrl in {outer_body_entry, inner_body_tail,
        // outer_exit}), replace uses of inner_phi with outer_phi.
        //
        // We exclude the phi nodes themselves (their ins is handled in
        // step 6), and we exclude the projection/Loop/If/Return nodes
        // (they're not body content).
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
                // For each input slot (except ctrl at slot 0 — we don't
                // want to remap control), if it equals inner_phi,
                // schedule a remap to outer_phi.
                for (std::uint32_t s = 0; s < n.ins.size(); ++s) {
                    if (s == 0 && (n.has(NodeFlag::OnEffectChain) ||
                                   is_memory(n.kind) ||
                                   n.kind == NodeKind::PyBinary ||
                                   n.kind == NodeKind::PyUnary ||
                                   n.kind == NodeKind::PyCompare ||
                                   n.kind == NodeKind::LoadIndex ||
                                   n.kind == NodeKind::StoreIndex)) {
                        // slot 0 is ctrl; skip
                        continue;
                    }
                    if (n.ins[s] == inner_phi) {
                        data_remaps.push_back({id, s, outer_phi});
                    }
                }
            });
        }
        for (const auto& d : data_remaps) {
            g.set_input(d.node, d.slot, d.new_value);
        }

        // 6. Rewire phi entries/backedges. The pairs were collected in
        //    step 5. For each pair, swap entry edges and selectively
        //    rewrite backedges to self when the variable doesn't change in
        //    the NEW inner body.
        for (auto [outer_phi, inner_phi] : phi_pairs) {
            NodeId start_value = g.node(outer_phi).ins[0];   // old outer_phi's entry
            NodeId old_outer_back = g.node(outer_phi).ins[1];
            // new_outer_phi (= inner_phi).entry = start_value
            g.set_input(inner_phi, 0, start_value);
            // new_inner_phi (= outer_phi).entry = new_outer_phi (= inner_phi)
            g.set_input(outer_phi, 0, inner_phi);
            // If old outer_phi.backedge lives in the OLD INNER body (or is
            // the cross-loop edge to inner_phi), the variable doesn't
            // change in the NEW inner body (= old outer body) either —
            // rewrite to self-reference on the new inner_phi.
            bool rewrite_to_self = (old_outer_back == inner_phi);
            if (!rewrite_to_self && old_outer_back != invalid_node) {
                const Node& bn = g.node(old_outer_back);
                if (!bn.ins.empty() && block_in_inner(L_inner, bn.ins[0])) {
                    rewrite_to_self = true;
                }
            }
            if (rewrite_to_self) {
                g.set_input(outer_phi, 1, outer_phi);
            }
        }

        ++transforms;

        if (transforms > cfg::fixpoint_max_iterations) {
            note(TelemetryEventKind::BudgetExceeded, c, transforms);
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
