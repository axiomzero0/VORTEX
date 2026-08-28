// =============================================================================
// Pass 05 — Algebraic Simplification (peephole, no e-graphs).
//
// Pattern rewrites on the SoA node stream: x+0, x-0, x|0, x&~0 -> x;
// x*1 -> x; x*0 -> 0 (pure x only); x^0 -> x. Only fires on PROVABLY
// numeric contexts (int constants on the identity operand), so Python
// operator overloads are never bypassed: for `x + 0` with x dynamic the
// rewrite is exactly Python semantics for ints; non-int x dispatches
// user __add__ — so the rewrite additionally requires the identity side
// to be the constant, which for +,-,|,& with an int operand cannot change
// observable behavior when x is an int, and when x is not an int the
// operation raises TypeError which 0-identities preserve only partially —
// therefore dynamic-x cases are skipped (the conservative, correct rule).
// =============================================================================

#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

namespace {

bool try_rewrite(Graph& g, NodeId id) noexcept {
    Node& n = g.node(id);
    if (n.kind != NodeKind::PyBinary || n.ins.size() < 4) return false;
    const Node& a = g.node(n.ins[2]);
    const Node& b = g.node(n.ins[3]);

    auto replace_with = [&](NodeId v) noexcept {
        g.replace_all_uses(id, v);
        g.kill(id);
        return true;
    };
    auto is_int_const = [](const Node& x, std::int64_t want) noexcept {
        return x.kind == NodeKind::ConstInt && x.const_value.as_i() == want;
    };

    // PASS-1/PASS-2 fix: every rewrite that drops an operand (x & 0 -> 0,
    // x * 0 -> 0) must verify the dropped operand is pure — otherwise
    // we silently drop side effects (TypeError, user __and__, etc.).
    // Every rewrite that REPLACES x op k with x (x+0, x|0, x&-1, x^0,
    // x*1, x-0) must verify x is provably int-typed — otherwise we
    // bypass a user __add__/__or__/etc. that may return a non-x value
    // or raise. The conservative check is x.has(NodeFlag::Pure) for
    // side-effect safety AND (x.kind == ConstInt OR x.type_tag is
    // integer-ish) for type safety. Since type_tag is set by shape
    // analysis later in the pipeline, the safe subset here is:
    //   - replace_with x: only if x is Pure (no observable dispatch)
    //   - replace_with 0: only if x is Pure (no dropped side effects)
    // Pure data nodes (ConstInt, ConstFloat, arithmetic on pure
    // operands) qualify. Effectful nodes (CallPy, LoadGlobal, attribute
    // access) do not.
    auto x_is_pure = [](const Node& x) noexcept {
        return x.has(NodeFlag::Pure);
    };

    switch (static_cast<BinOpKind>(n.subop)) {
        case BinOpKind::Add:
            // x + 0 -> x: safe only if x is pure (no __add__ dispatch).
            if (is_int_const(b, 0) && x_is_pure(a)) return replace_with(n.ins[2]);
            if (is_int_const(a, 0) && x_is_pure(b)) return replace_with(n.ins[3]);
            return false;
        case BinOpKind::Sub:
            if (is_int_const(b, 0) && x_is_pure(a)) return replace_with(n.ins[2]);
            return false;
        case BinOpKind::BitOr:
            if (is_int_const(b, 0) && x_is_pure(a)) return replace_with(n.ins[2]);
            if (is_int_const(a, 0) && x_is_pure(b)) return replace_with(n.ins[3]);
            return false;
        case BinOpKind::BitAnd:
            if (is_int_const(b, -1) && x_is_pure(a)) return replace_with(n.ins[2]);
            if (is_int_const(a, -1) && x_is_pure(b)) return replace_with(n.ins[3]);
            // PASS-1 fix: x & 0 -> 0 requires x to be pure. The previous
            // code fired unconditionally, dropping x's side effects
            // (TypeError for non-int x, user __and__, etc.) and
            // replacing with the constant b. With the purity guard, we
            // only fire when x is pure (no observable dispatch).
            if (is_int_const(b, 0) && x_is_pure(a)) {
                return replace_with(n.ins[3]);   // x & 0 == 0 (drop pure x)
            }
            return false;
        case BinOpKind::BitXor:
            if (is_int_const(b, 0) && x_is_pure(a)) return replace_with(n.ins[2]);
            return false;
        case BinOpKind::Mul:
            if (is_int_const(b, 1) && x_is_pure(a)) return replace_with(n.ins[2]);
            if (is_int_const(a, 1) && x_is_pure(b)) return replace_with(n.ins[3]);
            // x * 0 -> 0 only when x is a pure node (no dispatch effects).
            if (is_int_const(b, 0) && a.has(NodeFlag::Pure)) {
                NodeId zero = g.create(NodeKind::ConstInt);
                g.node(zero).const_value = Value::integer(0);
                g.node(zero).set_flag(NodeFlag::Pure);
                return replace_with(zero);
            }
            return false;
        default: return false;
    }
}

}  // namespace

Result<PassResult> P05_AlgebraicSimplification::run(Graph& g, const PassContext& c) noexcept {
    (void)c;
    std::uint32_t before = g.live_node_count();
    bool changed = true;
    while (changed) {
        changed = false;
        g.for_each_live([&](NodeId id) {
            if (try_rewrite(g, id)) changed = true;
        });
    }
    return result_of(g, before);
}

}  // namespace abi_v1
}  // namespace vortex::passes
