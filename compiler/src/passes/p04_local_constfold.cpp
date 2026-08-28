// =============================================================================
// Pass 04 — Local Constant Folding & Propagation.
//
// Evaluates pure operations with constant operands: int arithmetic (with
// overflow refusal — promotion to bignum stays a runtime concern), int
// comparisons, boolean identities on the numeric tower only. Effect-chained
// Python ops fold when both operands are constants (constant dispatch is
// total). Copy propagation runs through MOVE-shaped nodes at the IR level
// via RAUW after each fold.
// =============================================================================

#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

namespace {

bool try_fold(Graph& g, NodeId id) noexcept {
    Node& n = g.node(id);
    switch (n.kind) {
        case NodeKind::PyBinary: {
            if (n.ins.size() < 4) return false;
            const Node& a = g.node(n.ins[2]);
            const Node& b = g.node(n.ins[3]);
            if (a.kind != NodeKind::ConstInt || b.kind != NodeKind::ConstInt) return false;
            std::int64_t x = a.const_value.as_i(), y = b.const_value.as_i();
            std::int64_t out = 0;
            switch (static_cast<BinOpKind>(n.subop)) {
                case BinOpKind::Add:
                    if (__builtin_add_overflow(x, y, &out)) return false;
                    break;
                case BinOpKind::Sub:
                    if (__builtin_sub_overflow(x, y, &out)) return false;
                    break;
                case BinOpKind::Mul:
                    if (__builtin_mul_overflow(x, y, &out)) return false;
                    break;
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
            std::int64_t x = a.const_value.as_i(), y = b.const_value.as_i();
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

}  // namespace

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
    return result_of(g, before);
}

}  // namespace abi_v1
}  // namespace vortex::passes
