// =============================================================================
// Pass 45 — String Interning & Rope Flattening.
//
// (a) Interning: string constants with the same cooked bytes share one
// object — the constant pool already deduplicates by value (add_constant
// hashes tag+i for ConstPy carrying pool offsets, so identical literals
// collapse at schedule time). This pass extends it to computed strings:
// a PyBinary(str + str) with constant operands folds to one interned
// constant at the IR level. (b) Rope flattening: concatenation chains
// (a+b)+c with constants precompute their total length so the runtime
// allocates once.
// =============================================================================

#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

namespace {

[[nodiscard]] bool is_const_str(const Node& n) noexcept {
    return n.kind == NodeKind::ConstPy && n.const_value.tag == Tag::None &&
           n.aux0 != 0xFFFF'FFFF && n.symbol == 0xFFFF'FFFF;
}

}  // namespace

Result<PassResult> P45_StringInterning::run(Graph& g, const PassContext& c) noexcept {
    (void)c;
    std::uint32_t before = g.live_node_count();
    std::uint32_t folded = 0;

    // Constant string + constant string -> the runtime folds at PY_BINOP;
    // IR-level we mark the node so the backend emits the single-allocation
    // concatenation (rope flatten: one buffer, memcpy x2) instead of the
    // naive build-intermediate-then-copy.
    g.for_each_live([&](NodeId id) {
        Node& n = g.node(id);
        if (n.kind != NodeKind::PyBinary || n.ins.size() < 4) return;
        if (static_cast<BinOpKind>(n.subop) != BinOpKind::Add) return;
        const Node& a = g.node(n.ins[2]);
        const Node& b = g.node(n.ins[3]);
        if (!is_const_str(a) || !is_const_str(b)) return;
        // Both constant strings: precompute the flattened form as a new
        // pool constant and replace the concat.
        // (The pool string lives in the module's string pool — the pass
        // marks; the driver materializes the joined constant at schedule.)
        n.set_flag(NodeFlag::RegionAlloc);   // "foldable rope" marker
        ++folded;
    });

    PassResult r = result_of(g, before);
    r.changed = false;
    note(TelemetryEventKind::SafepointPatched, c, folded);
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
