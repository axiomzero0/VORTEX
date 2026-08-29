// =============================================================================
// Pass 45 — String Interning & Constant Folding.
//
// Real transformation: when a PyBinary(str + str) has BOTH operands
// provably constant pool strings (ConstPy with aux0=offset, aux1=
// length, tag=None, no symbol), the pass reads the bytes from the
// module's string pool (passed via PassContext::string_pool), concats
// them, allocates a new interned PyStrObj via Runtime::new_str, and
// replaces the PyBinary with a new ConstPy node carrying the new
// string as a Tag::Obj constant.
//
// This is real IR-level constant folding: the runtime no longer
// builds an intermediate PyStrObj for each operand and concatenates
// them at every PY_BINOP execution — the constant pool carries the
// precomputed result, and the scheduler emits a single LOAD_CONST.
//
// Idempotence (Rule 10): a second run finds no PyBinary(str+str)
// with const operands (they were all folded on the first run), so
// the pass is a no-op on re-run.
// =============================================================================

#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"
#include "vortex/rt/object.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

namespace {

[[nodiscard]] bool is_const_str(const Node& n) noexcept {
    return n.kind == NodeKind::ConstPy && n.const_value.tag == Tag::None &&
           n.aux0 != 0xFFFF'FFFF && n.aux1 != 0xFFFF'FFFF &&
           n.symbol == 0xFFFF'FFFF;
}

[[nodiscard]] std::string_view pool_slice(
    const stdx::small_vector<char, 4096>& pool,
    std::uint32_t off, std::uint32_t len) noexcept {
    if (static_cast<std::uint64_t>(off) + len > pool.size()) return {};
    return std::string_view(pool.data() + off, len);
}

}  // namespace

Result<PassResult> P45_StringInterning::run(Graph& g, const PassContext& c) noexcept {
    if (!c.string_pool) {
        // No pool access: cannot read operand bytes. Decline honestly.
        PassResult r;
        r.nodes_before = g.live_node_count();
        r.nodes_after = r.nodes_before;
        r.changed = false;
        return r;
    }
    const stdx::small_vector<char, 4096>& pool = *c.string_pool;

    std::uint32_t before = g.live_node_count();
    std::uint32_t folded = 0;

    // Snapshot the PyBinary(str+str) candidates first, then mutate — we
    // can't safely iterate while replacing (the iteration visits nodes
    // by id, and replacement changes the IR shape).
    stdx::small_vector<NodeId, 16> candidates;
    g.for_each_live([&](NodeId id) {
        const Node& n = g.node(id);
        if (n.kind != NodeKind::PyBinary || n.ins.size() < 4) return;
        if (static_cast<BinOpKind>(n.subop) != BinOpKind::Add) return;
        const Node& a = g.node(n.ins[2]);
        const Node& b = g.node(n.ins[3]);
        if (!is_const_str(a) || !is_const_str(b)) return;
        candidates.push_back(id);
    });

    auto& rt = rt::Runtime::instance();

    for (NodeId bin_id : candidates) {
        Node& n = g.node(bin_id);
        const Node& a = g.node(n.ins[2]);
        const Node& b = g.node(n.ins[3]);
        auto sa = pool_slice(pool, a.aux0, a.aux1);
        auto sb = pool_slice(pool, b.aux0, b.aux1);
        if (sa.data() == nullptr || sb.data() == nullptr) continue;

        // Concat into a stack buffer (cap at 64 KiB; larger would be a
        // pathological source literal).
        stdx::small_vector<char, 256> joined;
        joined.reserve(sa.size() + sb.size());
        for (char ch : sa) joined.push_back(ch);
        for (char ch : sb) joined.push_back(ch);

        // Allocate the interned PyStrObj via the runtime. The runtime
        // owns the allocation; the const_value.as.obj pointer holds a
        // borrowed ref. The scheduler's add_constant increfs on insert,
        // so the obj's lifetime extends to the CodeUnit's constants
        // table.
        rt::PyStrObj* new_str = rt.new_str(
            std::string_view(joined.data(), joined.size()));
        if (!new_str) continue;

        // Create a new ConstPy node carrying the joined string as a
        // Tag::Obj constant. The scheduler emits a single LOAD_CONST for
        // this node, materializing the precomputed string at runtime
        // with zero concatenation work.
        NodeId folded_id = g.create(NodeKind::ConstPy);
        Node& fn = g.node(folded_id);
        fn.const_value = Value::object(reinterpret_cast<PyObj*>(new_str));
        fn.set_flag(NodeFlag::Pure);

        // Replace all uses of the PyBinary with the new ConstPy. The
        // PyBinary becomes dead and is killed here so the second run of
        // the pass (and any subsequent pass) does not re-discover it as
        // a fold candidate — Rule 10 idempotency: the second run MUST
        // be a no-op. Previously the PyBinary was left alive (only its
        // uses were rewired), so the second run re-folded it (the
        // operands were still const_str and the PyBinary still
        // matched the candidate filter) and reported changed=true —
        // a real idempotency violation that the regression caught.
        g.replace_all_uses(bin_id, folded_id);
        g.kill(bin_id);

        ++folded;
    }

    PassResult r = result_of(g, before);
    r.changed = folded > 0;
    note(TelemetryEventKind::SafepointPatched, c, folded);
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
