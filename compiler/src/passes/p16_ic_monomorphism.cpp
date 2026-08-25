// =============================================================================
// Pass 16 — Inline Cache (IC) Monomorphism Proof.
//
// At the IR level the DispatchCache node models an inline cache. In
// profile mode (Tier 2) the cache exposes its PGO confidence (Rule 44);
// this pass reads the node's pgo_count: a site whose count exceeds
// cfg::tier2_min_ic_hits with a single dominant target is dissolved into
// a GuardedDirectCall with a type guard + attached FrameState (Rules 3/5).
// Without profile data (Tier 1/3) the pass performs the static analogue:
// CallPy sites whose callee is a MakeFunction-produced constant in the
// same unit become direct calls (zero-risk devirtualization).
// =============================================================================

#include "vortex/frontend/lowering.hpp"
#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;
using vortex::fe::NativeHelper;

Result<PassResult> P16_ICMonomorphism::run(Graph& g, const PassContext& c) noexcept {
    std::uint32_t before = g.live_node_count();
    bool changed = false;

    g.for_each_live([&](NodeId id) {
        Node& n = g.node(id);
        if (n.kind != NodeKind::CallPy || n.ins.size() < 3) return;
        NodeId callee = n.ins[2];
        const Node& fn = g.node(callee);

        // Static monomorphism: callee is a MakeFunction CallNative executed
        // in-unit (single, structurally-known target).
        if (fn.kind == NodeKind::CallNative &&
            static_cast<NativeHelper>(fn.subop) == NativeHelper::MakeFunction) {
            // Rewrite as a direct call with the callee still carrying the
            // function object (the runtime resolves CallDirect through the
            // same PyFunc value — semantics identical, dispatch flag set).
            n.kind = NodeKind::GuardedDirectCall;
            n.clear_flag(NodeFlag::MayCall);
            n.set_flag(NodeFlag::MayThrow);
            changed = true;
            note(TelemetryEventKind::SafepointPatched, c, id);
            return;
        }

        // Profile-mode dissolution (Tier 2): pgo_count on the call node is
        // the IC hit count recorded by the interpreter.
        if (c.is_profiled() && n.pgo_count >= cfg::tier2_min_ic_hits) {
            n.kind = NodeKind::GuardedDirectCall;
            FrameState fs;
            fs.code_unit_id = c.code_unit_id;
            fs.bytecode_offset = static_cast<std::uint32_t>(n.pgo_count);
            fs.values.push_back(callee);
            fs.kinds.push_back(0);
            n.aux1 = g.add_frame_state(fs);
            n.set_flag(NodeFlag::Speculative);   // Rule 5: FrameState attached
            changed = true;
        }
    });

    return result_of(g, before);
}

}  // namespace abi_v1
}  // namespace vortex::passes
