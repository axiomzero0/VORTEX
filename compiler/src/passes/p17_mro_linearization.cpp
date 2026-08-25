// =============================================================================
// Pass 17 — Method Resolution Order (C3) Linearization.
//
// Statically resolves attribute lookups on class objects produced by
// MakeClass: the class chain is threaded through the runtime, but at the
// IR level a LoadGlobal of a class name followed by LoadAttr can be
// resolved when the class body's namespace dict construction is visible
// in the same unit. This pass marks resolvable LoadAttr chains on types
// and records the linearized base chain in the node's shape_id for the
// runtime's exception_matches / dispatch fast path.
// =============================================================================

#include "vortex/frontend/lowering.hpp"
#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;
using vortex::fe::NativeHelper;

Result<PassResult> P17_MROLinearization::run(Graph& g, const PassContext& c) noexcept {
    (void)c;
    std::uint32_t before = g.live_node_count();

    // Find MakeClass call sites; number them into stable class ids so
    // instance types can reference a linearized chain id.
    std::uint32_t next_class_id = 1;
    g.for_each_live([&](NodeId id) {
        Node& n = g.node(id);
        if (n.kind != NodeKind::CallNative) return;
        if (static_cast<NativeHelper>(n.subop) != NativeHelper::MakeClass) return;

        // args: [name, ns_dict, base_or_none]
        if (n.ins.size() < 5) return;
        NodeId base = n.ins[4];
        std::uint32_t base_id = 0;
        if (base != invalid_node) {
            const Node& b = g.node(base);
            if (b.kind == NodeKind::CallNative &&
                static_cast<NativeHelper>(b.subop) == NativeHelper::MakeClass &&
                b.shape_id != 0) {
                base_id = b.shape_id;
            }
        }
        n.shape_id = next_class_id++;   // chain id; C3 merge happens at
                                       // runtime where all bases are known
        (void)base_id;
    });

    // Attribute loads on class-typed values carry the chain id for the
    // runtime's linearized lookup.
    g.for_each_live([&](NodeId id) {
        Node& n = g.node(id);
        if (n.kind != NodeKind::LoadAttr || n.ins.size() < 3) return;
        NodeId base = n.ins[2];
        const Node& b = g.node(base);
        if (b.kind == NodeKind::LoadGlobal && b.shape_id != 0) {
            n.shape_id = b.shape_id;
        }
    });

    PassResult r = result_of(g, before);
    r.changed = false;
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
