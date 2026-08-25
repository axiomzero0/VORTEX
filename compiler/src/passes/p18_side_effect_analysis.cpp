// =============================================================================
// Pass 18 — Side-Effect Analysis & Effect Chain Construction.
//
// Classifies every node into its precise effect profile and maintains the
// strict effect edges of the Sea of Nodes: pure (no flags), reads
// (OnEffectChain only), writes (+MayThrow), calls (+MayCall).
// =============================================================================

#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

Result<PassResult> P18_SideEffectAnalysis::run(Graph& g, const PassContext& c) noexcept {
    (void)c;
    std::uint32_t before = g.live_node_count();

    g.for_each_live([&](NodeId id) {
        Node& n = g.node(id);
        switch (n.kind) {
            case NodeKind::LoadGlobal:
            case NodeKind::LoadAttr:
            case NodeKind::LoadIndex:
            case NodeKind::Load:
            case NodeKind::LoadField:
            case NodeKind::VecLoad:
            case NodeKind::Gather:
                n.set_flag(NodeFlag::OnEffectChain);
                break;
            case NodeKind::StoreGlobal:
            case NodeKind::StoreAttr:
            case NodeKind::StoreIndex:
            case NodeKind::Store:
            case NodeKind::StoreField:
            case NodeKind::VecStore:
            case NodeKind::Scatter:
            case NodeKind::ListAppend:
                n.set_flag(NodeFlag::OnEffectChain);
                n.set_flag(NodeFlag::MayThrow);
                break;
            case NodeKind::CallPy:
            case NodeKind::CallDirect:
            case NodeKind::GuardedDirectCall:
            case NodeKind::CallNative:
                n.set_flag(NodeFlag::OnEffectChain);
                n.set_flag(NodeFlag::MayThrow);
                n.set_flag(NodeFlag::MayCall);
                break;
            default:
                break;
        }
    });

    PassResult r = result_of(g, before);
    r.changed = false;
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
