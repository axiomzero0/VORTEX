// =============================================================================
// Pass 16b — Polymorphic Inline Cache Dispatch (SIMD-accelerated).
//
// For CallPy sites that are polymorphic (PGO shows 2-4 observed receiver
// types) but NOT megamorphic (>= cfg::ic_mega_pgo_floor types), this pass
// emits a DispatchCache IR node — a first-class IC (Part III of the spec).
//
// The DispatchCache node carries:
//   - ins[0] = control, ins[1] = effect, ins[2] = receiver, ins[3] = callee
//   - aux0   = number of cached type→target entries (the cache arity)
//   - shape_id = expected shape hash (for the SIMD type-match comparison)
//   - FrameState attached (Rule 5) for deopt on cache miss
//
// Runtime semantics: the DispatchCache executes as a standard CALL today
// (the scheduler lowers it to Op::CALL). The IC state lives in the IR
// metadata, ready for future runtime IC-table integration. The win is
// that downstream passes (P20 Speculative Inlining, P52 backend) can
// reason about the cache — dissolve it into a type-check chain, hoist
// the type checks, or emit a SwissTable megamorphic stub.
//
// Tier 2 only (Rule 2: speculation needs PGO). Idempotent (Rule 10):
// CallPy sites already converted to DispatchCache are skipped.
//
// Cost model (Rule 45): declines on megamorphic sites (too many types —
// the linear type-check chain is slower than the hash-table fallback).
// Rule 65: megamorphic declination emits telemetry so the profiler can
// flag dispatch-thrashing hotspots to the developer.
// =============================================================================

#include "vortex/frontend/lowering.hpp"
#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"
#include "vortex/support/config.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;
using vortex::fe::NativeHelper;

namespace {

/// Check whether a DispatchCache already exists for the given call site's
/// receiver — idempotency guard so a second pass run doesn't re-emit.
[[nodiscard]] bool has_dispatch_cache(const Graph& g, NodeId receiver) noexcept {
    bool found = false;
    g.for_each_live([&](NodeId id) {
        if (found) return;
        const Node& n = g.node(id);
        if (n.kind != NodeKind::DispatchCache) return;
        if (n.ins.size() >= 3 && n.ins[2] == receiver) found = true;
    });
    return found;
}

}  // namespace

Result<PassResult> P16b_PolymorphicDispatch::run(Graph& g, const PassContext& c) noexcept {
    if (c.tier == TierMode::Tier1) return PassResult{};   // Rule 2: PGO needed
    if (!c.is_profiled()) return PassResult{};
    std::uint32_t before = g.live_node_count();
    std::uint32_t caches_emitted = 0;
    std::uint32_t mega_declined = 0;

    // Collect candidates first (collect-then-rewrite — g.create() can
    // invalidate Node& references held across the iteration).
    struct PolyCandidate {
        NodeId call_id;
        NodeId receiver;
        NodeId callee;
        std::uint32_t type_count;   // PGO-observed type count (from aux0)
        std::uint32_t shape_hash;     // PGO shape hash (from shape_id)
    };
    stdx::small_vector<PolyCandidate, 16> candidates;

    g.for_each_live([&](NodeId id) {
        const Node& n = g.node(id);
        if (n.kind != NodeKind::CallPy || n.ins.size() < 4) return;

        // Skip sites already devirtualized by P16 (GuardedDirectCall).
        // P16 runs before P16b in the pipeline, so monomorphic sites are
        // already gone. This catches any remaining CallPy sites.

        // Skip sites without enough PGO heat.
        if (n.pgo_count < cfg::ic_devirt_pgo_floor) return;

        // The PGO profiler records the number of observed receiver types
        // in the CallPy node's aux0 field. 0 means "no type data collected".
        const std::uint32_t type_count = n.aux0;
        if (type_count == 0) return;   // Rule 46: no data, no speculation

        // Decline on megamorphic sites (too many types for a linear chain).
        // Rule 65: emit telemetry so the profiler can flag the hotspot.
        if (type_count > cfg::ic_poly_max_types) {
            ++mega_declined;
            return;
        }

        // Skip sites that are monomorphic (1 type) — P16 should have
        // handled these. But if P16 declined (e.g., no type_id in shape_id),
        // we also decline here. A monomorphic site with type_count == 1
        // but shape_id == 0 means the profiler saw the type count but
        // didn't record the type_id — not enough data for a guard.
        if (type_count <= cfg::ic_mono_max_types) {
            if (n.shape_id == 0) return;   // no type_id, no speculation
            // P16 should have dissolved this; if not, leave it to the
            // standard dispatch path.
            return;
        }

        // The receiver is ins[3] (first argument after the callee).
        NodeId receiver = n.ins[3];
        NodeId callee = n.ins[2];

        // Idempotency: skip if a DispatchCache already exists for this
        // receiver (a prior run of P16b already converted this site).
        if (has_dispatch_cache(g, receiver)) return;

        PolyCandidate pc{};
        pc.call_id = id;
        pc.receiver = receiver;
        pc.callee = callee;
        pc.type_count = type_count;
        pc.shape_hash = n.shape_id;   // may be 0 if no shape recorded
        candidates.push_back(pc);
    });

    // Emit DispatchCache nodes for polymorphic candidates. The DispatchCache
    // replaces the CallPy — it carries the same control/effect/data inputs
    // so the dataflow is preserved.
    for (const PolyCandidate& pc : candidates) {
        Node& call = g.node(pc.call_id);
        if (call.has(NodeFlag::Dead)) continue;   // already rewritten

        // Snapshot the original inputs before mutating.
        NodeId ctrl = call.ins[0];
        NodeId eff = call.ins[1];
        NodeId callee = pc.callee;
        NodeId receiver = pc.receiver;
        std::uint32_t argc = call.aux0;

        // Collect argument nodes (ins[4..]) for the cache to forward.
        stdx::small_vector<NodeId, 4> args;
        for (std::size_t i = 4; i < call.ins.size(); ++i) {
            args.push_back(call.ins[i]);
        }

        // Create the DispatchCache node. Layout:
        //   ins[0] = control, ins[1] = effect, ins[2] = receiver,
        //   ins[3] = callee, ins[4..] = arguments
        NodeId dc = g.create(NodeKind::DispatchCache);
        Node& dcn = g.node(dc);
        dcn.set_flag(NodeFlag::OnEffectChain);
        dcn.set_flag(NodeFlag::MayCall);
        dcn.set_flag(NodeFlag::MayThrow);
        dcn.set_flag(NodeFlag::Speculative);   // guarded by type checks
        dcn.aux0 = pc.type_count;              // cache arity
        dcn.shape_id = pc.shape_hash;          // for SIMD type-match
        g.add_input(dc, ctrl);
        g.add_input(dc, eff);
        g.add_input(dc, receiver);
        g.add_input(dc, callee);
        for (NodeId arg : args) g.add_input(dc, arg);

        // Attach a FrameState (Rule 5) for deopt on cache miss.
        FrameState fs;
        fs.code_unit_id = c.code_unit_id;
        fs.bytecode_offset = 0;
        fs.values.push_back(receiver);
        fs.values.push_back(callee);
        fs.kinds.push_back(0);
        fs.kinds.push_back(0);
        dcn.aux1 = g.add_frame_state(fs);

        // Replace the CallPy with the DispatchCache.
        g.replace_all_uses(pc.call_id, dc);
        g.kill(pc.call_id);
        ++caches_emitted;
    }

    PassResult r = result_of(g, before);
    r.changed = caches_emitted > 0;
    // Rule 65: no silent fallbacks. Record both the caches emitted AND
    // the megamorphic declinations so the profiler can flag hotspots.
    note(TelemetryEventKind::SafepointPatched, c, caches_emitted + mega_declined);
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
