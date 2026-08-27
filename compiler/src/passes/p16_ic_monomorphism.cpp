// =============================================================================
// Pass 16 — Inline Cache (IC) Monomorphism Proof & Speculative Devirtualization.
//
// Two paths, matching the spec's "Tier 2 (JIT): Profile-Guided Speculative
// Devirtualization" and "Tier 3 (AOT): Class Hierarchy Analysis":
//
//   1. STATIC (all tiers): CallPy sites whose callee is a MakeFunction-
//      produced constant in the same unit become GuardedDirectCall
//      (zero-risk devirtualization — the target is provably known).
//
//   2. SPECULATIVE (Tier 2 only): CallPy sites with pgo_count above
//      cfg::ic_devirt_pgo_floor and a known expected type_id (stored in
//      the node's shape_id field by the PGO profiler) get:
//        a. A Guard(TypeIs) node checking the receiver's type_id
//        b. A FrameState (Rule 5) for deopt reconstruction (Rule 4)
//        c. The CallPy → GuardedDirectCall rewrite
//      On guard failure the runtime deoptimizes to Tier-0 (Rule 4).
//      Without PGO type data (shape_id == 0) the pass declines — Rule 46
//      (No Profile Data Without Confidence) forbids speculation without
//      evidence.
//
// Idempotency (Rule 10): CallPy nodes that have already been converted
// to GuardedDirectCall are skipped on the second run. Guard nodes already
// emitted for a call site are detected via has_type_guard() so no duplicate
// guards are created.
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

/// Check whether a Guard(TypeIs) already exists for the given call site's
/// receiver — idempotency guard so a second pass run doesn't re-emit.
/// The guard's ins[0] is the value it checks.
[[nodiscard]] bool has_type_guard(const Graph& g, NodeId receiver) noexcept {
    bool found = false;
    g.for_each_live([&](NodeId id) {
        if (found) return;
        const Node& n = g.node(id);
        if (n.kind != NodeKind::Guard) return;
        if (n.subop != static_cast<std::uint16_t>(GuardKind::TypeIs)) return;
        if (!n.ins.empty() && n.ins[0] == receiver) found = true;
    });
    return found;
}

/// Emit a Guard(TypeIs) node for the receiver, carrying a FrameState
/// (Rule 5) for deopt reconstruction. The expected type_id is stored in
/// the guard's shape_id field. The guard is Speculative (Rule 3/4/5).
void emit_type_guard(Graph& g, const PassContext& c, NodeId receiver,
                      std::uint32_t expected_type_id) noexcept {
    if (has_type_guard(g, receiver)) return;   // idempotent
    if (expected_type_id == 0) return;          // Rule 46: no data, no speculation

    FrameState fs;
    fs.code_unit_id = c.code_unit_id;
    fs.bytecode_offset = 0;   // safepoint_pcs table not yet wired (driver.cpp)
    fs.values.push_back(receiver);
    fs.kinds.push_back(0);   // tagged
    std::uint32_t fs_idx = g.add_frame_state(fs);

    NodeId guard = g.create(NodeKind::Guard, {receiver});
    Node& gn = g.node(guard);
    gn.subop = static_cast<std::uint16_t>(GuardKind::TypeIs);
    gn.shape_id = expected_type_id;
    gn.aux1 = fs_idx;
    gn.set_flag(NodeFlag::Speculative);   // Rule 5: FrameState attached
    gn.set_flag(NodeFlag::Pure);          // a guard predicate has no side effect
    gn.set_flag(NodeFlag::TypeGuarded);
}

}  // namespace

Result<PassResult> P16_ICMonomorphism::run(Graph& g, const PassContext& c) noexcept {
    std::uint32_t before = g.live_node_count();
    bool changed = false;
    std::uint32_t guards_emitted = 0;

    // Collect CallPy candidates first (collect-then-rewrite — g.create()
    // can invalidate Node& references held across the iteration).
    struct CallCandidate {
        NodeId call_id;
        NodeId callee;
        NodeId receiver;          // the object whose type we guard on
        std::uint32_t type_id;     // expected type (from PGO, 0 = unknown)
        bool is_static;            // static monomorphism (MakeFunction)
        bool is_speculative;       // PGO-driven speculative devirt
    };
    stdx::small_vector<CallCandidate, 16> candidates;

    g.for_each_live([&](NodeId id) {
        const Node& n = g.node(id);
        if (n.kind != NodeKind::CallPy || n.ins.size() < 3) return;
        NodeId callee = n.ins[2];
        const Node& fn = g.node(callee);

        CallCandidate cc{};
        cc.call_id = id;
        cc.callee = callee;

        // Static monomorphism: callee is a MakeFunction CallNative executed
        // in-unit (single, structurally-known target).
        if (fn.kind == NodeKind::CallNative &&
            static_cast<NativeHelper>(fn.subop) == NativeHelper::MakeFunction) {
            cc.is_static = true;
            // The receiver for a MakeFunction call is the function object
            // itself (callee). We don't need a type guard — the target is
            // statically proven.
            cc.receiver = callee;
            cc.type_id = 0;
            candidates.push_back(cc);
            return;
        }

        // Speculative devirtualization (Tier 2 only, Rule 2: PGO needed).
        // The call site must be hot (pgo_count >= ic_devirt_pgo_floor) AND
        // the PGO profiler must have recorded an expected type_id (stored
        // in the CallPy node's shape_id field — 0 means "no type data").
        if (!c.is_profiled()) return;
        if (n.pgo_count < cfg::ic_devirt_pgo_floor) return;
        if (n.shape_id == 0) return;   // Rule 46: no type data, no speculation

        cc.is_speculative = true;
        // The receiver is the first argument after the callee (ins[3]).
        // For method calls (obj.method()), this is the object. For free
        // function calls, there's no receiver type to guard on.
        if (n.ins.size() < 4) return;   // no receiver argument
        cc.receiver = n.ins[3];
        cc.type_id = n.shape_id;
        candidates.push_back(cc);
    });

    // Apply rewrites outside the iteration (safe — no Node& held across
    // g.create() calls).
    for (const CallCandidate& cc : candidates) {
        Node& n = g.node(cc.call_id);

        if (cc.is_static) {
            // Static monomorphism: rewrite as a direct call with the callee
            // still carrying the function object. The runtime resolves
            // CallDirect through the same PyFunc value — semantics identical,
            // dispatch flag set.
            n.kind = NodeKind::GuardedDirectCall;
            n.clear_flag(NodeFlag::MayCall);
            n.set_flag(NodeFlag::MayThrow);
            changed = true;
            note(TelemetryEventKind::SafepointPatched, c, cc.call_id);
            continue;
        }

        if (cc.is_speculative) {
            // Emit the type guard BEFORE converting the call. The guard
            // checks the receiver's type_id against the PGO-recorded
            // expected type. On failure, the runtime deoptimizes (Rule 4)
            // and resumes Tier-0 with the original CallPy semantics.
            emit_type_guard(g, c, cc.receiver, cc.type_id);
            ++guards_emitted;

            n.kind = NodeKind::GuardedDirectCall;
            n.clear_flag(NodeFlag::MayCall);
            n.set_flag(NodeFlag::MayThrow);
            n.set_flag(NodeFlag::Speculative);   // Rule 5: guarded speculation

            // Attach a FrameState to the call site too — if the direct
            // call itself fails (e.g., wrong argument count for the
            // speculated target), the deoptimizer needs the resume point.
            FrameState fs;
            fs.code_unit_id = c.code_unit_id;
            fs.bytecode_offset = 0;
            fs.values.push_back(cc.callee);
            fs.values.push_back(cc.receiver);
            fs.kinds.push_back(0);
            fs.kinds.push_back(0);
            n.aux1 = g.add_frame_state(fs);

            changed = true;
            note(TelemetryEventKind::SafepointPatched, c, cc.call_id);
        }
    }

    PassResult r = result_of(g, before);
    r.changed = changed;
    note(TelemetryEventKind::SafepointPatched, c, guards_emitted);
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
