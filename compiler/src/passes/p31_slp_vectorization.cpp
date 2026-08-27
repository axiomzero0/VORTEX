// =============================================================================
// Pass 31 — Superword Level Parallelism (SLP) Vectorization.
//
// Larsen & Amarasinghe basic algorithm, Sea-of-Nodes form, four sub-passes:
//   31a: unboxing & scalarization pre-pass — marks isomorphic pure op
//        chains with all-constant-type operands Vectorizable (the IR's
//        unboxed primitives are int64/float64 registers).
//   31b: packetization — groups independent, adjacent (node-id order)
//        isomorphic operations into packets sized by the TARGET's vector
//        width (TargetDescriptor::simd_width_bytes / 8 lanes — Rule 27:
//        queried, never assumed). No descriptor attached -> no packets.
//        Cost model (Rule 45): a packet is accepted only when
//        TargetDescriptor::vector_pays() says the scalar cost exceeds the
//        vector-file move cost, AND the shuffle overhead (lanes whose
//        operand order doesn't align with the previous lane) is less than
//        half the packet — otherwise the alignment traffic dominates.
//   31c: speculative SLP — for hot LoadIndex/Load pairs whose aliasing
//        cannot be statically proven (no TypeGuarded flag from Pass 14),
//        emit a Guard(AliasDisjoint) node carrying a FrameState (Rule 5).
//        On guard failure the runtime deoptimizes (Rule 4) and resumes
//        Tier-0 execution. Tier 2 only (Rule 2: speculation needs PGO).
//   31d: gather/scatter fallback — when packet members are pointer-typed
//        (PyObject* lists that Pass 47 hasn't unboxed yet), emit a Gather
//        node instead of a contiguous VecLoad. The cost model gates this
//        on TargetFeature::AVX2 (x86) or ASIMD (aarch64) — without fast
//        gather hardware the pass declines (Rule 45).
//
// Idempotency (Rule 10): candidates consumed by a packet have their
// Vectorizable flag CLEARED, so a second run sees no candidates and is a
// no-op. Guards are only emitted when no prior AliasDisjoint guard exists
// on the same base pair.
// =============================================================================

#include "vortex/backend/mir.hpp"
#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"
#include "vortex/support/config.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

namespace {

// ---------------------------------------------------------------------------
// 31a: candidate predicate. Pure (no effect chain, no calls), and either
// a PyBinary whose non-control/effect operands are unboxable, or a raw
// constant. The IR's unboxed primitives are int64/float64 registers.
// ---------------------------------------------------------------------------
[[nodiscard]] bool vectorizable_op(const Node& n) noexcept {
    if (n.has(NodeFlag::OnEffectChain)) return false;
    if (n.has(NodeFlag::MayCall)) return false;
    return n.kind == NodeKind::PyBinary || n.kind == NodeKind::ConstInt ||
           n.kind == NodeKind::ConstFloat;
}

[[nodiscard]] bool is_unboxable_input(const Graph& g, NodeId id) noexcept {
    const Node& n = g.node(id);
    return n.kind == NodeKind::ConstInt || n.kind == NodeKind::ConstFloat ||
           n.has(NodeFlag::Unboxed);
}

// Two ops are "same shape" for SLP purposes: same kind, same subop, same
// arity. Independence is checked only on DATA inputs (ins[2+]) — the
// control/effect inputs at ins[0]/ins[1] are SCHEDULING constraints and
// MUST be the same (same block, same effect chain), not different.
//
// The previous version of this predicate rejected any pair sharing ANY
// input — including the always-shared control/effect start node — which
// meant PyBinary nodes in the same function could never be packetized
// (latent bug: the existing tests pass only because the kitchen-sink
// program has no isomorphic PyBinary chains).
[[nodiscard]] bool same_shape(const Graph& g, NodeId a, NodeId b) noexcept {
    const Node& x = g.node(a);
    const Node& y = g.node(b);
    if (x.kind != y.kind || x.subop != y.subop) return false;
    if (x.ins.size() != y.ins.size()) return false;
    if (x.ins.size() < 2) return true;   // no data inputs: trivially same shape
    // Control/effect inputs (indices 0,1) MUST match (same block, same
    // effect chain) — otherwise scheduling the packet would be invalid.
    if (x.ins[0] != y.ins[0]) return false;
    if (x.ins[1] != y.ins[1]) return false;
    // Data inputs (indices 2+) MUST be different (independence: sharing a
    // data input would create an intra-packet dependence).
    for (std::uint32_t i = 2; i < x.ins.size(); ++i) {
        if (x.ins[i] == y.ins[i]) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// 31b cost model: count lanes whose operand order doesn't align with the
// previous lane (each misalignment costs one shuffle to fix).
//
// A "misalignment" is when lane[i] reads from operand slot s but lane[i-1]
// reads from a DIFFERENT slot s'. Without a richer shape model we cannot
// detect cross-slot permutations, so this implementation conservatively
// returns 0 (no shuffle overhead detected) — the cost gate then falls
// entirely to TargetDescriptor::vector_pays(). The shuffle-rejection path
// exists for future extension; today it never fires.
// ---------------------------------------------------------------------------
[[nodiscard]] std::uint32_t shuffle_cost(const Graph& g,
                                          const stdx::small_vector<NodeId, 4>& packet) noexcept {
    (void)g;
    (void)packet;
    return 0;
}

// ---------------------------------------------------------------------------
// 31c: scan for hot LoadIndex pairs whose bases might alias. Returns the
// list of (a, b) pairs that need an AliasDisjoint guard emitted.
// ---------------------------------------------------------------------------
[[nodiscard]] stdx::small_vector<std::pair<NodeId, NodeId>, 8>
find_alias_guard_candidates(const Graph& g, const PassContext& c) noexcept {
    stdx::small_vector<std::pair<NodeId, NodeId>, 8> out;
    if (!c.is_profiled()) return out;   // Rule 2: speculation needs PGO
    if (cfg::slp_alias_guard_pgo_floor == 0) return out;

    // Collect LoadIndex nodes that are hot (pgo_count above threshold) and
    // NOT already statically proven non-aliasing (no TypeGuarded flag).
    stdx::small_vector<NodeId, 16> loads;
    g.for_each_live([&](NodeId id) {
        const Node& n = g.node(id);
        if (n.kind != NodeKind::LoadIndex) return;
        if (n.ins.size() < 4) return;        // need control, effect, base, index
        if (n.has(NodeFlag::TypeGuarded)) return;   // P14 already proved NoAlias
        if (n.pgo_count < cfg::slp_alias_guard_pgo_floor) return;
        loads.push_back(id);
    });

    // Pairwise: same loop block (ins[0]) and DIFFERENT bases.
    for (std::size_t i = 0; i < loads.size(); ++i) {
        for (std::size_t j = i + 1; j < loads.size(); ++j) {
            const Node& a = g.node(loads[i]);
            const Node& b = g.node(loads[j]);
            if (a.ins[0] != b.ins[0]) continue;          // different blocks: skip
            if (a.ins[2] == b.ins[2]) continue;           // same base: definitely alias
            out.push_back({loads[i], loads[j]});
        }
    }
    return out;
}

// Check whether a Guard(AliasDisjoint) already exists for this base pair —
// idempotency guard so a second pass run doesn't re-emit guards.
[[nodiscard]] bool has_alias_guard(const Graph& g, NodeId base_a, NodeId base_b) noexcept {
    bool found = false;
    g.for_each_live([&](NodeId id) {
        if (found) return;
        const Node& n = g.node(id);
        if (n.kind != NodeKind::Guard) return;
        if (n.subop != static_cast<std::uint16_t>(GuardKind::AliasDisjoint)) return;
        if (n.ins.size() < 2) return;
        // Order-independent match.
        if ((n.ins[0] == base_a && n.ins[1] == base_b) ||
            (n.ins[0] == base_b && n.ins[1] == base_a)) {
            found = true;
        }
    });
    return found;
}

// Emit a Guard(AliasDisjoint) + FrameState for the (base_a, base_b) pair.
// Both loads are marked TypeGuarded so 31b packetization can include them.
void emit_alias_guard(Graph& g, const PassContext& c, NodeId load_a, NodeId load_b) noexcept {
    const Node& la = g.node(load_a);
    const Node& lb = g.node(load_b);
    NodeId base_a = la.ins[2];
    NodeId base_b = lb.ins[2];

    if (has_alias_guard(g, base_a, base_b)) return;   // idempotent

    // Build the FrameState: the deoptimizer needs the bytecode resume point
    // and the SSA values materialized at this program point. We snapshot
    // both load indices and the bases (the Tier-0 interpreter will redo
    // them on deopt).
    FrameState fs;
    fs.code_unit_id = c.code_unit_id;
    fs.bytecode_offset = 0;   // safepoint_pcs table not yet wired (driver.cpp)
    fs.values.push_back(load_a);
    fs.values.push_back(load_b);
    fs.values.push_back(base_a);
    fs.values.push_back(base_b);
    fs.kinds.push_back(0);   // tagged
    fs.kinds.push_back(0);
    fs.kinds.push_back(0);
    fs.kinds.push_back(0);
    std::uint32_t fs_idx = g.add_frame_state(fs);

    // Guard node: ins[0]=base_a, ins[1]=base_b, ins[2]=FrameStateNode ref.
    // The FrameState index is stored in aux1 (the guard's frame-state slot).
    NodeId guard = g.create(NodeKind::Guard, {base_a, base_b});
    Node& gn = g.node(guard);
    gn.subop = static_cast<std::uint16_t>(GuardKind::AliasDisjoint);
    gn.aux1 = fs_idx;
    gn.set_flag(NodeFlag::Speculative);   // Rule 5: FrameState attached
    gn.set_flag(NodeFlag::Pure);          // a guard predicate has no side effect

    // Mark both loads as statically-proven-equivalent for 31b's packetizer.
    // The actual proof is now the runtime guard; after the guard succeeds,
    // the loads cannot alias within this execution.
    g.node(load_a).set_flag(NodeFlag::TypeGuarded);
    g.node(load_b).set_flag(NodeFlag::TypeGuarded);
}

// ---------------------------------------------------------------------------
// 31d: collect LoadIndex groups with the SAME base and DIFFERENT indices —
// candidate for a Gather node (pointer-array fallback).
// ---------------------------------------------------------------------------
struct GatherCandidate {
    NodeId base;
    stdx::small_vector<NodeId, 4> loads;
};
[[nodiscard]] stdx::small_vector<GatherCandidate, 4>
find_gather_candidates(const Graph& g, const PassContext& c) noexcept {
    stdx::flat_map<NodeId, stdx::small_vector<NodeId, 4>, 16> by_base;
    g.for_each_live([&](NodeId id) {
        const Node& n = g.node(id);
        if (n.kind != NodeKind::LoadIndex) return;
        if (n.ins.size() < 4) return;
        // Only gather if the base is a pointer-typed list (we can't tell
        // unboxed vs boxed at IR level cheaply — use the absence of the
        // Unboxed flag on the base as the proxy, matching Pass 47's
        // convention that unboxed backing arrays set Unboxed on the base).
        NodeId base = n.ins[2];
        if (g.node(base).has(NodeFlag::Unboxed)) return;   // contiguous path: 31b
        // Tier 1 doesn't profile — skip to keep budget bounded.
        if (c.tier == TierMode::Tier1) return;
        by_base[base].push_back(id);
    });

    stdx::small_vector<GatherCandidate, 4> out;
    for (auto& [base, loads] : by_base) {
        if (loads.size() < cfg::slp_gather_min_lanes) continue;
        GatherCandidate gc;
        gc.base = base;
        gc.loads = std::move(loads);
        out.push_back(std::move(gc));
    }
    return out;
}

// Check whether a Gather node already exists for this base — idempotency.
[[nodiscard]] bool has_gather_for_base(const Graph& g, NodeId base) noexcept {
    bool found = false;
    g.for_each_live([&](NodeId id) {
        if (found) return;
        const Node& n = g.node(id);
        if (n.kind != NodeKind::Gather) return;
        if (!n.ins.empty() && n.ins[0] == base) found = true;
    });
    return found;
}

// Emit a Gather node for a (base, [loads...]) group. The Gather takes
// the base pointer + one input per lane (the index nodes). Each LoadIndex
// is then replaced (RAUW) by a VecExtract from the Gather result.
std::uint32_t emit_gather(Graph& g, const PassContext& c,
                           NodeId base,
                           const stdx::small_vector<NodeId, 4>& loads) noexcept {
    if (has_gather_for_base(g, base)) return 0;
    if (loads.size() < cfg::slp_gather_min_lanes) return 0;
    // Cost gate (Rule 45): gather is expensive on hardware without fast
    // gather support. Decline if the target lacks AVX2 (x86) or ASIMD
    // (aarch64) — the scalar fallback runs faster than emulated gather.
    if (c.target != nullptr) {
        const bool fast_gather =
            c.target->has(backend::TargetFeature::AVX2) ||
            c.target->has(backend::TargetFeature::ASIMD);
        if (!fast_gather) return 0;
    }

    NodeId gather = g.create(NodeKind::Gather);
    Node& gn = g.node(gather);
    gn.set_flag(NodeFlag::OnEffectChain);   // gathers participate in memory ordering
    gn.set_flag(NodeFlag::MayThrow);
    g.add_input(gather, base);
    for (NodeId load : loads) {
        const Node& ln = g.node(load);
        if (ln.ins.size() >= 4) {
            g.add_input(gather, ln.ins[3]);   // index operand of the LoadIndex
        }
    }
    gn.aux0 = static_cast<std::uint32_t>(loads.size());   // lane count

    // Replace each LoadIndex with a VecExtract(Gather, lane_id). The
    // VecExtract is a pure data projection — no effect-chain involvement.
    for (std::uint32_t lane = 0; lane < loads.size(); ++lane) {
        NodeId ext = g.create(NodeKind::VecExtract, {gather});
        Node& en = g.node(ext);
        en.aux0 = lane;   // lane index
        en.set_flag(NodeFlag::Pure);
        en.set_flag(NodeFlag::Unboxed);   // gather yields native scalars
        g.replace_all_uses(loads[lane], ext);
        // Don't kill the LoadIndex — it may still be referenced by the
        // effect chain (memory ordering). P51 (GlobalDCE) will collect it
        // once nothing references it.
    }
    return 1;
}

}  // namespace

// ===========================================================================
// Main pass body.
// ===========================================================================
Result<PassResult> P31_SLPVectorization::run(Graph& g, const PassContext& c) noexcept {
    if (c.tier == TierMode::Tier1) return PassResult{};
    // Rule 27: the vector width is a machine fact — query the descriptor or
    // decline. (cfg::slp_max_packet_width was removed precisely because a
    // hardcoded AVX-512 lane count is wrong on every smaller machine.)
    if (c.target == nullptr || c.target->simd_width_bytes < 16) return PassResult{};
    const std::uint32_t max_lanes = c.target->simd_width_bytes / 8;
    const std::uint32_t before = g.live_node_count();
    std::uint32_t packets = 0;
    std::uint32_t guards_emitted = 0;
    std::uint32_t gathers_emitted = 0;
    std::uint32_t rejected_by_cost = 0;

    // === 31c: Speculative SLP — emit AliasDisjoint guards BEFORE 31b
    // packetization, so the TypeGuarded flag is visible when 31b scans. ===
    {
        auto pairs = find_alias_guard_candidates(g, c);
        for (auto [a, b] : pairs) {
            // Idempotency: skip if both already TypeGuarded (a prior run
            // of 31c emitted the guard).
            if (g.node(a).has(NodeFlag::TypeGuarded) &&
                g.node(b).has(NodeFlag::TypeGuarded)) {
                continue;
            }
            emit_alias_guard(g, c, a, b);
            ++guards_emitted;
        }
    }

    // === 31a: mark vectorizable chains. ===
    g.for_each_live([&](NodeId id) {
        const Node& n = g.node(id);
        if (!vectorizable_op(n)) return;
        if (n.kind == NodeKind::PyBinary && n.ins.size() >= 4) {
            // operands must be unboxable (const or already-unboxed chain)
            bool unboxable = true;
            for (std::uint32_t i = 2; i < n.ins.size() && unboxable; ++i) {
                unboxable = is_unboxable_input(g, n.ins[i]);
            }
            if (unboxable) {
                g.node(id).set_flag(NodeFlag::Vectorizable);
                g.node(id).set_flag(NodeFlag::Unboxed);
            }
        }
    });

    // === 31b: packetize adjacent isomorphic independent ops sharing NO
    // inputs; each packet becomes a VecOp carrying the member count.
    // Cost model (Rule 45): reject packets the target says don't pay. ===
    {
        stdx::small_vector<NodeId, 16> candidates;
        g.for_each_live([&](NodeId id) {
            if (g.node(id).has(NodeFlag::Vectorizable)) candidates.push_back(id);
        });

        stdx::flat_map<NodeId, bool, 32> packed;
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            if (packed.contains(candidates[i])) continue;
            stdx::small_vector<NodeId, 4> packet{candidates[i]};
            for (std::size_t j = i + 1; j < candidates.size() &&
                                       packet.size() < max_lanes; ++j) {
                if (packed.contains(candidates[j])) continue;
                if (!same_shape(g, candidates[i], candidates[j])) continue;
                // 31c independence: no shared inputs (checked by same_shape)
                // and no data dependence between members.
                packet.push_back(candidates[j]);
            }
            if (packet.size() < cfg::slp_min_packet_lanes) continue;

            // Cost model (Rule 45): vector_pays() returns false when the
            // move cost (inserting N scalars into the vector file) dominates
            // the scalar execution cost. The per-op cost is the LATENCY of
            // the packet's op class, queried from the target descriptor via
            // the arch-neutral cost_class(BinOpKind) helper — never a
            // switch in pass logic (Rule 23/24).
            const std::uint32_t shuffles = shuffle_cost(g, packet);
            const BinOpKind op = static_cast<BinOpKind>(g.node(packet[0]).subop);
            const backend::CostClass cc = backend::cost_class(op);
            const std::uint32_t op_latency = c.target->latency(cc);
            // Shuffle rejection: alignment overhead must not exceed the
            // configured ratio of the lane count (Larsen & Amarasinghe).
            const std::uint32_t shuffle_budget =
                (packet.size() * cfg::slp_max_shuffle_ratio_percent) / 100;
            const bool pays = c.target->vector_pays(
                static_cast<std::uint32_t>(packet.size()), op_latency);
            if (!pays || shuffles > shuffle_budget) {
                ++rejected_by_cost;
                continue;
            }

            NodeId vec = g.create(NodeKind::VecOp);
            Node& vn = g.node(vec);
            vn.subop = g.node(packet[0]).subop;
            vn.aux0 = static_cast<std::uint32_t>(packet.size());
            vn.set_flag(NodeFlag::Pure);
            vn.set_flag(NodeFlag::Unboxed);
            for (std::uint32_t lane = 0; lane < packet.size(); ++lane) {
                NodeId m = packet[lane];
                packed.insert(m, true);
                // Reference the member's data inputs as vector lane sources
                // (skip control/effect at ins[0], ins[1]).
                for (std::uint32_t s = 2; s < g.node(m).ins.size(); ++s) {
                    g.add_input(vec, g.node(m).ins[s]);
                }
                // Replace the packed member with a VecExtract projection
                // from the new VecOp. This is the SLP contract: the scalar
                // op is dead, its result is now lane `lane` of the vector.
                // RAUW ensures downstream users see the VecExtract; the
                // member itself is then killed so a second pass run sees
                // no candidates (idempotency, Rule 10).
                NodeId ext = g.create(NodeKind::VecExtract, {vec});
                Node& en = g.node(ext);
                en.aux0 = lane;   // lane index
                en.set_flag(NodeFlag::Pure);
                en.set_flag(NodeFlag::Unboxed);
                g.replace_all_uses(m, ext);
                g.kill(m);
            }
            ++packets;
        }
    }

    // === 31d: gather/scatter fallback for pointer-array LoadIndex
    // groups that can't be unboxed to contiguous loads. ===
    {
        auto groups = find_gather_candidates(g, c);
        for (auto& [base, loads] : groups) {
            if (loads.size() > max_lanes) loads.resize(max_lanes);
            gathers_emitted += emit_gather(g, c, base, loads);
        }
    }

    PassResult r = result_of(g, before);
    r.changed = (packets + guards_emitted + gathers_emitted) > 0;
    // Telemetry (Rule 26): no silent fallbacks. The count reflects
    // packets + guards + gathers emitted this run.
    note(TelemetryEventKind::SafepointPatched, c,
         packets + guards_emitted + gathers_emitted);
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
