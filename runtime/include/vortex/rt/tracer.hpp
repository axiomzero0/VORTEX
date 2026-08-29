// =============================================================================
// vortex/rt/tracer.hpp — Meta-tracer for Tier 1 JIT
//
// Purpose:
//   Records sequences of Tier-0 bytecode instructions executed at hot loop
//   backedges. When a trace completes (returns to the loop header), it is
//   compiled into native machine code with guards. On subsequent executions,
//   the trace runs as native code, eliminating dispatch overhead entirely.
//
// Architecture (per the VORTEX hybrid tracing design):
//   - Tier 0: Direct-threaded register interpreter (baseline)
//   - Tier 1: Meta-tracing JIT (THIS FILE) — records interpreter execution
//     paths and compiles them to machine code with type guards
//   - Tier 2: Sea of Nodes optimizing JIT (existing 51-pass pipeline)
//
// The meta-tracer instruments the Tier-0 dispatch loop at backedges:
//   1. A backedge fires enough times (hot threshold)
//   2. Recording begins: each executed Instr is appended to a TraceBuffer
//   3. Recording ends when the backedge fires again (trace complete)
//   4. The trace is compiled to machine code with guards at type-dependent ops
//   5. On the next backedge, the compiled trace is invoked instead of
//      continuing the dispatch loop
//
// Rule 3: Every speculative decision requires a guard. The trace records
// the types seen at each op and emits guard instructions that deopt back
// to Tier-0 if the types change.
//
// Rule 5: FrameState is attached to every guard for deopt reconstruction.
// Rule 10: The tracer is idempotent — recording the same loop twice
// produces the same trace.
// Rule 88: Trace execution includes safepoint polls at the backedge.
// =============================================================================

#pragma once

#include "vortex/rt/code.hpp"
#include "vortex/stdx/flat_map.hpp"
#include "vortex/support/profiler.hpp"

#include <cstdint>
#include <cstring>

namespace vortex::rt {

inline namespace abi_v1 {

/// A single recorded instruction in a trace. Mirrors the Tier-0 Instr
/// but adds the observed type tag for guard emission.
struct TraceInstr {
    Instr instr;           // The Tier-0 bytecode instruction
    std::uint8_t tag_a;    // Observed tag of operand A (Tag::Int, Tag::Float, etc.)
    std::uint8_t tag_b;    // Observed tag of operand B
    std::uint8_t tag_dst;  // Observed tag of the result
    std::uint32_t pc;       // Tier-0 PC of this instruction (for JUMP target resolution)
};

/// A recorded trace — a linear sequence of bytecode instructions with
/// observed type information, ending at a backedge to the trace header.
struct Trace {
    CodeUnit* unit{nullptr};        // The CodeUnit this trace belongs to
    std::uint32_t header_pc{0};     // The PC of the loop header (trace start)
    std::uint32_t exit_pc{0};       // The PC after the backedge (trace end)
    stdx::small_vector<TraceInstr, 64> instrs{};  // Recorded instructions
    std::uint32_t hit_count{0};      // How many times this trace has executed
    void* native_code{nullptr};     // Compiled machine code (null = not yet compiled)
    std::size_t native_code_size{0};
    bool is_compiled{false};        // Has the trace been compiled to native code?
    bool is_recording{false};       // Are we currently recording this trace?

    // Layer 2: CorrelationId — causally links this trace to its Tier-0
    // execution context. Used by the Introspector (Rule 119) to answer:
    // "this deopt came from trace X which was compiled from loop Y."
    std::uint64_t program_hash{0};  // hash of source module (0 = unset)
};

/// The meta-tracer state. One per Vm (single-threaded for now).
/// Rule 7: No allocation on the hot path. TraceInstrs use SBO (Rule 19).
/// Rule 118: No locks — the tracer is thread-local (single-threaded).
struct MetaTracer {
    /// The trace currently being recorded, or nullptr if not recording.
    Trace* recording{nullptr};

    /// Compiled traces, keyed by (unit_id << 16 | header_pc).
    /// Bug fix 1.7.3: was a single Trace* (active), which leaked the
    /// previous trace's native_code when a second loop became hot.
    /// Now a flat_map so every hot loop keeps its trace.
    /// Rule 106: code cache has a max; eviction munmaps the old trace.
    static constexpr std::uint32_t kMaxCompiledTraces = 64;
    stdx::flat_map<std::uint64_t, Trace*, kMaxCompiledTraces> traces{};

    /// The PC where recording started (the loop header).
    std::uint32_t record_start_pc{0};

    /// The unit being recorded.
    CodeUnit* record_unit{nullptr};

    /// Hot threshold: number of backedge hits before recording starts.
    /// Rule 23: named constant, not a magic number.
    /// Rule 25: calibrated on the in-tree benchmark suite.
    static constexpr std::uint32_t kHotThreshold = 64;

    /// Max trace length — prevents unbounded recording on pathological loops.
    /// Rule 10: bounded by budget.
    static constexpr std::uint32_t kMaxTraceLen = 256;

    /// Giga Tracing: probabilistic profiler (Count-Min Sketch + EMA).
    /// Non-owning, set by the Vm at construction. May be null in tests
    /// that construct a MetaTracer without a Vm — every consult is
    /// null-checked. When non-null, on_backedge uses is_hot() as an
    /// alternative tier-promotion trigger, and compile_trace uses
    /// guard_always_passes() to elide redundant B-tag guards on traces
    /// that have >1000 successful executions.
    /// Rule 118: single-writer (the mutator thread) — no synchronization.
    support::ProbabilisticProfiler* profiler_{nullptr};

    /// Wire the probabilistic profiler. Called once by the owning Vm
    /// after both `tracer` and `profiler` members are constructed.
    void set_profiler(support::ProbabilisticProfiler* p) noexcept {
        profiler_ = p;
    }

    /// Called at every backedge in the dispatch loop. Checks if the loop
    /// is hot enough to start recording, or if a compiled trace exists.
    ///
    /// Returns true if the caller should jump to native trace code
    /// (active->native_code is non-null and ready). Returns false if
    /// the caller should continue in the interpreter (either recording
    /// or not-yet-hot).
    [[nodiscard]] bool on_backedge(CodeUnit* unit, std::uint32_t pc,
                                    std::uint32_t target_pc) noexcept;

    /// Record a single instruction during trace recording.
    /// Called from the dispatch loop after each instruction executes.
    /// `pc` is the Tier-0 bytecode PC of the instruction — needed for
    /// resolving JUMP targets to positions within the compiled trace
    /// (otherwise nested-loop backedges patch to the wrong header).
    void record_instr(const Instr& instr, std::uint32_t pc, std::uint8_t tag_a,
                      std::uint8_t tag_b, std::uint8_t tag_dst) noexcept;

    /// Finish recording and compile the trace.
    /// Called when the backedge fires again during recording.
    void finish_recording() noexcept;

    /// Check if we're currently recording a trace.
    [[nodiscard]] bool is_recording() const noexcept { return recording != nullptr; }
};

}  // namespace abi_v1
}  // namespace vortex::rt
