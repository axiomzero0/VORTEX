# VORTEX Compiler Laws & Architecture Specification

**Status:** Stable
**Owner:** VORTEX Systems Dev Team
**Last Updated:** 2026-08-25
**Target Language:** Python 3.16 (Dynamic semantics, adaptive bytecode, optional static typing)
**Related Sections:** `ir_spec.md`, `effect_system.md`, `abi.md`, `python316_semantics.md`

This document is the authoritative, uncompressed transcription of the laws that
govern the VORTEX compiler. **Every commit to the `compiler/`, `runtime/`,
`tools/`, and `tests/` trees must comply. CI verifies them. There are no
exceptions.**

---

## Part I: Architectural Overview

The VORTEX execution model for Python 3.16 consists of four distinct,
seamlessly integrated execution tiers. Transitions between tiers are governed
by profile data and static guarantees, never by arbitrary timeouts.

1. **Tier 0: Direct-Threaded Register Interpreter**
   The baseline execution engine. Uses computed gotos and a register-based
   bytecode representation (not stack-based) to minimize dispatch overhead.
   Fast startup, minimal memory footprint, and the ultimate fallback for all
   code.
2. **Tier 1: Baseline JIT**
   Triggered by low-level heat (e.g., loop backedges). Compiles in linear
   time. Performs basic type specialization, copy propagation, and simple
   inlining. Emits lightweight guards. Compilation time is strictly bounded
   to prevent mutator stalls.
3. **Tier 2: Optimizing JIT**
   Triggered by sustained heat and rich PGO data. Employs the full VORTEX
   optimization pipeline (Passes 1–51). Performs aggressive speculative
   optimizations (PEA, SLP vectorization, speculative effect reordering).
   Requires full `FrameState` and deoptimization infrastructure.
4. **Tier 3: AOT / Static JIT**
   Applied to code with provable static guarantees (e.g., `typing.Final`,
   `__static__` blocks, fully annotated modules with no dynamic introspection).
   Bypasses speculation and deoptimization overhead entirely. Maximum
   optimization, zero guard checks, direct native code emission.

---

## Part II: The Unified Pipeline & Speculation Laws

**Rule 1 — One Pipeline, Multiple Inputs**
The sequence of optimization passes (1–51) is **identical** for Tier 1, Tier 2,
and Tier 3.
- **Tier 1 Input:** Static IR + Minimal Heuristics (Budget-constrained).
- **Tier 2 Input:** Static IR + **PGO Data** (Aggressive, guard-emitting).
- **Tier 3 Input:** Static IR + **Static Proofs** (No guards, maximum optimization).
There is no "JIT-only" pass list. If a pass exists, it must handle all modes via
a unified interface.

**Rule 2 — PGO is a Force Multiplier, Not New Logic**
PGO data does not change *what* the compiler does; it changes *how aggressively*
it does it. Example: Pass 49 (Speculative Effect Reordering) runs in all tiers.
In Tier 3, it requires static CFL-Reachability proof. In Tier 2, it accepts
PGO-proven probability >99% and inserts a guard. In Tier 1, it is skipped due
to budget.

**Rule 3 — Every PGO-Driven Decision Requires a Guard**
If a pass makes a decision based on PGO (e.g., "Pointers A and B never alias",
"This Python dict lookup is monomorphic"), it **must** emit a hardware guard.
Guard success executes the optimized path; guard failure triggers
Deoptimization.

**Rule 4 — Deoptimization Must Reconstruct Tier 0 State**
When a JIT guard fails, the runtime must deoptimize to the **exact same state**
the Tier 0 Direct-Threaded Register Interpreter would have been in at that
instruction pointer. This includes restoring register values, re-materializing
stack frames, and rolling back speculatively reordered memory writes
(`Altered` nodes) to maintain sequential consistency.

**Rule 5 — FrameState is Mandatory for All Guards**
Every node that introduces a speculative assumption must have a `FrameState`
attachment. This snapshot allows the deoptimizer to rebuild the Python 3.16
execution world (including `f_locals`, `f_globals`, and reference counts) if
the speculation fails.

---

## Part III: Compilation Pipeline & Memory Laws

**Rule 6 — NO EXCEPTIONS ON THE HOT PATH**
The JIT compiler, AOT compiler, and Runtime Deoptimization engine **MUST** be
compiled with `-fno-exceptions`. Zero `throw` statements are allowed in any
code path executed during compilation or runtime specialization. All fallible
operations **MUST** use `std::expected<T, Diagnostic>` or `Result<T, Error>`.
If a JIT compilation fails, it returns an `Error` variant, causing the system
to silently fall back to Tier 0 or Tier 1. No stack unwinding. No catch
blocks. No overhead.

**Rule 7 — Zero-Allocation Hot Path**
Both AOT and JIT compilers must use `std::pmr::monotonic_buffer_resource` for
IR allocation. Bulk-free after compilation. No `malloc`/`free` in the compiler
hot path.

**Rule 8 — No RTTI**
Both pipelines are compiled with `-fno-rtti`. Use `enum class NodeKind` for
type switching. RTTI is forbidden in the IR and backend to ensure maximum
devirtualization and cache locality.

**Rule 9 — No `std::shared_ptr` / `std::function` in Hot IR Code**
They allocate and incur atomic overhead. Use raw pointers + stable `NodeId`s
inside passes.

**Rule 10 — Every Pass Must Be Idempotent and Monotonic**
Running the same pass twice must produce the identical IR. A pass either
reduces node count or moves the IR closer to a normal form. If a pass can grow
the IR (e.g., Loop Unrolling, SLP), it must run inside a guarded fixpoint with
a strict budget.

**Rule 11 — Mutator Threads Never Block on JIT**
If a function becomes "hot" and triggers a JIT compilation, the mutator thread
continues executing the current tier. The JIT runs asynchronously on a
background compiler thread. Once ready, a safe-point patch swaps the function
pointer.

**Rule 12 — Thread-Local Allocation for Mutators**
Mutator threads use thread-local bump pointers (lexical regions) for their own
runtime allocations (e.g., temporary Python objects). Global synchronization
happens only at explicit yield points.

**Rule 13 — Compiler Threads Never Block on Mutator State**
The compiler works on a frozen snapshot of the IR and PGO data. Mutator updates
after the snapshot are picked up by the next compilation.

**Rule 14 — Epoch-Based Reclamation**
Old JIT code and IR nodes are reclaimed using epoch-based garbage collection.
When the optimizer replaces a `Node`, the old node is tagged with an epoch.
Once all threads advance past that epoch, the memory is bulk-freed. This
avoids both locks and use-after-free.

---

## Part IV: The Numbered Rules (15–80)

### Data Structures & IR Design

**Rule 15 — Index-Based Graph (No Raw Pointers in the IR)**
Never use raw pointers (`Node*`) for edges in the Sea of Nodes. All node
references must use a 32-bit integer index (`using NodeId = uint32_t;`). This
cuts memory footprint in half, doubles L1/L2 cache capacity, and makes the IR
trivially serializable and immune to pointer invalidation during arena
reallocation.

**Rule 16 — Interned Symbols (No Strings in the Hot Path)**
Never pass, compare, or store `std::string` or `std::string_view` in the IR or
passes. All identifiers, variable names, and Python attribute names must be
interned into a global `SymbolTable` at the frontend. The IR must only use a
`SymbolId` (`uint32_t`).

**Rule 17 — Cache-Friendly Hash Maps (Ban `std::unordered_map`)**
`std::unordered_map` and `std::map` are forbidden in the compiler hot path.
For Global Value Numbering (GVN), Hash-Consing, and any pass requiring a hash
table, you must use a cache-friendly, open-addressing hash map (e.g.,
SwissTable / `flat_hash_map`).

**Rule 18 — Sparse Sets and BitVectors for Pass Data**
Ban `std::set`, `std::unordered_set`, and `std::vector<bool>` for dataflow
analysis. Passes tracking sets of `NodeId`s (liveness, dominators, visited)
must use **Sparse Sets** (for small, dense sets) or **BitVectors** (for large,
sparse sets).

**Rule 19 — Small Buffer Optimization (SBO) for Variable-Length Data**
Ban `std::vector` for data that usually has 1 to 4 elements. For Use-Def
chains, instruction operands, and basic block predecessors/successors, use a
`SmallVector<T, N>` (where N is typically 2, 3, or 4).

**Rule 20 — Structure of Arrays (SoA) for Bulk Pass Processing**
When a pass needs to process a specific field of millions of nodes, do not
iterate over the `Node` structs (AoS). Extract that attribute into a
contiguous `std::pmr::vector` (SoA layout) to allow perfect CPU prefetching
and SIMD vectorization on the compiler's own passes.

### Performance & Hardware

**Rule 21 — Exploit C++26 Compiler Hints**
Use `[[likely]]` and `[[unlikely]]` on all PGO-driven branches and
deoptimization traps. Use C++23/26 `[[assume(condition)]]` to tell the
compiler about invariants (e.g.,
`[[assume(node_id < graph.size())]]`) to eliminate bounds checks in internal
compiler data structures.

**Rule 22 — Zero-Cost Error Propagation**
Do not use verbose `if (err)` chains that ruin branch prediction. Use
`std::expected<T, Error>` and monadic operations (`and_then`, `transform`) or
a custom `TRY()` macro that compiles down to a single branch, keeping the hot
path instruction cache pristine.

**Rule 23 — No Hard-Coded Constants in Optimization Logic**
Magic numbers are forbidden. Every threshold, budget, limit, and heuristic
constant used in any optimization pass MUST be defined as a named, documented
`constexpr` constant or configuration parameter. CI static analysis fails if
numeric literals > 2 appear in pass logic without a named constant reference.

**Rule 24 — No Target-Specific Hacks in Generic Passes**
Mid-level and research passes (GVN, LICM, SLP, CFL-Alias) MUST NOT contain
`#ifdef X86` or target-specific conditionals. All target knowledge must be
abstracted behind the `Target` interface and queried via cost models or
capability flags.

**Rule 25 — No Heuristics Without Empirical Validation**
Every heuristic MUST be backed by benchmark data showing measurable
improvement, a mechanism to override/tune it, and documentation explaining why
the value was chosen.

**Rule 26 — No Silent Fallbacks Without Telemetry**
When the JIT falls back to a lower tier, when a speculative guard fails, or
when regalloc spills excessively, the event MUST be recorded in
telemetry/profile data. Silent fallbacks hide performance problems.

**Rule 27 — No Assumption of Stable Hardware**
No pass may assume fixed cache line sizes, SIMD widths, or memory latency
ratios. All hardware parameters MUST be queried at runtime (for JIT) or
build-time (for AOT) via the `Target` interface.

**Rule 28 — No Optimization Without Measurable Win**
Every optimization pass added to the pipeline MUST demonstrate a ≥1% geometric
mean improvement across the benchmark suite, OR enable a correctness/safety
property that cannot be achieved otherwise. Underperforming passes are removed.

### Correctness & Python 3.16 Semantics

**Rule 29 — No FFI Optimization Without ABI Proof**
FFI optimizations (e.g., calling C extensions or Python C-API) must prove
calling convention correctness, stack alignment, register clobbering, and
memory ownership transfer (especially Python reference counting).

**Rule 30 — No Vectorization Without Dependence Proof**
Vectorization (SLP or loop) must prove no aliasing (or use versioned checks),
bounds safety, alignment, and correct scalar fallback.

**Rule 31 — No Persistent State Without Versioning**
Profile caches, code caches, and AOT artifacts must be versioned. A change in
the IR format, Python 3.16 bytecode version, or pass order invalidates the
cache.

**Rule 32 — All Orthogonal Boolean State Must Be Bitmasked**
Any set of independent boolean properties on a hot-path data structure (e.g.,
`NodeFlags`, `EffectTags`) must be represented as a bitmask with type-safe
`Flags<E>` wrappers. Raw integers are forbidden for flag-like state.

**Rule 33 — No Implicit Conversions or Coercions in IR**
The VORTEX IR MUST NOT perform implicit type conversions, integer promotions,
or pointer coercions. All conversions must be explicit nodes (`IntToPtr`,
`SExt`, `ZExt`, `BitCast`, `PyObjToNative`). The frontend lowering pass
inserts these explicitly.

### Testing & Verification

**Rule 34 — Five Regression Tests Per Bug Fix**
Every bug fix must include at least **5 regression tests**:
1. Minimal reproducer.
2. Variant trigger (different code pattern, same root cause).
3. Boundary/negative (ensures the fix doesn't over-correct).
4. Integration/contextual (bug in realistic surrounding code).
5. Deopt/State Reconstruction (verifies that if the JIT speculates wrongly,
   the deopt to Tier 0 produces the exact same state).

*Enforcement:* CI fails if a PR labeled `bugfix` has fewer than 5 new test
cases.

**Rule 35 — Golden Tests for Every Pass**
Every optimization pass must have ≥10 golden IR tests. Checked-in
`.in.vortex` / `.expected.ir` file pairs. Tests must run in both Static Mode
(AOT/Tier 3) and Profile Mode (Tier 2).

**Rule 36 — Differential Testing is Mandatory in CI**
`Tier 0 Interpreter` ↔ `Tier 2 JIT Apex` ↔ `Tier 3 AOT` comparisons run on
every PR. Divergence blocks merge. Assert byte-for-byte identical results and
memory layouts.

**Rule 37 — Deopt Paths Must Be Fuzzed Weekly**
Scheduled CI job. Results triaged within 24 hours. Untriaged deopt fuzz
failures block releases.

**Rule 38 — Replay Logs Retained for All CI Failures**
Failed test runs automatically save full compile replay artifacts (IR snapshot
+ PGO profile + compiler options + RNG seed). Debugging starts from replay,
not reproduction.

**Rule 39 — Performance Regressions Require Explicit Waiver**
If a benchmark regresses >5%, the PR must include root cause analysis,
justification, a tracking issue, and approval. No silent performance
degradation.

**Rule 40 — Graph Verifier Runs in Debug Builds After *Every* Pass**
The verifier checks: no dangling `NodeId`s, Effect chain continuity, control
dominance, use-def consistency, and `FrameState` attached to every PGO-driven
guard.

**Rule 41 — Test Names Encode the Bug/Feature They Cover**
Bad: `test_pea_3`. Good:
`pea_non_escaping_region_object_with_deopt_materializes_correctly`.
Searchable, self-documenting.

### Speculation & Guarding

**Rule 42 — No Assumption Without Invalidation**
Every PGO-driven assumption must have a registry entry (Watchdog), an
invalidation path (Trip), and a fallback to static proof or lower-tier
execution.

**Rule 43 — No Specialization Without Fallback**
Every specialized clone (e.g., a bounds-check-eliminated Python list loop)
must have a generic fallback, a deopt path, and a budget limit.

**Rule 44 — No Profile Data Without Confidence**
Profile data must include sample count, stability, age, decay, variance, and
deopt correlation (Meter). Low-confidence data must not trigger aggressive
speculation.

**Rule 45 — No Aggressive Pass Without a Cost Model**
Inlining, cloning, unrolling, SLP vectorization, and PEA materialization must
all use a strict cost model (Regulator) based on target hardware latencies and
Python 3.16 object overhead.

---

## Part V: Code Quality & Developer Velocity Laws

*These rules are designed to be maximally strict on correctness and
maintainability, while actively protecting and enhancing developer speed.*

**Rule 46 — Local Pre-Commit Checks Must Complete in < 2 Seconds**
Strictness must not impede velocity. The local `pre-commit` hook (formatting,
basic linting, copyright headers) must execute in under 2 seconds. Heavy
checks (full test suites, differential testing) are deferred to asynchronous
CI.

**Rule 47 — Actionable Compiler Diagnostics**
The compiler must never output opaque errors (e.g., "Error: something went
wrong"). All `Diagnostic` objects must include: the exact source location
(file, line, column), a clear human-readable message, the expected vs. actual
state, and a suggested fix. This saves developers hours of debugging.

**Rule 48 — `[[nodiscard]]` on All Result Types**
All functions returning `std::expected`, `Result`, or `Error` must be marked
`[[nodiscard]]`. Ignoring an error is a compilation failure. This forces
developers to handle edge cases explicitly without requiring verbose,
performance-killing `if` chains (use `TRY()` macros instead).

**Rule 49 — No `#define` Macros for Logic**
C-style macros for control flow or logic are forbidden. Use `constexpr`
functions, `inline` functions, or templates. Macros are exempt only for header
guards and trivial token pasting. This ensures the debugger can step through
the code and the compiler can inline/optimize it properly.

**Rule 50 — Fast Incremental Builds via Modular CMake**
The build system must be structured to allow sub-second incremental builds for
single-file changes. Heavy dependencies (e.g., testing frameworks, LLVM
backend if used) must be isolated. Developers must not wait minutes to test a
single IR pass modification.

**Rule 51 — Automated Refactoring Tools Over Manual Edits**
When a structural change is required (e.g., renaming a `NodeKind`, adding a
field to `FrameState`), a scripted refactoring tool (e.g., `clang-tidy` fixits
or custom Python scripts) must be provided and run as part of the PR. Manual,
error-prone find-and-replace across 100 files is forbidden.

**Rule 52 — Self-Contained, Reproducible Test Cases**
Every test must be fully self-contained. It must not rely on external network
calls, specific local directory structures, or non-deterministic system state.
Tests must run identically on a developer's ChromeOS machine, a Linux CI
runner, or a macOS workstation.

---

## Part VI: Anti-Slop & Robustness Laws

**Rule 53 — No "Small Bug" or "Minor Edge Case" Rationalization**
The phrases "small bug," "minor edge case," "rarely happens," "only affects
cold paths," and "good enough for now" are banned. In a systems compiler,
"small" bugs cause silent data corruption or catastrophic performance cliffs.
All bugs must be triaged with explicit severity.

**Rule 54 — No Workarounds for Compiler/Runtime Bugs**
Adding code to "work around" a bug in the compiler, runtime, or standard
library is forbidden. The underlying defect MUST be fixed. Temporary
mitigations require a tracking issue, a removal deadline ≤ 2 weeks, and
explicit approval from the tech lead.

**Rule 55 — No Implicit Knowledge Transfer**
All design decisions, trade-offs, historical context, and operational
knowledge MUST be captured in persistent, searchable documentation (code
comments, ADRs, wiki, specs). Oral tradition and chat messages are not valid
knowledge stores.

**Rule 56 — No Premature Simplification**
Do not simplify, abstract, or generalize code until the full problem space is
understood and at least two concrete use cases exist. Premature simplification
creates leaky abstractions that fail under real-world Python 3.16 conditions.

**Rule 57 — No Copy-Paste Code or Structural Duplication**
If two code blocks share structure, extract a helper, template, or data-driven
approach. ABI definitions, register lists, and pass boilerplate must use
generators, `constexpr` helpers, or declarative tables.

**Rule 58 — No Silent Fallbacks or Default Returns**
Switch statements on closed enums must be exhaustive. Non-exhaustive switches
require `[[assume(false)]]` + `VORTEX_UNREACHABLE()`. Functions must not return
arbitrary default values (`return 0;`, `return nullptr;`) when input is
invalid.

**Rule 59 — No Lazy Data Structures or Algorithms**
Use the right tool, not the convenient tool. Linear search is forbidden where
O(1) lookup is feasible. String comparison is forbidden where symbol IDs
suffice.

**Rule 60 — No Untested or Unverified Code Paths**
Every branch, edge case, and error path must have explicit test coverage. "It
compiles" is not verification. Only automated, reproducible tests count.

**Rule 61 — No Performance-Agnostic Implementation**
Hot-path code must avoid allocations, exceptions, RTTI, virtual dispatch, and
cache-unfriendly patterns. Performance is a feature. Ignoring it in
implementation guarantees degradation.

**Rule 62 — No Deletion-by-Avoidance ("Too Hard" Is Not a Valid Reason)**
Deleting, disabling, commenting out, or stubbing functionality because it is
"too hard" or "too complex" is strictly forbidden. When encountering difficult
problems: Decompose, Research, Prototype, Document, and Escalate.

**Rule 63 — No Fragile Implementations**
All implementations MUST be resilient to malformed input, concurrent access,
resource exhaustion, and platform/hardware variation. Fragile patterns
(implicit ordering dependencies, global mutable state, unchecked pointer
arithmetic) are forbidden.

**Rule 64 — No Documentation Debt**
Every public API, internal helper, IR node, pass, configuration knob, and
non-obvious algorithm MUST have documentation at the point of definition
covering: Purpose, Invariants, Rationale, Edge Cases, and Cross-References.
Stale documentation is treated as a bug with the same severity as stale code.

**Rule 65 — No Easy Fixes — Only Correctness-Preserving Performance Fixes**
When fixing a bug, you must implement the fix that *simultaneously* preserves
performance and correctness. "Easy" fixes that sacrifice either property are
forbidden unless explicitly documented as temporary mitigations with tracking
issues and removal deadlines.

**Rule 66 — Slop Detection Checklist (For Code Review)**
Every PR reviewer must verify:
- [ ] No unnamed numeric constants in logic
- [ ] No duplicated code blocks or copy-paste patterns
- [ ] No silent fallbacks or unsafe default returns
- [ ] No prohibited containers (`std::unordered_map`, `std::vector` for small
      collections) in hot paths
- [ ] All invariants documented and validated
- [ ] No premature abstractions without ≥2 consumers
- [ ] No untracked workarounds or `HACK` comments
- [ ] No target-specific logic outside `backend/`
- [ ] All new code paths have test coverage
- [ ] Hot-path changes justified with profiling/benchmarks

*Failure on any item blocks merge. No exceptions. No "small slop." No "we'll
fix it later." Slop is banned.*

---

*End of VORTEX Compiler Laws & Architecture Specification.*
*Compliance is not optional. It is the foundation of VORTEX.*
