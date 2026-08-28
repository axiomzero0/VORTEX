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
The IR, pass interfaces, verifier constraints, and correctness constraints are
**unified** across Tier 1, Tier 2, and Tier 3. Tiers differ only by budgets,
enabled speculation policies, available proofs, and telemetry requirements.
- **Tier 1 Input:** Static IR + Minimal Heuristics (Budget-constrained).
- **Tier 2 Input:** Static IR + **PGO Data** (Aggressive, guard-emitting).
- **Tier 3 Input:** Static IR + **Static Proofs** (No guards, maximum optimization).
There is no "JIT-only" pass list. Forcing the exact same pass sequence can
waste compile time in Tier 1 or prevent legitimate tier-specific lowering.
If a pass exists, it must handle all modes via a unified interface, but the
pipeline may apply tier-specific filters.

**Rule 2 — PGO is a Force Multiplier, Not New Logic**
PGO data does not change *what* the compiler does; it changes *how aggressively*
it does it. Example: Pass 49 (Speculative Effect Reordering) runs in all tiers.
In Tier 3, it requires static CFL-Reachability proof. In Tier 2, it accepts
PGO-proven probability >99% and inserts a guard. In Tier 1, it is skipped due
to budget.

**Rule 3 — Every PGO-Driven Decision Requires a Guard**
If a pass makes a decision based on PGO (e.g., "Pointers A and B never alias",
"This Python dict lookup is monomorphic"), it **must** emit a validated guard
mechanism: runtime check, shape/version guard, patchpoint, trap, dependency
invalidation, or hardware check. Not all speculation is best expressed as a
hardware branch. Guard success executes the optimized path; guard failure
triggers Deoptimization.

**Rule 4 — Deoptimization Must Reconstruct Tier 0 State**
When a JIT guard fails, the runtime must deoptimize to the **exact same state**
the Tier 0 Direct-Threaded Register Interpreter would have been in at that
instruction pointer. This includes restoring register values, re-materializing
stack frames, and rolling back speculatively reordered memory writes
(`Altered` nodes) to maintain sequential consistency.

Speculative execution must not perform irreversible Python-visible side effects
before the last guard protecting that speculation. If side effects are moved,
they must be either proven non-observable, deferred until committed, or
supported by a verified compensation mechanism. "Rolling back memory writes"
is dangerous if Python-visible effects already happened.

**Rule 5 — FrameState is Mandatory for All Guards**
Every node that introduces a speculative assumption must have a `FrameState`
attachment. This snapshot allows the deoptimizer to rebuild the Python 3.16
execution world (including `f_locals`, `f_globals`, and reference counts) if
the speculation fails.

---

## Part III: Compilation Pipeline & Memory Laws

**Rule 6 — NO EXCEPTIONS ON THE HOT PATH**
Native C++ exceptions are forbidden on compiler/runtime hot paths. The JIT
compiler, AOT compiler, and Runtime Deoptimization engine **MUST** be compiled
with `-fno-exceptions`. Zero `throw` statements are allowed in any code path
executed during compilation or runtime specialization. All fallible operations
**MUST** use `std::expected<T, Diagnostic>` or `Result<T, Error>`. Python
exceptions remain first-class runtime values and must be modeled explicitly —
do not confuse Python exceptions with C++ exceptions. If a JIT compilation
fails, it returns an `Error` variant, causing the system to silently fall back
to Tier 0 or Tier 1. No stack unwinding. No catch blocks. No overhead.

**Rule 7 — Zero-Allocation Hot Path**
Both AOT and JIT compilers must use `std::pmr::monotonic_buffer_resource` for
IR allocation. Bulk-free after compilation. No `malloc`/`free` in the compiler
hot path.

"Hot path" means: compiler pass execution, guard execution, inline-cache fast
paths, allocation fast paths, and deopt entry trampolines. Deopt materialization
may allocate only through a controlled runtime path with explicit budgets —
deopt may legitimately need to materialize objects.

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
pointer. Function pointer publication must be atomic and safe against
concurrent execution. Old code must remain valid until quiescence — a
safe-point patch is not enough if publication is torn or old code is freed
too early.

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
avoids both locks and use-after-free. Generated code must not be reclaimed
until no thread can be executing it or depend on its deopt metadata. IR nodes
and machine code need different reclamation guarantees.

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
every PR. Tier outputs must be observationally equivalent according to the
Python 3.16 semantic oracle. Any permitted differences must be explicitly
listed, versioned, and tested. Memory layout is tested separately only where
it is a supported guarantee (e.g., not with moving GC, ASLR, or hash
randomization). Divergence blocks merge.

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
- [ ] Every new guard has a `FrameState` attachment (Rule 5)
- [ ] Every speculative node carries metadata (speculation kind, PGO source,
      confidence, guard plan, deopt target, invalidation dependency)
- [ ] Every GC reference in generated code across a safepoint has a stack map
- [ ] Every deopt point is reachable and has complete deopt metadata
- [ ] No raw object pointers held across safepoints without GC map entries
- [ ] No `getenv()` or mutex-locking calls in dispatch loops or hot paths
- [ ] No atomic RMW in per-instruction or per-backedge hot paths unless
      explicitly justified (Rule 118)
- [ ] Every memory store of a reference executes the correct write barrier
      if the GC requires one
- [ ] W^X is maintained: no page is simultaneously writable and executable
- [ ] Code publication is atomic with release semantics; consumers acquire

*Failure on any item blocks merge. No exceptions. No "small slop." No "we'll
fix it later." Slop is banned.*

---

## Part VII: Python 3.16 Semantic Fidelity Laws

**Rule 67 — CPython 3.16 Is the Semantic Oracle**
All executable Python behavior must match the supported CPython 3.16 reference
semantics unless a divergence is explicitly documented, justified, versioned,
and approved. Observable behavior includes at least: program output, exceptions
and tracebacks, side effects, object mutation, weakref behavior, finalization
behavior where specified, frame introspection, debugging and monitoring events,
`id()`/`is` semantics where supported, module and import semantics, and
documented built-in behavior. Any unapproved divergence is a correctness bug.
Enforcement: CPython test suite, differential Tier 0/1/2/3 runs, Python-specific
semantic tests.

**Rule 68 — Observable Effects Must Not Be Reordered, Duplicated, or Deleted**
No optimization may delete, duplicate, hoist, sink, merge, or reorder
Python-visible effects unless the effect system proves semantic equivalence.
Python-visible effects include: attribute reads/writes, global/builtin
reads/writes, descriptor protocol invocation, `__getattr__`/`__getattribute__`/
`__setattr__`/`__delattr__`, `__getitem__`/`__setitem__`/`__delitem__`, import
side effects, allocation side effects where visible, exceptions raised,
finalizers and weakref callbacks where observable, tracing/profiling/monitoring
events, I/O effects, and changes to object identity or container membership
where visible. Enforcement: effect-chain verifier, golden IR tests, differential
semantic tests.

**Rule 69 — Only Provably Pure Expressions May Be Constant-Folded**
Constant folding may only apply to expressions whose result is independent of:
runtime state, object identity, hash randomization, environment variables,
time, randomness, locale, filesystem state, import state, global/builtin
mutation, object layout, GC state, refcounts, and thread scheduling. No call
with possible side effects may be constant-folded. Enforcement: verifier rule,
pure-node annotations, negative tests.

**Rule 70 — Dynamic Python Features Are First-Class Correctness Requirements**
The JIT must correctly handle or safely fall back for: `eval`, `exec`, `compile`,
dynamic imports, `setattr`/`delattr`, `getattr`, `vars`, `dir`, `inspect`,
`sys._getframe`, frame locals mutation where supported, code object introspection,
class mutation, monkey patching, builtin shadowing, module reloading where
supported, tracing/profiling hooks, `sys.monitoring`, weakrefs, finalizers, and
GC introspection. If a dynamic feature cannot be optimized safely, the system
must deoptimize or disable JIT for the affected scope. It must never silently
produce wrong introspection or semantics. Enforcement: dynamic-feature test
matrix, deopt tests, CPython API tests.

**Rule 71 — Specialization Requires Versioned, Invalidatable Dependencies**
Every specialization assumption must record a dependency on a versioned entity.
Examples: object shape/hidden class version, class version, type version, module
dictionary version, globals version, builtins version, function/code object
version, method resolution order version, descriptor version, dictionary
key/layout version, signature version, import state version, bytecode version,
and profile version. If any dependency changes, all dependent compiled code
must be invalidated or guarded. Enforcement: dependency graph tests, invalidation
fuzzing, IC tests.

**Rule 72 — Python Numeric Semantics Must Be Preserved Exactly**
Numeric specializations must preserve Python semantics, including: arbitrary
precision integers, integer overflow to larger integers where required, `bool`
as a subclass of `int`, float NaN behavior, negative zero, float/int conversion
rules, `OverflowError` behavior where specified, `ZeroDivisionError` behavior,
complex number semantics where supported, operator fallback to dunder methods,
and subclass overrides. Fast-math-style optimizations are forbidden unless
explicitly scoped, proven safe, and disabled by default. Enforcement: numeric
differential tests, overflow tests, NaN tests, dunder override tests.

**Rule 73 — Escape Analysis Must Not Eliminate Observable Objects**
Objects may be scalarized or eliminated only if they cannot be observed by:
`id()`, weak references, finalizers, `gc.get_objects` where supported, reference
counting APIs where supported, exception tracebacks, frame locals,
profiling/debugging hooks, user code escaping, dynamic introspection, native/C
extension calls, and monitoring events. If any escape path exists, the object
must be materialized. Enforcement: escape-analysis verifier, materialization
tests, deopt tests.

**Rule 74 — Python Exceptions Are Control-Flow Values, Not Native Exceptions**
Python exceptions must be represented as runtime values and control-flow edges.
Native C++ exceptions must not be used to implement Python exception propagation.
The JIT must preserve: exception type, exception value, traceback, exception
chaining (`raise ... from ...`), exception context, exception notes where
supported, `sys.exc_info()` semantics, traceback line numbers, frame
association, finally-block semantics, and with-block cleanup semantics.
Enforcement: exception semantic tests, traceback golden tests.

**Rule 75 — Frames Must Be Reconstructible on Demand**
If the JIT inlines, merges, elides, or optimizes frames, it must be able to
materialize a semantically correct Python frame when required by: tracebacks,
exceptions, debuggers, `sys._getframe`, `inspect`, frame locals access,
tracing/profiling, deoptimization, monitoring hooks, and user introspection.
`FrameState` must be sufficient to reconstruct: bytecode offset, `lasti`
equivalent, line number, locals, globals version, builtins version, closure
cells, free variables, exception state, tracing state, monitoring state, and
generator/coroutine suspension state where applicable. Enforcement: frame
materialization tests, deopt tests, debugger tests.

**Rule 76 — Generators, Coroutines, and Async Suspension Must Be JIT-Safe**
Generator and coroutine suspension points are semantic boundaries. The JIT
must correctly handle: yield/resume, await/resume, throw into
generators/coroutines, close semantics, `StopIteration` handling,
`GeneratorExit` handling, exception propagation across suspension, frame
reconstruction after suspension, local state after resume, and finalization of
unawaited/unresumed objects where specified. Suspension points must be valid
deopt/safepoint candidates. Enforcement: async/generator stress tests,
deopt-at-yield tests.

**Rule 77 — Debugging, Tracing, Profiling, and Monitoring Must Remain Correct**
The JIT must not break Python tooling. Supported tooling includes at least:
`sys.settrace`, `sys.setprofile`, `sys.monitoring`, debuggers, breakpoints
where supported, line events, call events, return events, exception events, and
instruction-level events where supported. When tooling is active, the JIT must
either: emit correct events with correct semantics, run unoptimized lower-tier
code, or fall back to Tier 0. Missing events, duplicate events, wrong line
numbers, or wrong exception events are correctness bugs. Enforcement:
tracing/profiling differential tests.

**Rule 78 — Object Identity and Reference-Count Semantics Must Be Explicit**
If VORTEX exposes CPython-compatible `sys.getrefcount`, reference-count-visible
behavior must match or be explicitly disabled. If using a moving GC: object
identity semantics for `id()` and `is` must remain correct, moving objects must
not expose unstable addresses to Python, pinned objects must be used where
address identity is observable, and handles or stable identity mechanisms must
be provided. No optimization may assume that object addresses are stable unless
the object model explicitly pins the object. Enforcement: identity tests, weakref
tests, GC tests.

**Rule 79 — No Assumptions About Hashes, Randomness, or Addresses**
The compiler must not persist or bake assumptions about: hash values,
`PYTHONHASHSEED`, dictionary iteration order beyond language guarantees, ASLR
addresses, object addresses, code addresses, randomized runtime values, or
nondeterministic allocation order. Persistent artifacts must not contain
address-dependent assumptions unless explicitly relocated/validated at load
time. Enforcement: hash-randomization CI, ASLR replay tests, persistent
artifact validation.

**Rule 80 — Static Typing Is Not Runtime Proof Unless Certified**
Type hints, `typing.Final`, `__static__`, or annotations do not by themselves
justify unsafe optimization. Static proofs must be based on: sealed types,
finality guarantees, module isolation, absence of dynamic mutation, verified
ABI constraints, verified import boundaries, absence of introspection/monkey
patching, and mechanically checked proof artifacts. If static proof cannot be
maintained, the code must use guards or fall back. Enforcement: static-mode
verifier, negative mutation tests.

---

## Part VIII: Deoptimization and GC Integration Laws

**Rule 81 — Deoptimization Metadata Is a Required Compilation Output**
A compilation is not complete until it has produced: deopt points, `FrameState`
snapshots, stack maps, GC reference maps, live range information, materialization
plan, interpreter re-entry information, dependency list, guard metadata, and
exception-state reconstruction info. If deopt metadata cannot be generated, the
compilation must fail and fall back. Enforcement: compiler verifier,
missing-deopt compile-fail tests.

**Rule 82 — FrameState Must Be Complete and Machine-Checkable**
`FrameState` must describe enough information to reconstruct the exact lower-tier
state. It must include: bytecode offset/instruction position, register-to-interpreter
slot mapping, stack slot types, object references, primitive values, constants,
closure cells, free variables, globals version, builtins version, exception state,
tracing/monitoring state, async/generator suspension state where relevant, any
materialized object graph, and reference-count or GC-handle state. The verifier
must reject incomplete `FrameState`. Enforcement: `FrameState` verifier, forced-deopt
tests.

**Rule 83 — Guard Failure Must Produce Exact Lower-Tier State**
When a guard fails, the runtime must resume in a state observationally
indistinguishable from the state the lower tier would have reached at that point.
This includes: local variables, stack state, exception state, side effects already
committed, reference counts or GC-equivalent state, frame visibility, line number,
monitoring/tracing state, and object materialization state. If exact reconstruction
is impossible, the speculation must be rejected at compile time. Enforcement:
differential deopt tests, forced-guard-failure fuzzing.

**Rule 84 — Deopt Loops Must Be Detected and Throttled**
Repeated deoptimization at the same site is a performance and correctness hazard.
The runtime must track: deopt count per site, deopt count per function, deopt
reason, time window, and tier history. If thresholds are exceeded, the system
must: disable the failing speculation, recompile with weaker assumptions, downgrade
tier, blacklist the function temporarily or permanently, and emit telemetry.
Enforcement: deopt-loop regression tests, telemetry validation.

**Rule 85 — Speculative Side Effects Must Be Reversible or Deferred**
Speculative optimization must not commit irreversible Python-visible side effects
before the speculation is proven. If an effect cannot be proven safe: defer it,
guard before it, materialize fallback state, or do not perform the optimization.
Memory stores that may be observed by Python, C extensions, finalizers, weakrefs,
or debugging tools are not freely rollback-able. Enforcement: effect-system audit,
speculative-store tests.

**Rule 86 — All GC References in JIT Code Must Be Tracked**
Generated code must not hold raw object pointers in registers, stack slots, or
embedded constants across safepoints unless those references are recorded in GC
maps. Rules: every reference register across a call/safepoint must be in a stack
map, every reference spill must be visible to GC, embedded object pointers must
use handles or be otherwise tracked, and object references must not be hidden in
untracked integer registers unless explicitly tagged and supported. Enforcement:
GC map verifier, GC stress tests.

**Rule 87 — Read and Write Barriers Must Be Correct**
If the GC requires write barriers, every store of a reference in generated code
must execute the correct barrier. If the GC requires read barriers, load barriers,
or forwarding checks, every relevant reference load must execute them. Missing
barriers are blocker bugs. Enforcement: barrier verifier, GC stress, moving-GC
tests.

**Rule 88 — Generated Code Must Poll Safepoints**
JIT code must include safepoint polls at: loop backedges, function calls,
allocation sites, long native transitions where specified, OSR entry/exit points,
tier transition points, and invalidation points where required. Safepoint latency
must be bounded. Enforcement: safepoint stress tests, GC pause tests.

**Rule 89 — All JIT Frames Must Be Walkable**
Every JIT frame must be walkable by: GC, deoptimizer, profiler, debugger,
exception unwinder, stack overflow checks, and diagnostic tools. Frame metadata
must include: frame size, return address location, saved registers, stack map,
deopt info, callee-saved register locations, Python frame association, and
native/managed transition markers. Enforcement: stack-walking tests, GC/deopt/
profiler integration tests.

**Rule 90 — Stack Overflow and Recursion Limits Must Be Checked**
JIT code must respect Python recursion limits and native stack limits. Checks
must occur: before entering JIT frames, before inlined calls, before recursive
calls, before OSR entry where applicable, and before native transitions where
stack usage changes. Failure must produce the correct Python exception, not a
crash. Enforcement: recursion-limit tests, stack-overflow tests.

**Rule 91 — Allocation Fast Paths Must Handle Failure Safely**
Allocation fast paths may optimize the common case, but slow paths must handle:
heap exhaustion, memory allocation failure, GC pressure, object finalization
hooks, allocation callbacks where specified, and Python `MemoryError` semantics.
Generated code must not abort the VM on allocation failure unless the VM is in an
unrecoverable state defined by the runtime spec. Enforcement: low-memory tests,
allocation-failure injection.

**Rule 92 — Runtime Call Transitions Must Preserve ABI and Runtime State**
Calls from JIT code into runtime helpers must preserve: calling convention, stack
alignment, callee-saved registers, GC state, exception state, thread state,
Python interpreter state, and floating-point/vector register state as required.
Runtime helpers must not assume JIT register contents beyond ABI. Enforcement:
ABI tests, register-clobber tests.

**Rule 93 — C Extensions and FFI Are Opaque Unless Proven**
Calls into C extensions or Python C-API code are opaque barriers unless a formal
ABI/effect proof exists. Assume C calls may: mutate arbitrary Python state, call
back into Python, allocate, raise Python exceptions, change classes/modules/globals,
invalidate specialization assumptions, acquire/release locks/GIL, trigger GC,
observe object layout, and corrupt assumptions if misused. Optimizations across
FFI boundaries require explicit proof and invalidation rules. Enforcement: FFI
barrier tests, C callback mutation tests.

**Rule 94 — Weak References, Finalizers, and GC Callbacks Must See Valid State**
JIT code must not leave weak references, finalizers, or GC callbacks in states
where they observe: partially initialized objects, invalid forwarding pointers,
untracked references, missing barriers, stale object headers, inconsistent
reference counts, or objects that should have been materialized but were not.
Enforcement: weakref/finalizer stress tests.

**Rule 95 — Object Shape and Class Mutation Must Invalidate Specialized Code**
Any change to assumptions used by inline caches or specialization must invalidate
dependent code. This includes: class attribute changes, instance dictionary layout
changes, `__slots__` changes where supported, metaclass changes, method
redefinition, property replacement, descriptor replacement, MRO changes, builtin
shadowing, global rebinding, module dictionary mutation, and code object
replacement. Enforcement: mutation-after-JIT tests.

**Rule 96 — Tier 0 Is the Universal Correctness Fallback**
Every executable function must be runnable in Tier 0. No feature may be "JIT-only"
unless explicitly part of a documented static mode. If Tier 1/2/3 cannot compile,
patch, deopt, or execute code correctly, execution must fall back to Tier 0.
Enforcement: fallback tests, JIT-disable tests.

---

## Part IX: Code Cache, Patching, and Security Laws

**Rule 97 — Executable Memory Must Be W^X**
JIT memory pages must never be simultaneously writable and executable. Code
generation and patching must use one of: write-then-execute with protection
changes, separate staging and executable pages, atomic patching of existing
executable locations where safe, or platform-approved JIT memory mechanisms.
Enforcement: memory-protection tests, OS-specific audits.

**Rule 98 — Code Publication Must Be Atomic**
Function entry points, OSR entry points, trampolines, and metadata pointers must
be published atomically. No thread may observe: partially initialized code,
uninitialized metadata, missing deopt info, missing GC maps, or half-patched jump
tables. Publication must use release semantics; consumers must use acquire
semantics. Enforcement: TSAN tests, concurrent-install stress tests.

**Rule 99 — Runtime Patching Must Be Safe Against Concurrent Execution**
Patching running code must be safe. Requirements: patch sites must be aligned and
architecturally safe, instruction sequences must not create invalid intermediate
instructions, instruction cache coherence must be handled where required,
concurrent threads must never execute corrupted instructions, and patching must
either use safepoints or architecture-safe atomic sequences. Enforcement:
patch-under-load tests, architecture-specific tests.

**Rule 100 — Old Code May Be Freed Only After Quiescence**
Old compiled code, deopt metadata, and dependency records must not be reclaimed
until no thread can be executing or depending on them. Use: epoch-based
reclamation, RCU-like quiescence, safepoint-based retirement, or reference
counting for code objects where appropriate. Code reclamation must be distinct
from IR node reclamation. Enforcement: concurrent code retirement tests.

**Rule 101 — Every Compiled Artifact Must Record Dependencies**
Every compiled function must record dependencies sufficient for invalidation.
Dependency examples: code object identity/version, function identity/version,
class/shape versions, global/builtin versions, module versions, type versions,
profile version, IR version, compiler version, target feature set, ABI version,
and runtime configuration. Enforcement: dependency graph verifier.

**Rule 102 — Generated Code Must Be Constrained**
Generated code must only call approved runtime entrypoints and must not directly:
perform arbitrary syscalls unless mediated by the runtime, write outside its own
frame/runtime-approved memory, execute arbitrary user-provided machine code, load
arbitrary dynamic libraries unless approved, or bypass sandbox/security policy.
Enforcement: codegen allowlist, backend audit, security tests.

**Rule 103 — Platform Exploit Mitigations Must Be Enabled Where Available**
JIT must integrate with platform security features where available: non-executable
stack, non-executable heap, CFI, shadow stacks, PAC/BTI on ARM64, pointer
authentication where supported, CET where supported, ASLR-safe code generation,
code signing where required, and sandbox compatibility. If a mitigation is
unavailable, the risk must be documented and configurable. Enforcement: platform
security matrix.

**Rule 104 — JIT Spraying Defenses Are Required**
The JIT must not turn attacker-controlled data into executable instruction
streams without mitigation. Mitigations may include: constant blinding, avoiding
embedding uncontrolled immediate sequences, separating executable code from
embedded data, limiting executable constant islands, code cache entropy/
randomization where appropriate, and validating inputs that influence codegen.
Enforcement: security review, exploit PoC tests.

**Rule 105 — Profiles, Bytecode, and AOT Artifacts Are Untrusted**
Profile data, serialized IR, AOT artifacts, caches, and bytecode inputs must be
validated before use. Malformed inputs must not cause: undefined behavior, memory
corruption, arbitrary code execution, VM crashes, or silent miscompilation. Invalid
artifacts must be rejected or ignored with telemetry. Enforcement: artifact
fuzzing, schema validation.

**Rule 106 — Code Cache Pressure Must Be Managed**
The code cache must have explicit budgets and eviction policies. The system must
monitor: total code size, metadata size, dependency graph size, number of live
compiled functions, number of invalidated functions, patchpoint count, and deopt
metadata size. When pressure exceeds budgets, the system must throttle compilation,
evict cold code, or fall back. Enforcement: code-cache stress tests.

**Rule 107 — AOT Artifacts Must Include a Compatibility Manifest**
AOT artifacts must include: Python version, IR version/hash, compiler version,
pass pipeline hash, target architecture, target feature set, ABI hash, runtime
configuration hash, dependency fingerprints, security policy version, and creation
metadata. Enforcement: manifest validation tests.

**Rule 108 — AOT Artifacts Must Be Verified Before Loading**
AOT loading must verify: manifest compatibility, integrity checksum/signature
where required, dependency validity, target feature support, ABI compatibility,
and security policy compatibility. On mismatch, the artifact must be rejected.
Silent loading of incompatible artifacts is forbidden. Enforcement: stale/corrupt
AOT artifact tests.

---

## Part X: Concurrency, Compilation Scheduling, and Tiering Laws

**Rule 109 — Compiler, Runtime, and GC Shared State Must Be Race-Free**
All shared state accessed by mutator threads, compiler threads, GC threads, and
background services must be synchronized using documented atomic/locking
protocols. TSAN-clean is mandatory for supported concurrent tests. Enforcement:
TSAN CI, concurrency stress tests.

**Rule 110 — Function Pointer Swaps Must Be Safe and Reversible**
Installing new code must: use atomic publication, preserve old code until safe,
avoid torn calls, avoid invalidating metadata still needed by running threads,
and support rollback where possible. Function installation must be testable
independently of compilation. Enforcement: concurrent install/uninstall tests.

**Rule 111 — Safepoint Latency Must Be Bounded**
Threads must be able to reach a safepoint within a documented bounded time.
Long-running generated loops must contain polls. Native helpers that run for long
durations must cooperate with suspension protocols. Enforcement: GC pause tests,
suspension stress tests.

**Rule 112 — Compilation Latency and Memory Budgets Must Be Defined**
Each tier must have explicit budgets: Tier 1 compile latency, Tier 2 compile
latency, Tier 2 memory usage, Tier 3 AOT compile time where relevant, IR memory
usage, pass fixpoint iteration limits, code size limits, and deopt metadata
limits. Budget violations must trigger fallback or cancellation, not mutator
stalls. Enforcement: compile-budget benchmarks.

**Rule 113 — Compilations Must Be Cancellable**
If a function is invalidated while compiling, the compiler must be able to cancel
or discard the result without leaking memory or installing stale code.
Enforcement: invalidation-during-compilation tests.

**Rule 114 — Hotness Counters Must Be Robust**
Profiling counters must be: thread-safe or explicitly racy-with-bounded-error,
saturating or overflow-safe, decaying where appropriate, resistant to pathological
overflow, and correlated with deopt feedback. Undefined behavior from counter
overflow is forbidden. Enforcement: counter fuzzing, long-run soak tests.

**Rule 115 — Recompilation Must Be Throttled**
Repeated compilation of the same function must be limited by: maximum recompiles
per function, exponential backoff, deopt-history awareness, code-cache pressure
awareness, and budget awareness. No function may cause unbounded compile churn.
Enforcement: recompile-thrash tests.

**Rule 116 — OSR Entry and Exit Must Be Semantically Exact**
On-stack replacement must preserve exact program state at OSR entry and exit.
OSR must handle: loop induction variables, iterator state, exception state,
closure cells, locals, stack values, generator/coroutine state if supported,
and deopt from OSR code back to interpreter. Enforcement: OSR state
reconstruction tests.

**Rule 117 — Invalidation Must Be Ordered and Visible**
Invalidation of dependencies must be visible before new assumptions are relied
upon. The system must avoid: executing stale code after invalidation beyond
allowed grace, installing code based on already-invalid dependencies, and racing
invalidation with installation. Enforcement: invalidation race tests.

**Rule 118 — No Global Locks on Hot Runtime Paths**
Global locks are forbidden in hot runtime paths unless explicitly approved and
budgeted. Hot paths include: inline-cache updates, guard checks, function entry
dispatch, allocation fast paths, read/write barriers, safepoint polls, and basic
object access. Enforcement: lock profiling, scalability tests.

**Rule 119 — Tier Transitions Must Be Observable**
All tier transitions must be recorded: Tier 0→1, Tier 1→2, Tier 2→3 where
applicable, deopt to lower tier, code invalidation, blacklist events, fallback
events, and recompilation events. Telemetry must include reasons and counters.
Enforcement: telemetry schema tests.

**Rule 120 — Compiler Bugs Must Not Crash User Programs**
A compiler failure should degrade performance, not terminate the application.
Compiler/runtime JIT bugs should result in: fallback, disabled optimization,
diagnostic log, telemetry, and replay artifact where possible. Process aborts
are only acceptable for unrecoverable VM corruption and must be treated as P0
bugs. Enforcement: fault-injection tests.

---

## Part XI: IR, Passes, and Backend Laws

**Rule 121 — The IR Must Have an Explicit Effect Model**
The IR must explicitly represent effects and ordering. Effect classes should
include at least: pure computation, allocation, Python object mutation,
global/builtin mutation, import effects, exception effects, I/O effects, FFI
effects, GC effects, monitoring/tracing effects, deopt/guard effects, memory
reads/writes, and reference-count or GC barrier effects. Passes must not reorder
effects without proof. Enforcement: effect-chain verifier.

**Rule 122 — Speculative Nodes Must Carry Metadata**
Every speculative node must record: speculation kind, PGO/static source,
confidence, guard plan, `FrameState`, deopt target, cost, invalidation
dependency, and rollback/deferred-effect plan. No implicit speculation is
allowed. Enforcement: IR verifier.

**Rule 123 — Passes Must Declare Contracts**
Each pass must declare: required IR properties, produced IR properties,
invalidated analyses, supported tiers, budget, determinism requirements, target
dependencies, required verifier checks, and telemetry hooks. Passes that cannot
satisfy their contract must fail safely. Enforcement: pass registry, contract
tests.

**Rule 124 — Compilation Must Be Deterministic and Replayable**
Given the same source/bytecode, compiler version, flags, profile snapshot, target
description, RNG seed, and feature configuration, compilation must produce
deterministic IR and code selection, except for explicitly documented
nondeterminism. Nondeterminism sources must be logged. Enforcement: deterministic
replay tests.

**Rule 125 — Passes Must Not Use Hidden Global Mutable State**
Hot-path passes must not depend on hidden global mutable state. Allowed global
state: immutable configuration, interned symbol tables with proper
synchronization, read-only target descriptions, and versioned caches with
explicit invalidation. Hidden singletons in pass logic are forbidden.
Enforcement: static analysis, code review checklist.

**Rule 126 — The Verifier Must Check Deopt and GC Metadata**
The graph verifier must check not only IR consistency but also: every guard has
`FrameState`, every deopt point is reachable, every GC reference across safepoint
has a map, every effect chain is continuous, every speculative node has
invalidation info, and every materialized object graph is acyclic or properly
handled. Enforcement: debug verifier runs after every pass.

**Rule 127 — Backend Lowering Must Preserve IR Semantics**
Lowering from high/mid IR to machine code must preserve: effect order, exception
semantics, numeric semantics, overflow behavior, GC reference liveness, safepoint
placement, deopt point mapping, and stack layout constraints. Backend
optimizations may not silently change IR semantics. Enforcement: backend golden
tests, differential tests.

**Rule 128 — Register Allocation Must Be GC-Reference Safe**
The register allocator must ensure: GC references are not lost across
calls/safepoints, spills of references are tracked, register maps are generated,
callee-saved/caller-saved conventions are respected, and reference registers do
not alias untracked integer registers unless allowed by object representation.
Enforcement: register-map verifier, GC stress tests.

**Rule 129 — Target Features Must Be Gated and Recorded**
Use of CPU features must be: runtime-detected for JIT, build-time validated for
AOT, recorded in code metadata, and protected by feature guards where needed.
Generated code must not execute unsupported instructions. Enforcement:
target-feature mismatch tests.

**Rule 130 — Every IR Node and Trampoline Must Have a Specification**
No IR node, runtime stub, or trampoline may exist without documentation covering:
semantics, effects, tier behavior, lowering, verifier constraints, deopt behavior,
GC behavior, and tests. Enforcement: documentation lint, IR node registry.

**Rule 131 — Static Proofs Must Be Mechanically Checked**
Tier 3 static optimizations may not rely on human-only proof. Static proofs must
be represented as machine-checkable artifacts or verifier constraints. If proof
cannot be checked automatically, the optimization must use guards or be disabled.
Enforcement: proof-verifier tests.

**Rule 132 — Every Optimization Must Have a Kill Switch**
Every nontrivial optimization should be disableable by: compiler flag, environment
variable, configuration knob, runtime feature gate, or per-function annotation
where appropriate. This enables bisection and incident response. Enforcement:
feature-flag matrix.

---

## Part XII: Testing, Observability, and Governance Laws

**Rule 133 — Differential Oracle Testing Must Run Continuously**
CI must compare behavior across: Tier 0, Tier 1, Tier 2, Tier 3/AOT where
available, and CPython reference where applicable. Tests must include: normal
programs, exceptions, async/generators, C extension interactions, dynamic class
mutation, tracing/profiling enabled, GC stress, low-memory stress, recursion
limits, and large integers/floats/NaNs. Enforcement: CI differential matrix.

**Rule 134 — Fuzzing Must Cover Bytecode, IR, Profiles, and Artifacts**
Fuzzing must target: Python source/bytecode inputs, IR inputs, serialized
profiles, AOT artifacts, code cache metadata, patching sequences, deopt metadata,
GC barrier sequences, FFI boundaries, and class/attribute mutation schedules.
Untriaged fuzz failures block release. Enforcement: scheduled fuzz jobs.

**Rule 135 — Sanitizer Matrix Is Mandatory**
CI must run supported configurations with: ASan, UBSan, TSan where concurrency
exists, MSan where supported, debug asserts, release builds, interpreter-only
mode, JIT-enabled mode, and AOT mode where applicable. Enforcement: CI matrix.

**Rule 136 — GC and Deopt Stress Tests Must Be First-Class**
Dedicated stress modes must: force frequent GC, force moving GC where applicable,
force allocation failure, force guard failure, force deopt at every supported
point, force weakref/finalizer activity, and force code invalidation under load.
Enforcement: nightly stress, release gating.

**Rule 137 — Code Installation and Patching Must Be Concurrency-Tested**
CI must test: installing code while executing old code, invalidating code while
running, patching under load, retiring code under load, OSR entry during
invalidation, and deopt during patching. Enforcement: concurrency stress tests.

**Rule 138 — Performance Gates Must Measure More Than Throughput**
Performance CI must measure: startup time, warmup time, peak throughput, tail
latency, compile latency p50/p99, deopt rate, guard overhead, code size, memory
usage, GC pause impact, compile CPU cost, memory pressure, and tier transition
counts. A regression in any critical dimension requires waiver. Enforcement:
benchmark suite with thresholds.

**Rule 139 — Telemetry Must Be Structured, Stable, and Privacy-Safe**
Telemetry must record: compile attempts, compile failures, fallback reasons,
guard failures, deopt reasons, invalidations, code cache pressure, budget
violations, blacklist events, and performance counters. Telemetry must not
include source code, user data, or secrets unless explicitly opted in.
Enforcement: schema validation, privacy review.

**Rule 140 — Replay Artifacts Must Be Sufficient for Debugging**
A failed compilation or deopt event should be replayable from: source/bytecode
hash, IR snapshot, pass pipeline state, profile snapshot, compiler flags, target
description, RNG seed, runtime config, dependency versions, and failure location.
Debugging should start from replay, not anecdote. Enforcement: replay artifact
tests.

**Rule 141 — ABI and FFI Must Have Dedicated Tests**
Dedicated tests must cover: Python→native calls, native→Python callbacks,
register clobbering, stack alignment, exception propagation through FFI, GIL/
free-threading interactions where relevant, reference ownership transfer, struct
passing where supported, varargs/keyword conventions, and error return conventions.
Enforcement: ABI test suite.

**Rule 142 — Security Tests Must Be Part of CI**
Security checks should include: W^X scans, executable memory accounting, JIT
spraying PoCs, malformed artifact loading, code-cache exhaustion, patch race
attempts, sandbox escape tests where applicable, and dependency vulnerability
scans. Enforcement: security CI lane.

**Rule 143 — CPython Compatibility Must Be Tracked Explicitly**
The project must maintain: supported CPython version range, supported
standard-library subset, known divergences, unsupported features, test suite pass
requirements, and allowed failure list with owners and expiry dates. Enforcement:
compatibility dashboard.

**Rule 144 — All Major Optimizations Must Be Feature-Gated**
Every major optimization must be capable of being disabled independently for
bisection and emergency response. Examples: inlining, PEA, SLP, LICM, GVN, effect
reordering, IC specialization, type specialization, unrolling, OSR, Tier 2
compilation, and Tier 3 AOT loading. Enforcement: flag matrix test.

**Rule 145 — Exceptions to Rules Require an Exception Register**
No rule may be silently bypassed. Exceptions must include: rule ID, reason,
owner, risk assessment, mitigation, telemetry, expiry date, and tech lead
approval. Expired exceptions automatically become release blockers. Enforcement:
exception register.

**Rule 146 — Every Rule Must Have Enforcement Metadata**
Each rule in this document must specify: enforcement mechanism, owner, severity,
test coverage, and waiver policy. Rules without enforcement should be moved to
guidelines or given an enforcement plan. Enforcement: rule metadata lint.

**Rule 147 — Maintain a Compliance Matrix**
The repository must maintain a mapping from each rule to: CI check, test suite,
verifier, review checklist, documentation, and owner. This matrix must be
reviewed each release. Enforcement: release checklist.

**Rule 148 — Architectural Decisions Require ADRs**
Any significant compiler/runtime decision must have an Architecture Decision
Record. ADRs must cover: context, options considered, decision, consequences,
performance impact, correctness impact, security impact, and rollback plan.
Enforcement: PR template requirement.

**Rule 149 — Builds Must Be Hermetic and Dependencies Must Be Pinned**
Compiler/runtime builds must be reproducible. Requirements: pinned dependencies,
locked toolchains where practical, no network access during tests, reproducible
artifact hashes, and supply-chain review for new dependencies. Enforcement: build
reproducibility tests.

**Rule 150 — Stale Documentation Is a Defect**
Documentation must be updated in the same PR as behavior changes. This includes:
rules doc, IR spec, effect system spec, ABI spec, pass documentation, runtime
documentation, telemetry schema, and compatibility matrix. Stale docs are treated
like stale code. Enforcement: docs lint, PR checklist.

---

## Part XIII: Definitions

**Hot path:** Compiler pass execution, guard execution, inline-cache fast paths,
allocation fast paths, deopt entry trampolines, and dispatch loops. Not deopt
materialization (which may allocate under budget).

**Guard:** A runtime mechanism that validates a speculative assumption. May be a
hardware branch, shape/version check, patchpoint, trap, dependency invalidation,
or hardware check.

**FrameState:** A snapshot attached to a speculative node that allows the
deoptimizer to reconstruct the exact lower-tier execution state.

**Deopt:** The process of transferring execution from a higher tier to a lower
tier (typically Tier 0) while preserving observable program state.

**Safe point:** A point in generated code where the thread can safely pause for
GC, deopt, or suspension. Must have bounded latency.

**Observable behavior:** Program output, exceptions, side effects, object mutation,
weakref behavior, finalization, frame introspection, and monitoring events.

**Effect:** A Python-visible operation that cannot be freely reordered, deleted,
or duplicated without semantic proof.

**Dependency:** A versioned entity that a specialization relies on. If the
dependency changes, the specialization must be invalidated or guarded.

**Code installation:** The atomic publication of a compiled function's entry
point, metadata, and deopt info.

**Quiescence:** A state where no thread is executing old code or depending on
old metadata, allowing safe reclamation.

**Speculation:** An optimization that assumes a runtime property holds, guarded
by a mechanism that triggers deopt on failure.

**Static proof:** A mechanically-checked artifact demonstrating a property holds
without runtime guards.

**Profile confidence:** A metric combining sample count, stability, age, decay,
variance, and deopt correlation. Low-confidence data must not trigger aggressive
speculation.

**Tier transition:** A change in execution tier (Tier 0→1→2→3 or deopt to lower).
Must be observable and recorded.

**Fallback:** Graceful degradation to a lower tier or disabled optimization when
compilation or speculation fails.

**Kill switch:** A mechanism to disable a specific optimization for bisection
and incident response.

---

## Part XIV: Normative References

- CPython 3.16 semantics (the semantic oracle per Rule 67)
- `docs/ir_spec.md` — IR node specifications
- `docs/effect_system.md` — effect model and effect-chain rules
- `docs/abi.md` — calling conventions and FFI
- `docs/python316_semantics.md` — Python 3.16-specific behavior
- VORTEX telemetry schema
- VORTEX compatibility matrix
- Target platform ABI documents (SysV x86-64, AAPCS64)
- Security policy document

---

## Part XV: Rule Severity and Waiver Process

**Severity levels:**
- **P0 (Blocker):** Silent data corruption, security vulnerabilities, crashes on
  valid input. Blocks release.
- **P1 (Critical):** Wrong results, missing deopt metadata, GC unsafety. Blocks
  merge to main.
- **P2 (Major):** Performance regressions >5%, missing tests, documentation debt.
  Requires waiver with expiry.
- **P3 (Minor):** Style, naming, minor optimizations. Tracked but non-blocking.

**Waiver process:**
1. File an exception in the exception register (Rule 145).
2. Include: rule ID, reason, owner, risk assessment, mitigation, telemetry,
   expiry date, and tech lead approval.
3. Waivers auto-expire. Expired waivers become release blockers.
4. No rule may be silently bypassed — silent bypass is itself a Rule 145 violation.

---

## Part XVI: Compliance Matrix

The full compliance matrix (Rule → CI check, test suite, owner) is maintained
in `docs/compliance_matrix.md` and reviewed each release. Example entries:

| Rule | Enforcement | Test Suite | Owner | Notes |
|------|------------|------------|-------|-------|
| 3 | IR verifier | guard_tests | compiler team | Every PGO decision needs a guard |
| 67 | differential CI | cpython_compat | runtime team | CPython is the oracle |
| 86 | GC map verifier | gc_stress | GC team | No untracked refs across safepoints |
| 97 | OS memory tests | security_ci | security team | W^X mandatory |

---

*End of VORTEX Compiler Laws & Architecture Specification.*
*Compliance is not optional. It is the foundation of VORTEX.*
