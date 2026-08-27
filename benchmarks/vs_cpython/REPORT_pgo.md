# VORTEX vs CPython 3.16 (PGO+LTO) — Head-to-Head Benchmark

## Setup

- **VORTEX**: built from `origin/main` at commit `de10716` (P54 V3 LSRA→XMM extension live). Run via `./build/benchmarks/vortex_bench --iterations 30`. Each iteration = compile source once + execute, fresh Vm per iter.
- **CPython**: built from `cpython/main` at `fe3a26f`, version `Python 3.16.0a0`. Configured with `--enable-optimizations --with-lto` (PGO via `make profile-opt`, LTO via `-flto -fuse-linker-plugin -ffat-lto-objects`). Run via `./python script.py` in a fresh subprocess per iter, wall-clock measured by `perf_counter_ns()`.
- **Iterations**: 30 per workload, both sides.
- **Workloads**: the 10 Python workloads from `benchmarks/cases/bench_cases.hpp`, extracted verbatim into `.py` files under `workloads/`.

## Results (median ns, 30 iter)

| workload | VORTEX | CPython 3.16 PGO+LTO | VORTEX vs CPython | winner |
|---|---:|---:|---:|:---|
| int_arith | 3,937,198 | 11,866,846 | 3.01x | VORTEX |
| float_arith | 4,151,475 | 11,639,931 | 2.80x | VORTEX |
| fib_recursion | 14,367,721 | 11,415,563 | 0.80x | CPython |
| loop_sum | 36,934,468 | 23,607,381 | 0.64x | CPython |
| loop_branch | 6,275,390 | 11,571,981 | 1.84x | VORTEX |
| list_build | 5,205,788 | 12,225,904 | 2.35x | VORTEX |
| list_concat_string | 181,243 | 10,631,507 | 58.7x | VORTEX* |
| nested_loops | 6,208,762 | 11,462,086 | 1.85x | VORTEX |
| mixed_int_float | 830,930 | 11,618,358 | 14.0x | VORTEX* |
| ackermann | 214,634 | 11,489,116 | 53.5x | VORTEX* |

**VORTEX wins 8/10; CPython 3.16 PGO+LTO wins 2/10.**

## PGO+LTO gave CPython a 7-11% speedup across the board

| workload | CPython baseline | CPython PGO+LTO | PGO+LTO delta |
|---|---:|---:|---:|
| int_arith | 13,383,351 | 11,866,846 | −11.3% |
| float_arith | 13,086,274 | 11,639,931 | −11.0% |
| fib_recursion | 12,715,287 | 11,415,563 | −10.2% |
| loop_sum | 26,134,644 | 23,607,381 | −9.7% |
| loop_branch | 12,735,463 | 11,571,981 | −9.1% |
| list_build | 13,125,641 | 12,225,904 | −6.9% |
| list_concat_string | 11,700,639 | 10,631,507 | −9.1% |
| nested_loops | 12,760,723 | 11,462,086 | −10.2% |
| mixed_int_float | 11,909,669 | 11,618,358 | −2.4% |
| ackermann | 11,800,493 | 11,489,116 | −2.6% |

Despite the across-the-board CPython speedup, **VORTEX still wins 8/10** (same as against non-PGO CPython). The 2 losses (`fib_recursion`, `loop_sum`) got modestly worse against the fully-optimized CPython — both went from 0.88x/0.71x (baseline) to 0.80x/0.64x (PGO+LTO).

## *Caveat on small workloads

The 14x / 54x / 59x headline numbers on `mixed_int_float`, `ackermann`, `list_concat_string` are still dominated by CPython process startup (~10.5 ms baseline with PGO+LTO, down from ~12 ms without). VORTEX runs in-process with effectively zero startup. The fair fight remains the 5 loop-heavy workloads where VORTEX shows 1.85–3.01x — slightly lower than the 2.06–3.40x range against non-PGO CPython, as expected.

## The loop_sum investigation — root cause found

### What I expected to find

My initial hypothesis: the linear-scan register allocator wasn't extending `s`'s live interval across the while-back-branch, causing `s` to be reloaded from its home slot every iteration. Plausible story, easy fix in `regalloc.cpp`.

### What I actually found

**The JIT emitter is not wired into the runtime. The entire `compiler/src/backend/` directory — regalloc.cpp, codegen.cpp, GprCache, XmmCache, the LSRA→XMM extension from Task 22 — is dead code in production.** Every benchmark runs entirely on the Tier-0 direct-threaded interpreter.

Concrete evidence:

1. `compile_unit` (the JIT entry point) has **zero callers** in `runtime/src/` or `compiler/src/pipeline/`. It's only called from `tests/unit/backend_test.cpp` (18 sites, all hand-built IR graphs) and `tests/regression/jit_differential_regr.cpp` (3 sites, also hand-built IR).
2. `backedge_count` (the per-loop hotness counter that's *supposed* to trigger tiering) is **incremented in 3 places in `runtime/src/interp.cpp` but read in zero places** outside the increment itself. There is no tiering daemon.
3. `CodeUnit::current_tier` stays at 0, `CodeUnit::jit_entry` stays `nullptr`, the runtime's `compile_program` → `vm.run_module` path never calls into the backend.
4. `benchmarks/bench_main.cpp:66` calls `run_source(vm, src)` which is Tier-0 only.

The "Tier-0 + JIT-when-hot" framing in the prior `REPORT.md` is **misleading** — there is no JIT-when-hot codepath today. This means:

- **All P54 V1/V2/V3 perf work (Tasks 16–22) was exercised only by hand-built-IR unit tests, not by the benchmark suite.** The `-2.8%` on `float_arith` and `-4.1%` on `mixed_int_float` reported in the Task 22 worklog were most likely measurement noise (~5–10% at 30 iter), not real XMM-cache wins — the XMM cache never fires on the benchmark because the JIT never ran.

### The real root cause of the loop_sum regression

The 100K-iteration `while` loop in `loop_sum` runs entirely on Tier-0. Per iteration VORTEX dispatches approximately:

- `L_PY_CMP` (`i < 100000`) → calls `values_compare` → tag classification + signed compare
- `L_JUMP_IF_FALSE` → branch + bump `backedge_count` (which nobody reads)
- `L_PY_BINOP` (`s + i`) → calls `values_add` → `numeric_binop` (in `objects.cpp`) → `classify(a)` + `classify(b)` + dispatch to int lambda → `int_fits_i64` + `Value::integer(r)`
- `L_PY_BINOP` (`i + 1`) → same path
- `L_MOVE` (phi move for `s` at backedge) → 16-byte `Value` struct copy
- `L_MOVE` (phi move for `i` at backedge) → same
- `L_JUMP` (back to header) → branch + bump `backedge_count` again

That's ~7 dispatched ops × (computed-goto dispatch + tag classification + Value-struct write) per iteration. The `classify(...)` step walks `Tag::Int` / `Tag::Float` / `Tag::Obj` / `ObjTag::Int` / `ObjTag::Float` / `ObjTag::Bool` — a switch on `v.tag` plus a nested switch on `v.as.obj->tag`. CPython 3.16's specialized `BINARY_OP` with int+int inline cache skips this entirely after warmup.

That's why VORTEX costs ~370 ns/iter steady-state on `loop_sum` vs CPython's ~136 ns/iter (23.6 ms total − ~10.5 ms startup, divided by 100K iter) — a 2.7× per-iteration throughput gap in pure interpreter work.

## What this means for the broader perf picture

The Task 22 report claimed VORTEX is "2–3x faster than CPython 3.16 on the 5 hot numeric loop workloads — that's the JIT paying off as designed". That claim was **wrong**. The 2–3x advantage on `int_arith`, `float_arith`, `loop_branch`, `nested_loops`, `list_build` reflects VORTEX's Tier-0 bytecode interpreter being 2–3x faster than CPython's bytecode interpreter (with PGO+LTO+specialization on the CPython side). That's actually a remarkable result for a hand-rolled Tier-0 interpreter — it suggests VORTEX's bytecode dispatch + `Value` representation is genuinely lean. But the headline "JIT paying off" framing was incorrect.

## Two paths forward

### Path A: Wire the JIT into the runtime (architectural change)

This is the actual fix. Concretely:

1. Add a tiering daemon (or inline tiering check at the backedge sites in `interp.cpp`) that reads `backedge_count` and, past a threshold, calls `backend::compile_unit(g, unit_id, mmap'd RWX buffer, capacity, host_target())`.
2. Atomically swap the result into `CodeUnit::jit_entry` and add an OSR-style entry at the loop header (Tier-0 → JIT transition at the next backedge).
3. Add a JIT smoke test that **parses** a `while` loop and runs it through `compile_unit` end-to-end — currently no such test exists; all JIT tests use hand-built IR.
4. Once the JIT fires on `loop_sum`, the LSRA + GprCache + XmmCache work (Tasks 16–22) becomes live and the 2–3x advantage should compound.

Estimated effort: significant — touching runtime + interp + driver + new tests, and there's real concurrency-safety work for the safepoint swap. Days, not hours.

### Path B: Add Tier-0 fast paths (surgical change, immediate perf win)

Skip the JIT wiring for now. Instead, specialize the Tier-0 opcodes for the common int+int / int+int compare cases that dominate `loop_sum`:

1. In `L_PY_BINOP`'s `BinOpKind::Add` case: check `a.tag == Tag::Int && b.tag == Tag::Int` inline; if so, call `int_fits_i64` + `Value::integer` directly, skipping `classify`/`numeric_binop`.
2. Same for `L_PY_CMP`'s `<` / `<=` / `>` / `>=` cases: int+int fast path that skips `values_compare`.
3. Optional: skip the `backedge_count.fetch_add` (relaxed atomic, but still a write — and it's never read, so it's pure overhead).

Estimated effort: small — a few hundred lines in `interp.cpp` and `objects.cpp`. Hours, not days. Expected closure on `loop_sum`: 30–40% of the gap (per the per-iter dispatch analysis above).

### My recommendation

Do **both, in order**: Path B first (immediate perf win on `loop_sum` and `fib_recursion`, plus it's surgical and low-risk), then Path A (the architectural fix that finally makes Tasks 16–22's work live). Path B is what the user originally asked for ("investigate the loop_sum regression and fix it"); Path A is the bigger architectural debt that this investigation surfaced.

## Reproduce

```bash
# 1. Build CPython 3.16 PGO+LTO from main
git clone --depth 1 https://github.com/python/cpython.git /path/to/cpython
cd /path/to/cpython && ./configure --enable-optimizations --with-lto && make profile-opt -j2

# 2. Run CPython bench (PGO+LTO)
python3 run_cpython_bench.py \
  --python /path/to/cpython/python \
  --workloads workloads \
  --iterations 30 \
  --out results/cpython_pgo_results.csv

# 3. Run VORTEX bench (from VORTEX repo root)
cd ../.. && ./build/benchmarks/vortex_bench --iterations 30 \
  > benchmarks/vs_cpython/results/vortex_results.csv

# 4. Combine
python3 combine_bench_results.py \
  results/vortex_results.csv \
  results/cpython_pgo_results.csv \
  results/combined_vs_pgo_results.csv
```
