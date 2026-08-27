# VORTEX

Tiered optimizing compiler infrastructure for Python 3.16, written in C++26.

## Documentation

- [VORTEX Compiler Laws & Architecture Specification](docs/LAWS.md) the
  authoritative, uncompressed transcription of the laws that govern the VORTEX
  compiler. Every commit to `compiler/`, `runtime/`, `tools/`, and `tests/`
  must comply. CI verifies them. **There are no exceptions.**

## Project Layout

```
compiler/   — IR, passes (P03..P51), backend (lowering, regalloc, codegen)
runtime/    — Tier-0 VM, deopt, JIT bridge, driver
common/     — header-only shared vocabulary (Value, ids)
tools/      — vortex CLI
tests/      — unit + regression suites (self-registering, no test framework)
benchmarks/ — perf harness
docs/       — laws, ADRs, design notes
```

## Building

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j2
ctest --test-dir build
```

## Tiers

1. Tier 0: Direct-threaded register interpreter (computed goto, the
   ultimate fallback).
2. Tier 1: Baseline JIT (budget-constrained, linear-time passes only).
3. Tier 2: Optimizing JIT (PGO-driven, guard-emitting, full 51-pass
   pipeline).
4. Tier 3: AOT/static (proofs only, zero guards).

## Opt-out flags

Polyhedral loop interchange (Pass 33) is **on by default** in Tier 2/3.
Opt out via `CompileOptions::disable_polyhedral = true` when
compilation-time is more important than cache-locality wins.
