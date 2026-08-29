# ADR-0001: Introspection Architecture (Layers 0-3)

## Status
Accepted — 2026-08-29

## Context
VORTEX's LAWS.md mandates observability (Rules 26, 119, 123, 139, 140, 148)
but the runtime hot paths (deopt, jit bridge, CLI exit) used raw `fprintf` and
silent returns without recording telemetry events. The compiler pipeline had
no per-pass instrumentation. There was no causal linking between tier
transitions (Tier-0 → trace → deopt → Tier-0).

## Decision
Implement a four-layer observability architecture:

### Layer 0: Telemetry Wiring (Rule 26/119)
- `deopt.cpp`: Records `GuardFailed` + `DeoptExecuted` + `TierDowngrade`
- `jit.cpp`: Records `CompileFailed` on bridge error paths
- `vortex_main.cpp`: Dumps `telemetry.write_report(stderr)` on exit
- Deleted dead `g_pass_telemetry` global (Rule 23)

### Layer 1: Introspector (Rule 123)
- `support/introspect.hpp`: `PhaseSpan` (64-byte record) + `PhaseScope` (RAII)
- `g_phase_hook`: function-pointer seam (mirrors `g_verify_after_each_pass`)
- Wired into `run_pipeline()` — every pass gets a span with duration + node delta
- Zero overhead when off (single branch on `armed_`)
- No virtual dispatch (Rule 9), no heap allocation (Rule 7)

### Layer 2: CorrelationId (Rule 119)
- `Trace` struct carries `program_hash` + `header_pc`
- Links a deopt back to the trace that generated it
- Links the trace back to the Tier-0 loop that became hot

### Layer 3: Exporters (future — Rules 139, 140)
- `VORTEX_ENABLE_TELEMETRY_JSON`: NDJSON export (CI artifacts)
- `VORTEX_ENABLE_REPLAY_ARTIFACTS`: IR + PGO + seed bundles on failures (Rule 38/140)
- `VORTEX_ENABLE_TELEMETRY_PERFETTO`: chrome://tracing visualization
- All behind CMake flags (Rule 149: hermetic builds)

## Consequences
- Every fallback, deopt, and tier transition has a recorded telemetry trail
- Per-pass compilation can be profiled (duration, node-count delta, result)
- Zero overhead when no hook is set (single branch on `armed_`)
- No external dependencies (Rule 149) — pure C++26, no OTel SDK
- Future replay artifacts will enable "debug from replay, not reproduction"

## Non-decisions
- Not pulling OpenTelemetry SDK (Rule 149 forbids)
- Not using spdlog/glog (house style is fprintf + Diagnostic)
- Not instrumenting per-bytecode in Tier-0 (would tank 32.7x speedup)
- Not adding virtual Pass::trace() (Rule 9 forbids)
