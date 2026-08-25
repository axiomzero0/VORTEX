// =============================================================================
// tests/regression/pass_budget_guards_regr.cpp — Rule 26 budget guard.
//
// Rule 26: every pass respects PassContext::node_budget. A pass that would
// blow the budget must (a) NOT run / (b) emit a TelemetryEventKind::BudgetExceeded
// event so the tiering daemon sees it and (c) leave the IR in a valid
// state.
//
// The bug this guards: a future pass that grows the IR (unrolling,
// specialization, inlining) without checking the budget — under load it
// would silently expand the IR until the OOM killer fires, or worse,
// the JIT would spend 50 ms on a function that should have been <1 ms.
// =============================================================================

#include "regression_harness.hpp"

#include <cstdio>

using namespace vortex;
using namespace vortex::ir;
namespace passes = vortex::passes;
namespace rt = vortex::rt;

namespace {

// Count BudgetExceeded events in the telemetry ring. We don't override
// record() (it's non-virtual by design — Rule 9 forbids virtual dispatch
// in the hot path); instead, we walk the event ring after the pipeline
// finishes. The event ring is bounded (1024 events) — if the pipeline
// records more than 1024 budget events, older ones are overwritten; we
// treat that as "many budget events" (still > 0), so the assertion holds.
[[nodiscard]] std::uint32_t count_budget_events(const Telemetry& tele) noexcept {
    std::uint32_t n = 0;
    for (const TelemetryEvent* it = tele.events_begin();
         it != tele.events_end(); ++it) {
        if (it->kind == TelemetryEventKind::BudgetExceeded) ++n;
    }
    return n;
}

}  // namespace

// =============================================================================
// Run the pipeline with a node_budget = 1. The pipeline's budget guard
// (`g.live_node_count() > ctx.node_budget`) MUST fire on every pass after
// the first that would exceed it. We don't crash, we don't produce bad
// IR — we just stop and emit telemetry.
//
// The IR we lower has >1 node (every real program does), so the budget
// fires immediately on the first pass.
// =============================================================================
TEST(regr_pass_budget_guard_fires_on_tight_budget) {
    bool ok = false;
    Graph g = vortex_test::lower_function(
        "def f(x):\n    return x + 1\n", &ok);
    CHECK(ok);
    if (!ok) return;

    Telemetry tele;
    passes::PassContext ctx;
    ctx.tier = passes::TierMode::Tier2;
    ctx.node_budget = 1;   // impossibly tight
    ctx.telemetry = &tele;
    passes::OptPipeline pipeline;
    Result<void> r = passes::run_pipeline(g, ctx, pipeline);

    // The pipeline doesn't fail on budget — it just stops early. The
    // telemetry records at least one BudgetExceeded event.
    CHECK(r.has_value());
    CHECK(count_budget_events(tele) > 0);
}

// =============================================================================
// Run the pipeline with a reasonable budget on a real program. NO
// BudgetExceeded event should fire. A spurious telemetry event here
// would mean the budget check is wrong (off-by-one, etc.) and would
// mask real budget breaches.
// =============================================================================
TEST(regr_pass_budget_guard_no_false_alarm_on_reasonable_budget) {
    bool ok = false;
    Graph g = vortex_test::lower_function(
        "def f(items):\n"
        "    total = 0\n"
        "    for x in items:\n"
        "        total = total + x\n"
        "    return total\n", &ok);
    CHECK(ok);
    if (!ok) return;

    Telemetry tele;
    passes::PassContext ctx;
    ctx.tier = passes::TierMode::Tier2;
    ctx.node_budget = 1'000'000;   // plenty
    ctx.telemetry = &tele;
    passes::OptPipeline pipeline;
    Result<void> r = passes::run_pipeline(g, ctx, pipeline);

    CHECK(r.has_value());
    CHECK_EQ(count_budget_events(tele), 0u);
}

// =============================================================================
// Budget = 0 (the pathological zero): the pipeline must still produce
// a valid IR — it must run NO passes, but the verifier must be clean
// on the un-optimized IR.
// =============================================================================
TEST(regr_pass_budget_guard_zero_budget_leaves_valid_ir) {
    bool ok = false;
    Graph g = vortex_test::lower_function(
        "def f(x):\n    return x + 1\n", &ok);
    CHECK(ok);
    if (!ok) return;

    Telemetry tele;
    passes::PassContext ctx;
    ctx.tier = passes::TierMode::Tier2;
    ctx.node_budget = 0;
    ctx.telemetry = &tele;
    passes::OptPipeline pipeline;
    Result<void> r = passes::run_pipeline(g, ctx, pipeline);

    CHECK(r.has_value());
    // The first pass attempts to run; the budget guard fires BEFORE
    // any work; the IR is unchanged. Verify it.
    stdx::small_vector<Diagnostic, 4> diags = verify_graph(g);
    bool verifier_clean = true;
    for (const Diagnostic& d : diags) {
        if (d.is_error()) verifier_clean = false;
    }
    CHECK(verifier_clean);
}
