// =============================================================================
// tests/regression/verifier_after_each_pass_regr.cpp — Rule 40 verifier guard.
//
// Rule 40: the structural verifier runs after EVERY pass in debug builds.
// It catches the silent IR corruption that would otherwise compound: a
// pass leaving dangling NodeIds, breaking the effect chain, or stripping
// FrameState from a speculative Guard.
//
// The hook (`passes::g_verify_after_each_pass`) is normally called from
// `passes::run_pipeline` when set. The regression installs a recording
// hook (see regression_harness.hpp's VerifierScope) that runs the verifier
// and appends every failure to a vector, then asserts the vector is empty
// after running the full pipeline on each program in the language corpus.
//
// A failure here means SOME pass between P03 and P51 left the IR in an
// invalid state — the test message names the pass and the verifier's
// complaint. That's the regression: not a runtime crash, but an
// invariant violation that would propagate to the backend and cause
// much harder-to-diagnose failures during codegen.
// =============================================================================

#include "regression_harness.hpp"

#include <cstdio>

#include "../lang/lang_cases.hpp"

using namespace vortex;
using namespace vortex::ir;
namespace passes = vortex::passes;
namespace rt = vortex::rt;

// =============================================================================
// Run the full pipeline over each lang_case under a recording verifier.
// No failures expected — every pass in the production pipeline must keep
// the IR in verifier-clean state.
// =============================================================================
TEST(regr_verifier_after_each_pass_full_pipeline_corpus) {
    int total_failures = 0;
    int programs_run = 0;
    for (std::size_t i = 0; i < kLangCaseCount; ++i) {
        const LangCase& c = kLangCases[i];
        bool ok = false;
        Graph g = vortex_test::lower_function(c.src, &ok);
        if (!ok) {
            // Some lang_cases are module-level only (no top-level def) —
            // skip those; they aren't function-shaped.
            continue;
        }
        ++programs_run;

        // The VerifierScope installs our recorder for the duration of
        // this scope; on destruction it restores the previous hook
        // (nullptr in production builds).
        vortex_test::VerifierScope vs;
        bool ok_run = vortex_test::run_full_pipeline(g, passes::TierMode::Tier2);
        if (!ok_run) {
            std::fprintf(stderr,
                         "  [verifier] pipeline raised a diagnostic on %s\n",
                         c.name);
            ++total_failures;
            continue;
        }
        if (!vs.empty()) {
            std::fprintf(stderr,
                         "  [verifier] %s: %zu post-pass verifier failures:\n",
                         c.name, vs.size());
            for (const std::string& msg : vs.fails.msgs) {
                std::fprintf(stderr, "    %s\n", msg.c_str());
            }
            total_failures += static_cast<int>(vs.size());
        }
    }
    // We must have run at least SOME programs (the corpus is diverse);
    // if zero ran, lower_function has a bug.
    CHECK(programs_run > 0);
    CHECK_EQ(total_failures, 0);
}

// =============================================================================
// Same as above, but explicitly on the polyhedral default-on path. The
// polyhedral pass rewires the IR more aggressively than any other pass —
// if the verifier is going to break, it breaks here. Run on the nested-
// while polyhedral candidate only (most lang_cases have no nested loops
// and the pass self-declines, which is correct).
// =============================================================================
TEST(regr_verifier_after_polyhedral_default_on) {
    static const char* kSrc =
        "def f(a):\n"
        "    total = 0\n"
        "    i = 0\n"
        "    while i < 3:\n"
        "        j = 0\n"
        "        while j < 3:\n"
        "            total = total + a[j]\n"
        "            j = j + 1\n"
        "        i = i + 1\n"
        "    return total\n";

    bool ok = false;
    Graph g = vortex_test::lower_function(kSrc, &ok);
    CHECK(ok);
    if (!ok) return;

    vortex_test::VerifierScope vs;
    // Default PassContext: polyhedral is ON by default, no opt-out flag set.
    passes::PassContext ctx;
    ctx.tier = passes::TierMode::Tier2;
    passes::OptPipeline pipeline;
    Result<void> r = passes::run_pipeline(g, ctx, pipeline);
    CHECK(r.has_value());
    if (!r) return;
    if (!vs.empty()) {
        std::fprintf(stderr,
                     "  [verifier] polyhedral default-on produced %zu failures:\n",
                     vs.size());
        for (const std::string& msg : vs.fails.msgs) {
            std::fprintf(stderr, "    %s\n", msg.c_str());
        }
    }
    CHECK(vs.empty());
}
