// =============================================================================
// tests/regression/determinism_regr.cpp — pipeline determinism guard.
//
// A compiler that produces different IR for the same input across two runs
// is broken — there's a non-deterministic walk somewhere (usually an
// iteration over a hash-based container whose order depends on insertion
// order, or on std::less<T*> whose order depends on the allocator). This
// is the kind of bug that breaks CI intermittently and is miserable to
// bisect.
//
// We catch it here by running the pipeline twice on the same input with
// two independent Vm/Graph stacks and asserting the fingerprints match
// byte-for-byte.
// =============================================================================

#include "regression_harness.hpp"

#include <cstdio>

#include "../lang/lang_cases.hpp"

using namespace vortex;
using namespace vortex::ir;
namespace passes = vortex::passes;

// =============================================================================
// Determinism over the full pipeline. Two fresh Graph instances lowered
// from the same source, two independent pipeline runs, two fingerprints.
// They MUST be byte-identical.
// =============================================================================
TEST(regr_pipeline_determinism_corpus) {
    int drift = 0;
    int ran = 0;
    for (std::size_t i = 0; i < kLangCaseCount; ++i) {
        const LangCase& c = kLangCases[i];
        bool ok1 = false, ok2 = false;
        Graph g1 = vortex_test::lower_function(c.src, &ok1);
        Graph g2 = vortex_test::lower_function(c.src, &ok2);
        if (!ok1 || !ok2) continue;   // skip non-function cases
        bool r1 = vortex_test::run_full_pipeline(g1, passes::TierMode::Tier2);
        bool r2 = vortex_test::run_full_pipeline(g2, passes::TierMode::Tier2);
        if (!r1 || !r2) {
            std::fprintf(stderr, "  [determinism] %s: pipeline raised a diagnostic\n",
                         c.name);
            ++drift;
            continue;
        }
        ++ran;
        std::string fp1 = vortex_test::fingerprint_ir(g1);
        std::string fp2 = vortex_test::fingerprint_ir(g2);
        if (fp1 != fp2) {
            std::fprintf(stderr,
                         "  [determinism] %s DRIFT:\n    run1: %s\n    run2: %s\n",
                         c.name, fp1.c_str(), fp2.c_str());
            ++drift;
        }
    }
    CHECK(ran > 0);
    CHECK_EQ(drift, 0);
}

// =============================================================================
// Determinism across tiers: the Tier 1 (budget-constrained) pipeline
// must produce a stable fingerprint too. A pass that uses a hash-ordered
// walk in its Tier 1 codepath but a stable order in Tier 2 would slip
// past the test above; this test pins the Tier 1 path independently.
// =============================================================================
TEST(regr_pipeline_determinism_tier1) {
    int drift = 0;
    int ran = 0;
    for (std::size_t i = 0; i < kLangCaseCount; ++i) {
        const LangCase& c = kLangCases[i];
        bool ok1 = false, ok2 = false;
        Graph g1 = vortex_test::lower_function(c.src, &ok1);
        Graph g2 = vortex_test::lower_function(c.src, &ok2);
        if (!ok1 || !ok2) continue;
        bool r1 = vortex_test::run_full_pipeline(g1, passes::TierMode::Tier1);
        bool r2 = vortex_test::run_full_pipeline(g2, passes::TierMode::Tier1);
        if (!r1 || !r2) {
            std::fprintf(stderr, "  [determinism-t1] %s: pipeline raised a diagnostic\n",
                         c.name);
            ++drift;
            continue;
        }
        ++ran;
        std::string fp1 = vortex_test::fingerprint_ir(g1);
        std::string fp2 = vortex_test::fingerprint_ir(g2);
        if (fp1 != fp2) {
            std::fprintf(stderr,
                         "  [determinism-t1] %s DRIFT:\n    run1: %s\n    run2: %s\n",
                         c.name, fp1.c_str(), fp2.c_str());
            ++drift;
        }
    }
    CHECK(ran > 0);
    CHECK_EQ(drift, 0);
}

// =============================================================================
// Determinism on polyhedral default-on path: the swap transformation must be
// deterministic — same input, same swap. A non-deterministic legality
// check (e.g. iterating a hash set of access sites) would produce
// different IR across runs.
// =============================================================================
TEST(regr_pipeline_determinism_polyhedral_default_on) {
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

    bool ok1 = false, ok2 = false;
    Graph g1 = vortex_test::lower_function(kSrc, &ok1);
    Graph g2 = vortex_test::lower_function(kSrc, &ok2);
    CHECK(ok1);
    CHECK(ok2);
    if (!ok1 || !ok2) return;

    // No opt-out flag set: polyhedral runs by default. The transformation
    // must be deterministic across runs.
    Flags<passes::OptOption> opts;

    bool r1 = vortex_test::run_full_pipeline(g1, passes::TierMode::Tier2, opts);
    bool r2 = vortex_test::run_full_pipeline(g2, passes::TierMode::Tier2, opts);
    CHECK(r1);
    CHECK(r2);
    if (!r1 || !r2) return;

    std::string fp1 = vortex_test::fingerprint_ir(g1);
    std::string fp2 = vortex_test::fingerprint_ir(g2);
    if (fp1 != fp2) {
        std::fprintf(stderr,
                     "  [determinism-poly] DRIFT:\n    run1: %s\n    run2: %s\n",
                     fp1.c_str(), fp2.c_str());
    }
    CHECK_EQ(fp1, fp2);
}
