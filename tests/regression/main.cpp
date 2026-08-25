// =============================================================================
// tests/regression/main.cpp — regression runner entry point.
//
// Identical structure to tests/unit/main.cpp on purpose: the harness is
// shared, so the per-test reporting (one line per test, [ ok ] / [FAIL])
// is uniform across both executables. The only difference is this binary
// runs the regression suite and reports it under the ctest NAME=regression.
// =============================================================================

#include "harness.hpp"

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // crash-safe: unbuffered
    std::size_t count = 0;
    int total_failures = 0;
    for (::vortex_test::TestCase* tc = ::vortex_test::head(); tc; tc = tc->next) {
        ++count;
        int before = ::vortex_test::failures();
        tc->fn();
        if (::vortex_test::failures() > before) {
            std::printf("[FAIL] %s\n", tc->name);
            total_failures += ::vortex_test::failures() - before;
        } else {
            std::printf("[ ok ] %s\n", tc->name);
        }
    }
    if (total_failures > 0) {
        std::printf("\n%d check(s) FAILED across %zu regression tests\n",
                    total_failures, count);
        return 1;
    }
    std::printf("\nall %zu regression tests passed\n", count);
    return 0;
}
