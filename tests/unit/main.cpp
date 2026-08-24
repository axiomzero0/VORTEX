// =============================================================================
// tests/unit/main.cpp — runs all self-registered tests (Rule 52).
// =============================================================================

#include "harness.hpp"

int main() {
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
        std::printf("\n%d check(s) FAILED across %zu tests\n", total_failures, count);
        return 1;
    }
    std::printf("\nall %zu tests passed\n", count);
    return 0;
}
