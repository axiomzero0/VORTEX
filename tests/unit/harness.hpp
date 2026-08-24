// =============================================================================
// tests/unit/harness.hpp — self-registering test harness (Rule 52).
// =============================================================================

#pragma once

#include <cstdio>
#include <cstring>

namespace vortex_test {

struct TestCase {
    const char* name;
    void (*fn)();
    TestCase* next;
};

inline TestCase*& head() {
    static TestCase* h = nullptr;
    return h;
}

inline int& failures() {
    static int f = 0;
    return f;
}

struct Registrar {
    Registrar(const char* name, void (*fn)()) {
        TestCase* tc = new TestCase{name, fn, head()};
        head() = tc;
    }
};

}  // namespace vortex_test

#define TEST(name)                                                        \
    static void vortex_test_##name();                                     \
    static ::vortex_test::Registrar vortex_reg_##name(#name, vortex_test_##name); \
    static void vortex_test_##name()

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) [[unlikely]] {                                       \
            ++::vortex_test::failures();                                  \
            std::fprintf(stderr, "  CHECK failed: %s (%s:%d)\n", #cond,   \
                         __FILE__, __LINE__);                             \
        }                                                                 \
    } while (0)

#define CHECK_EQ(a, b)                                                    \
    do {                                                                  \
        auto _va = (a);                                                   \
        auto _vb = (b);                                                   \
        if (!(_va == _vb)) [[unlikely]] {                                 \
            ++::vortex_test::failures();                                  \
            std::fprintf(stderr, "  CHECK_EQ failed: %s == %s (%s:%d)\n", \
                         #a, #b, __FILE__, __LINE__);                     \
        }                                                                 \
    } while (0)
