// =============================================================================
// tests/unit/support_test.cpp — support layer unit tests (Rules 52, 60).
// =============================================================================

#include "harness.hpp"
#include "vortex/stdx/stdx.hpp"
#include "vortex/support/arena.hpp"
#include "vortex/support/diagnostic.hpp"
#include "vortex/support/epoch.hpp"
#include "vortex/support/flags.hpp"
#include "vortex/support/result.hpp"
#include "vortex/support/sparse_containers.hpp"
#include "vortex/support/symbol_table.hpp"
#include "vortex/support/telemetry.hpp"

#include <cstring>

using namespace vortex;
using vortex::stdx::small_vector;

TEST(small_vector_inline_no_alloc) {
    small_vector<int, 4> v;
    for (int i = 0; i < 4; ++i) v.push_back(i * 10);
    CHECK(v.is_inline());
    CHECK_EQ(v.size(), 4u);
    CHECK_EQ(v[3], 30);
    v.push_back(40);   // spill
    CHECK(!v.is_inline());
    CHECK_EQ(v.size(), 5u);
    CHECK_EQ(v[4], 40);
    CHECK_EQ(v.front(), 0);
    CHECK_EQ(v.back(), 40);
}

TEST(small_vector_relocation_correctness) {
    // Nested SBO: vectors of a type containing vectors must survive growth
    // (the bug class that killed naive memcpy relocation).
    struct Boxed {
        small_vector<int, 2> inner;
    };
    small_vector<Boxed, 2> outer;
    for (int i = 0; i < 32; ++i) {
        Boxed b;
        b.inner.push_back(i);
        b.inner.push_back(i * 2);
        b.inner.push_back(i * 3);   // force inner spill for some
        outer.push_back(b);
    }
    CHECK(outer.size() == 32);
    for (int i = 0; i < 32; ++i) {
        CHECK_EQ(outer[static_cast<std::size_t>(i)].inner.size(), 3u);
        CHECK_EQ(outer[static_cast<std::size_t>(i)].inner[0], i);
        CHECK_EQ(outer[static_cast<std::size_t>(i)].inner[1], i * 2);
        CHECK_EQ(outer[static_cast<std::size_t>(i)].inner[2], i * 3);
    }
}

TEST(small_vector_copy_move) {
    small_vector<int, 2> a;
    a.push_back(1);
    a.push_back(2);
    a.push_back(3);
    small_vector<int, 2> b = a;
    CHECK_EQ(b.size(), 3u);
    CHECK_EQ(b[2], 3);
    small_vector<int, 2> c = std::move(b);
    CHECK_EQ(c.size(), 3u);
    CHECK_EQ(c[0], 1);
    a.erase(0);
    CHECK_EQ(a.size(), 2u);
    CHECK_EQ(a[0], 2);
    a.insert(0, 99);
    CHECK_EQ(a[0], 99);
    CHECK_EQ(a[1], 2);
}

TEST(flat_map_basic) {
    stdx::flat_map<int, const char*, 4> m;
    m.insert(10, "ten");
    m.insert(5, "five");
    m.insert(20, "twenty");
    CHECK(m.contains(5) && m.contains(10) && m.contains(20));
    CHECK(!m.contains(15));
    CHECK(std::strcmp(*m.get(10), "ten") == 0);
    // duplicate insert keeps first
    m.insert(10, "TEN");
    CHECK(std::strcmp(*m.get(10), "ten") == 0);
    // ordered iteration
    int prev = -1;
    for (auto& kv : m) {
        CHECK(kv.first > prev);
        prev = kv.first;
    }
    m.erase(10);
    CHECK(!m.contains(10));
    m.insert_or_assign(5, "FIVE");
    CHECK(std::strcmp(*m.get(5), "FIVE") == 0);
}

enum class F : unsigned { A = 1, B = 2, C = 4 };

TEST(flags_type_safety) {
    Flags<F> f = F::A | F::B;
    CHECK(f.has(F::A));
    CHECK(f.has(F::B));
    CHECK(!f.has(F::C));
    CHECK(f.has_any(F::C | F::B));
    CHECK(f.has_all(F::A | F::B));
    CHECK(!f.has_all(F::A | F::C));
    f.clear(F::A);
    CHECK(!f.has(F::A));
    CHECK_EQ(f.popcount(), 1u);
    CHECK_EQ(f.at(0), F::B);
}

TEST(sparse_set_and_bitvector) {
    SparseSet s(64);
    s.insert(10);
    s.insert(20);
    s.insert(30);
    CHECK(s.contains(10) && s.contains(30));
    CHECK_EQ(s.size(), 3u);
    s.erase(20);
    CHECK_EQ(s.size(), 2u);
    s.insert(7);
    CHECK(s.contains(7) && s.contains(30));
    s.clear();
    CHECK(s.empty());

    BitVector b(130);
    b.set(0);
    b.set(64);
    b.set(129);
    CHECK(b.test(0) && b.test(64) && b.test(129));
    CHECK(!b.test(1));
    CHECK_EQ(b.popcount(), 3u);
    BitVector o(130);
    o.set(1);
    CHECK(b.union_with(o));
    CHECK_EQ(b.popcount(), 4u);
    CHECK(!b.union_with(o));
}

TEST(symbol_table_interning) {
    SymbolTable& t = global_symbols();
    SymbolId a = t.intern("hello_world_test_sym");
    SymbolId b = t.intern("hello_world_test_sym");
    SymbolId c = t.intern("another_test_sym_2");
    CHECK(a == b);
    CHECK(a != c);
    CHECK(t.text(a) == "hello_world_test_sym");
    // many interns force a rehash
    char buf[32];
    for (int i = 0; i < 2000; ++i) {
        std::snprintf(buf, sizeof(buf), "sym_%d", i);
        SymbolId id = t.intern(buf);
        CHECK(t.text(id) == std::string_view(buf));
    }
}

TEST(result_and_try_macro) {
    auto fallible = [](int v) -> Result<int> {
        if (v < 0) return fail_msg("negative", 42);
        return v * 2;
    };
    auto outer = [&](int v) -> Result<int> {
        int doubled = VORTEX_TRY(fallible(v));
        return doubled + 1;
    };
    Result<int> ok = outer(5);
    CHECK(ok && *ok == 11);
    Result<int> bad = outer(-1);
    CHECK(!bad);
    CHECK_EQ(bad.error().code, 42u);

    auto void_outer = [&](int v) -> Result<void> {
        VORTEX_TRY_VOID(fallible(v));
        return {};
    };
    CHECK(void_outer(1).has_value());
    CHECK(!void_outer(-1).has_value());
}

TEST(arena_bump_and_reset) {
    BumpArena arena;
    int* a = arena.create<int>(7);
    double* b = arena.create<double>(3.5);
    CHECK_EQ(*a, 7);
    CHECK_EQ(*b, 3.5);
    for (int i = 0; i < 10000; ++i) {
        auto* p = arena.create<std::uint64_t>(i);
        CHECK_EQ(*p, static_cast<std::uint64_t>(i));
    }
    CHECK(arena.bytes_allocated() > 10000 * 8);
    arena.reset();
    int* c = arena.create<int>(9);
    CHECK_EQ(*c, 9);
}

TEST(telemetry_records_events) {
    Telemetry tel;
    tel.record(TelemetryEventKind::GuardFailed, 7, 49, 123);
    tel.bump(Telemetry::counter_guard_failures);
    tel.bump(Telemetry::counter_guard_failures);
    CHECK_EQ(tel.counter(Telemetry::counter_guard_failures), 2u);
    CHECK_EQ(tel.event_count(), 1u);
    Telemetry merged;
    merged.record(TelemetryEventKind::DeoptExecuted);
    merged.bump(Telemetry::counter_total_deopts, 5);
    tel.merge_from(merged);
    CHECK_EQ(tel.counter(Telemetry::counter_total_deopts), 5u);
    CHECK_EQ(tel.event_count(), 2u);
}

TEST(epoch_gc_frees_after_advance) {
    static int destroyed = 0;
    destroyed = 0;
    EpochGC gc;
    gc.enter(0);
    static int payload = 42;
    auto destroy = [](void* p, std::size_t) noexcept {
        (void)p;
        ++destroyed;
    };
    gc.retire({&payload, sizeof(int), destroy});
    gc.leave(0);
    gc.enter(0);          // thread advanced into current epoch
    gc.try_advance();     // retire happened in epoch 1; grace 3
    CHECK_EQ(destroyed, 0);   // still within grace
    gc.leave(0);
    gc.enter(0);
    gc.try_advance();
    gc.leave(0);
    gc.enter(0);
    gc.try_advance();
    gc.leave(0);
    gc.enter(0);
    gc.try_advance();
    gc.leave(0);
    CHECK_EQ(destroyed, 1);
}

TEST(diagnostic_actionable_fields) {
    Diagnostic d = Diagnostic::error("type mismatch in node", diag_code::type_mismatch);
    d.where.line = 10;
    d.expected = "int64";
    d.actual = "str";
    d.fix = "insert explicit conversion";
    CHECK(d.is_error());
    CHECK_EQ(d.code, diag_code::type_mismatch);
    // Rule 47: reporting must not crash with empty fields.
    Diagnostic::note("bare note").report(stderr);
}
