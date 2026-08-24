// =============================================================================
// vortex/support/telemetry.hpp — No silent fallbacks (Rule 26)
//
// Purpose:
//   Every JIT tier-down, guard failure, excessive spill, cache invalidation
//   and watchdog trip is recorded. Telemetry is a fixed-size, lock-free(1
//   writer) event ring + counters so recording never allocates on the hot
//   path. Post-run, tools dump a human-readable report.
//
// Invariants:
//   - Recording is O(1), allocation-free, exception-free.
//   - Counters are 64-bit and monotonically increasing.
//   - Ring overflow drops the OLDEST event and bumps `dropped_events`.
// =============================================================================

#pragma once

#include <cstdint>

#include "vortex/stdx/small_vector.hpp"

namespace vortex {

inline namespace abi_v1 {

enum class TelemetryEventKind : std::uint8_t {
    TierDowngrade,        // JIT fell back to a lower tier
    GuardFailed,          // speculative guard failed -> deopt
    DeoptExecuted,        // deopt completed with state reconstruction
    WatchdogInvalidation, // shape/type assumption invalidated
    RegallocSpillExcess,  // spill ratio above cfg::regalloc_max_spill_ratio_percent
    BudgetExceeded,       // pass budget guard tripped
    MegamorphicStubBuilt, // IC went megamorphic (dispatch thrashing hint)
    ModuleVersionMiss,    // LOAD_GLOBAL IC invalidated by module mutation
    CodeCacheInvalidated, // versioned cache invalidated (Rule 31)
    CompileFailed,        // compilation returned Error (fallback follows)
    EpochReclaimed,       // epoch GC freed retired code/IR
    SafepointPatched,     // tier entry pointer swapped
};

struct TelemetryEvent {
    TelemetryEventKind kind{};
    std::uint32_t detail0{};   // e.g. code unit id
    std::uint32_t detail1{};   // e.g. pass number
    std::uint64_t detail2{};   // e.g. sample count / spill ratio
    std::uint64_t timestamp_ns{};
};

class Telemetry {
public:
    static constexpr std::size_t event_ring_capacity = 1024;

    void record(TelemetryEventKind kind, std::uint32_t d0 = 0, std::uint32_t d1 = 0,
                std::uint64_t d2 = 0) noexcept;

    /// Aggregate counters (fast-path integer bumps).
    void bump(std::uint32_t counter_index, std::uint64_t by = 1) noexcept {
        if (counter_index < counter_count) counters_[counter_index] += by;
    }
    [[nodiscard]] std::uint64_t counter(std::uint32_t counter_index) const noexcept {
        return counter_index < counter_count ? counters_[counter_index] : 0;
    }

    // Canonical counter slots (indices, not raw numbers — Rule 23).
    static constexpr std::uint32_t counter_total_deopts = 0;
    static constexpr std::uint32_t counter_guard_failures = 1;
    static constexpr std::uint32_t counter_tier1_compiles = 2;
    static constexpr std::uint32_t counter_tier2_compiles = 3;
    static constexpr std::uint32_t counter_tier3_compiles = 4;
    static constexpr std::uint32_t counter_watchdog_trips = 5;
    static constexpr std::uint32_t counter_dropped_events = 6;
    static constexpr std::uint32_t counter_count = 7;

    [[nodiscard]] const TelemetryEvent* events_begin() const noexcept { return ring_.data(); }
    [[nodiscard]] const TelemetryEvent* events_end() const noexcept {
        return ring_.data() + ring_size_;
    }
    [[nodiscard]] std::size_t event_count() const noexcept { return ring_size_; }

    /// Merge another telemetry into this one (background compile threads).
    void merge_from(const Telemetry& other) noexcept;

    void write_report(std::FILE* out) const noexcept;

private:
    stdx::small_vector<TelemetryEvent, 64> ring_{event_ring_capacity};
    std::size_t ring_head_{0};
    std::size_t ring_size_{0};
    std::uint64_t counters_[counter_count]{};
    std::uint64_t clock_ns_{0};   // monotonic logical clock for ordering
};

}  // namespace abi_v1
}  // namespace vortex
