// =============================================================================
// vortex/support/telemetry.cpp — event ring + report (Rule 26).
// =============================================================================

#include "vortex/support/telemetry.hpp"

#include <cstdio>

namespace vortex {
inline namespace abi_v1 {

namespace {
const char* event_name(TelemetryEventKind k) noexcept {
    switch (k) {
        case TelemetryEventKind::TierDowngrade: return "tier_downgrade";
        case TelemetryEventKind::GuardFailed: return "guard_failed";
        case TelemetryEventKind::DeoptExecuted: return "deopt_executed";
        case TelemetryEventKind::WatchdogInvalidation: return "watchdog_invalidation";
        case TelemetryEventKind::RegallocSpillExcess: return "regalloc_spill_excess";
        case TelemetryEventKind::BudgetExceeded: return "budget_exceeded";
        case TelemetryEventKind::MegamorphicStubBuilt: return "megamorphic_stub_built";
        case TelemetryEventKind::ModuleVersionMiss: return "module_version_miss";
        case TelemetryEventKind::CodeCacheInvalidated: return "code_cache_invalidated";
        case TelemetryEventKind::CompileFailed: return "compile_failed";
        case TelemetryEventKind::EpochReclaimed: return "epoch_reclaimed";
        case TelemetryEventKind::SafepointPatched: return "safepoint_patched";
    }
    VORTEX_UNREACHABLE();
}
}  // namespace

void Telemetry::record(TelemetryEventKind kind, std::uint32_t d0, std::uint32_t d1,
                       std::uint64_t d2) noexcept {
    TelemetryEvent ev{kind, d0, d1, d2, ++clock_ns_};
    if (ring_size_ < ring_.size()) {
        ring_[(ring_head_ + ring_size_) % ring_.size()] = ev;
        ++ring_size_;
    } else {
        // Ring full: overwrite oldest, account the drop (Rule 26 — never silent).
        ring_[ring_head_] = ev;
        ring_head_ = (ring_head_ + 1) % ring_.size();
        bump(counter_dropped_events);
    }
}

void Telemetry::merge_from(const Telemetry& other) noexcept {
    for (const TelemetryEvent* e = other.events_begin(); e != other.events_end(); ++e) {
        record(e->kind, e->detail0, e->detail1, e->detail2);
    }
    for (std::uint32_t c = 0; c < counter_count; ++c) counters_[c] += other.counters_[c];
}

void Telemetry::write_report(std::FILE* out) const noexcept {
    std::fprintf(out, "== VORTEX telemetry ==\n");
    std::fprintf(out, "deopts=%llu guard_failures=%llu t1_compiles=%llu t2_compiles=%llu "
                      "t3_compiles=%llu watchdog_trips=%llu dropped_events=%llu\n",
                 static_cast<unsigned long long>(counters_[counter_total_deopts]),
                 static_cast<unsigned long long>(counters_[counter_guard_failures]),
                 static_cast<unsigned long long>(counters_[counter_tier1_compiles]),
                 static_cast<unsigned long long>(counters_[counter_tier2_compiles]),
                 static_cast<unsigned long long>(counters_[counter_tier3_compiles]),
                 static_cast<unsigned long long>(counters_[counter_watchdog_trips]),
                 static_cast<unsigned long long>(counters_[counter_dropped_events]));
    std::size_t shown = 0;
    for (std::size_t i = 0; i < ring_size_ && shown < 32; ++i, ++shown) {
        const TelemetryEvent& e = ring_[(ring_head_ + i) % ring_.size()];
        std::fprintf(out, "  [%llu] %s unit=%u pass=%u aux=%llu\n",
                     static_cast<unsigned long long>(e.timestamp_ns), event_name(e.kind),
                     e.detail0, e.detail1, static_cast<unsigned long long>(e.detail2));
    }
    if (ring_size_ > shown) {
        std::fprintf(out, "  ... %zu older events elided\n", ring_size_ - shown);
    }
}

}  // namespace abi_v1
}  // namespace vortex
