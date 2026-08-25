// =============================================================================
// vortex/support/epoch.hpp — Epoch-based reclamation (Rule 14)
//
// Purpose:
//   Retired JIT code blobs and IR graphs are tagged with the epoch in which
//   they died. Each mutator thread publishes the epoch it has advanced past.
//   Once every thread has advanced beyond E, everything that died in E is
//   bulk-freed. This avoids locks on the free path and use-after-free.
//
// Invariants:
//   - enter()/leave() are lock-free (a single atomic store each).
//   - retire(blob) is O(1) amortized: append to the current epoch's list.
//   - try_advance() is called at safepoints/yields only; it frees every epoch
//     strictly older than min(global_epoch, oldest thread epoch).
// =============================================================================

#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <limits>
#include <mutex>

#include "vortex/stdx/small_vector.hpp"
#include "vortex/support/config.hpp"

namespace vortex {

inline namespace abi_v1 {

/// Retirable payload: virtual-free function pointer + owning arena hint.
struct Retirable {
    void* allocation{};
    std::size_t size{};
    void (*destroy)(void* allocation, std::size_t size) noexcept;
};

class EpochGC {
public:
    static constexpr std::uint32_t max_threads = 64;

    EpochGC() {
        // Slots start parked: a thread only blocks reclamation between
        // enter() and leave() (Rule 13 discipline).
        for (auto& slot : thread_epochs_) {
            slot.store(std::numeric_limits<std::uint32_t>::max(), std::memory_order_relaxed);
        }
    }

    /// Called by a mutator when entering a region where retired memory must
    /// stay alive (i.e., publishing its progress).
    void enter(std::uint32_t thread_slot) noexcept {
        thread_epochs_[thread_slot].store(current_epoch_.load(std::memory_order_relaxed),
                                          std::memory_order_release);
    }

    /// Called when leaving the protected region (slot parked at "future").
    void leave(std::uint32_t thread_slot) noexcept {
        thread_epochs_[thread_slot].store(std::numeric_limits<std::uint32_t>::max(),
                                          std::memory_order_release);
    }

    void retire(Retirable blob) noexcept {
        std::lock_guard<std::mutex> guard(retired_lock_);
        retired_.push_back(RetiredEntry{blob, current_epoch_.load(std::memory_order_relaxed)});
    }

    /// Advance the epoch and reclaim everything all threads have passed.
    /// Called at safepoints (yield points, tier transitions).
    std::uint32_t try_advance() noexcept {
        std::uint32_t target = current_epoch_.load(std::memory_order_relaxed) + 1;
        std::uint32_t min_thread = std::numeric_limits<std::uint32_t>::max();
        for (std::uint32_t t = 0; t < max_threads; ++t) {
            std::uint32_t e = thread_epochs_[t].load(std::memory_order_acquire);
            if (e != std::numeric_limits<std::uint32_t>::max() && e < min_thread) min_thread = e;
        }
        // Only advance if every active thread has entered the *current* epoch
        // (min_thread == current). Otherwise stay put — Rule 14 safety.
        if (min_thread >= target - 1) {
            current_epoch_.store(target, std::memory_order_release);
            std::uint32_t grace = cfg::epoch_grace_epochs;
            std::uint32_t older_than = target > grace ? target - grace : 0;
            sweep(older_than);
            return target;
        }
        return current_epoch_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::uint32_t current_epoch() const noexcept {
        return current_epoch_.load(std::memory_order_relaxed);
    }

private:
    struct RetiredEntry {
        Retirable blob;
        std::uint32_t epoch;
    };

    void sweep(std::uint32_t older_than) noexcept {
        if (older_than == 0 || retired_.empty()) return;
        std::lock_guard<std::mutex> guard(retired_lock_);
        std::size_t write = 0;
        for (std::size_t read = 0; read < retired_.size(); ++read) {
            if (retired_[read].epoch < older_than) {
                retired_[read].blob.destroy(retired_[read].blob.allocation,
                                            retired_[read].blob.size);
            } else {
                if (write != read) retired_[write] = retired_[read];
                ++write;
            }
        }
        retired_.truncate(write);
    }

    std::atomic<std::uint32_t> current_epoch_{1};
    std::array<std::atomic<std::uint32_t>, max_threads> thread_epochs_{};
    std::mutex retired_lock_;
    stdx::small_vector<RetiredEntry, 32> retired_;
};

}  // namespace abi_v1
}  // namespace vortex
