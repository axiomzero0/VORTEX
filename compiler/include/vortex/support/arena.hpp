// =============================================================================
// vortex/support/arena.hpp — Bump allocation for compiler IR (Rule 7)
//
// Purpose:
//   Zero malloc/free on the compiler hot path. All IR nodes, pass state, and
//   temporaries live in a monotonic bump arena (std::pmr::monotonic equivalent
//   with stronger guarantees) that is bulk-freed after compilation.
//
// Invariants:
//   - Allocation is pointer bump with alignment; O(1), branch-light.
//   - Individual objects are never freed — the arena is released whole
//     (Rule 14 handles retired JIT code via epochs, not per-object free).
//   - Growth doubles chunk size; chunks are kept in an intrusive list.
//
// Edge cases:
//   - Degenerate alignment requests clamp to alignof(std::max_align_t).
//   - Exhaustion aborts with an actionable diagnostic (Rule 47 + Rule 63).
// =============================================================================

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>

#include "vortex/support/diagnostic.hpp"

namespace vortex {

inline namespace abi_v1 {

class BumpArena {
public:
    static constexpr std::size_t first_chunk_bytes = 16 * 1024;   // 16 KiB
    static constexpr std::size_t max_chunk_bytes = 8 * 1024 * 1024;  // 8 MiB

    BumpArena() = default;
    explicit BumpArena(std::size_t initial_chunk_bytes) { new_chunk(initial_chunk_bytes); }
    ~BumpArena() { release(); }
    BumpArena(const BumpArena&) = delete;
    BumpArena& operator=(const BumpArena&) = delete;
    BumpArena(BumpArena&& other) noexcept { steal_from(other); }
    BumpArena& operator=(BumpArena&& other) noexcept {
        if (this == &other) return *this;
        release();
        steal_from(other);
        return *this;
    }

    void* allocate(std::size_t bytes, std::size_t align = alignof(std::max_align_t)) noexcept {
        if (align > alignof(std::max_align_t)) align = alignof(std::max_align_t);
        std::uintptr_t base = reinterpret_cast<std::uintptr_t>(cursor_);
        std::uintptr_t aligned = (base + align - 1) & ~(align - 1);
        std::size_t needed = aligned - base + bytes;
        if (needed > remaining_) [[unlikely]] {
            if (!grow_for(bytes, align)) [[unlikely]] {
                std::fprintf(stderr,
                             "VORTEX FATAL: BumpArena exhausted (requested %zu B). "
                             "Increase arena budget — docs/adr/0007-resource-policy.md\n",
                             bytes);
                std::abort();
            }
            base = reinterpret_cast<std::uintptr_t>(cursor_);
            aligned = (base + align - 1) & ~(align - 1);
            needed = aligned - base + bytes;
        }
        cursor_ = reinterpret_cast<std::byte*>(aligned) + bytes;
        remaining_ -= needed;
        total_allocated_ += bytes;
        return reinterpret_cast<void*>(aligned);
    }

    template <typename T, typename... Args>
    [[nodiscard]] T* create(Args&&... args) noexcept {
        void* mem = allocate(sizeof(T), alignof(T));
        return new (mem) T(std::forward<Args>(args)...);
    }

    template <typename T>
    [[nodiscard]] T* allocate_array(std::size_t n) noexcept {
        return static_cast<T*>(allocate(sizeof(T) * n, alignof(T)));
    }

    /// Mark point for rewinding transient pass state (not object-safe — only
    /// use when no live objects past the mark depend on later allocations).
    struct Mark {
        std::byte* cursor{};
        std::size_t remaining{};
    };
    [[nodiscard]] Mark mark() const noexcept { return Mark{cursor_, remaining_}; }
    void rewind(Mark m) noexcept {
        // Only valid if `m` is in the current chunk.
        cursor_ = m.cursor;
        remaining_ = m.remaining;
    }

    [[nodiscard]] std::size_t bytes_allocated() const noexcept { return total_allocated_; }

    void reset() noexcept {
        // Keep the largest chunk for reuse; bulk-free the rest (Rule 7).
        release_all_but_head();
        if (head_) {
            cursor_ = head_->data;
            remaining_ = head_->size;
        }
        total_allocated_ = 0;
    }

    void release() noexcept {
        Chunk* c = head_;
        while (c) {
            Chunk* next = c->next;
            std::free(c);
            c = next;
        }
        head_ = nullptr;
        cursor_ = nullptr;
        remaining_ = 0;
        total_allocated_ = 0;
    }

private:
    struct Chunk {
        Chunk* next{};
        std::size_t size{};
        std::byte* data{};   // points into this allocation after the header
    };

    void steal_from(BumpArena& other) noexcept {
        head_ = other.head_;
        cursor_ = other.cursor_;
        remaining_ = other.remaining_;
        total_allocated_ = other.total_allocated_;
        other.head_ = nullptr;
        other.cursor_ = nullptr;
        other.remaining_ = 0;
        other.total_allocated_ = 0;
    }

    bool grow_for(std::size_t bytes, std::size_t align) noexcept {
        std::size_t want = current_capacity_ * 2;
        if (want < bytes + align + 64) want = bytes + align + 64;
        if (want > max_chunk_bytes) want = max_chunk_bytes;
        if (want < bytes + align + 64) return false;   // single request > max chunk
        return new_chunk(want);
    }

    bool new_chunk(std::size_t bytes) noexcept {
        std::size_t total = sizeof(Chunk) + bytes;
        if (total < bytes) return false;   // overflow
        void* raw = std::malloc(total);
        if (!raw) return false;
        auto* c = static_cast<Chunk*>(raw);
        c->next = head_;
        c->size = bytes;
        c->data = reinterpret_cast<std::byte*>(c + 1);
        head_ = c;
        cursor_ = c->data;
        remaining_ = bytes;
        current_capacity_ = bytes;
        return true;
    }

    void release_all_but_head() noexcept {
        if (!head_) return;
        Chunk* c = head_->next;
        while (c) {
            Chunk* next = c->next;
            std::free(c);
            c = next;
        }
        head_->next = nullptr;
    }

    Chunk* head_{};
    std::byte* cursor_{};
    std::size_t remaining_{0};
    std::size_t current_capacity_{first_chunk_bytes};
    std::size_t total_allocated_{0};
};

}  // namespace abi_v1
}  // namespace vortex
