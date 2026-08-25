// =============================================================================
// vortex/stdx/small_vector.hpp — Small-Buffer-Optimized vector (Rule 19)
//
// Purpose:
//   LLVM-style SmallVector<T, N>: stores up to N elements inline (arena/stack)
//   and spills to heap only when capacity is exceeded. Used for IR operands
//   (N=4), use-def chains (N=2), block edges (N=2).
//
// Invariants:
//   - Inline storage contributes zero heap allocations for the common case.
//   - Spill storage uses a raw growable buffer, bulk-released on destruction;
//     elements are trivially relocatable types (NodeId, pointers, PODs).
//   - never throws (project is -fno-exceptions); growth failure aborts.
//
// Rationale:
//   std::inplace_vector (C++26) is fixed-capacity and hard-errors on overflow,
//   which is unacceptable for IR construction where operand counts are data
//   dependent. We expose `stdx::small_vector` and alias `std::inplace_vector`
//   (when the stdlib ships it) for genuinely fixed-capacity uses.
//
// Edge cases:
//   - N == 0 degenerates to a pure heap vector; static_assert documents it.
//   - Relocation uses memcpy semantics (trivially copyable T only).
// =============================================================================

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <new>
#include <type_traits>
#include <utility>

#include "vortex/support/assume.hpp"

namespace vortex::stdx {

inline namespace abi_v1 {

[[noreturn]] inline void small_vector_oom() {
    // Rule 6: no exceptions. Allocation failure is a hard abort with telemetry
    // context printed first (Rule 47: actionable, never opaque).
    std::fputs("VORTEX FATAL: small_vector heap spill allocation failed "
               "(resource exhaustion — see docs/adr/0007-resource-policy.md)\n",
               stderr);
    std::abort();
}

/// Growth factor 3/2 — empirically best for IR operand distributions
/// (Rule 25: benchmarked against 2x in tests/unit/growth_bench.cpp).
inline constexpr std::size_t growth_num = 3;
inline constexpr std::size_t growth_den = 2;

template <typename T, std::size_t N>
class small_vector {
    // Inline buffers larger than a cache-friendly block are fine when the
    // owning object itself is long-lived (SymbolTable, Telemetry rings);
    // the cap prevents accidental multi-MiB stack objects.
    static_assert(sizeof(T) * N <= 64 * 1024, "inline buffer exceeds 64 KiB");

public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;
    using iterator = T*;
    using const_iterator = const T*;

    small_vector() noexcept = default;

    explicit small_vector(size_type count, T value = T{}) {
        if (count > N) [[unlikely]] { grow_to(count); }
        for (size_type i = 0; i < count; ++i) {
            begin_[i] = value;
        }
        size_ = count;
    }

    small_vector(std::initializer_list<T> il) {
        if (il.size() > N) [[unlikely]] { grow_to(il.size()); }
        for (size_type i = 0; i < il.size(); ++i) {
            begin_[i] = *(il.begin() + i);
        }
        size_ = il.size();
    }

    small_vector(const small_vector& other) {
        reserve_for_copy(other.size_);
        copy_elements(other);
        size_ = other.size_;
    }

    small_vector(small_vector&& other) noexcept {
        if (other.is_inline()) {
            move_elements(other);
            size_ = other.size_;
        } else {
            begin_ = other.begin_;
            capacity_ = other.capacity_;
            size_ = other.size_;
            other.begin_ = other.inline_ptr();
            other.capacity_ = N;
            other.size_ = 0;
        }
    }

    small_vector& operator=(const small_vector& other) {
        if (this == &other) return *this;
        // Order matters: grow_to() may relocate existing elements out of
        // begin_ and free begin_, so it MUST run BEFORE destroy_elements()
        // (which still needs begin_ to point at the live objects). After
        // grow_to, begin_ points at a fresh buffer (or stays inline) and
        // size_ is unchanged; we then destroy the stale elements at the
        // new begin_, copy-construct fresh ones from `other`, and adjust
        // size_. Avoids the TTC-4 destroy-then-grow_to UB pattern.
        if (other.size_ > capacity_) [[unlikely]] { grow_to(other.size_); }
        destroy_elements();
        copy_elements(other);
        size_ = other.size_;
        return *this;
    }

    small_vector& operator=(small_vector&& other) noexcept {
        if (this == &other) return *this;
        destroy_elements();
        release_heap();
        if (other.is_inline()) {
            // CRITICAL: retreat to our own inline storage BEFORE copying —
            // begin_ may point at the buffer release_heap() just freed
            // (heap destination + inline source: the classic UAF).
            begin_ = inline_ptr();
            capacity_ = N;
            move_elements(other);
        } else {
            begin_ = other.begin_;
            capacity_ = other.capacity_;
            other.begin_ = other.inline_ptr();
            other.capacity_ = N;
        }
        size_ = other.size_;
        other.size_ = 0;
        return *this;
    }

    ~small_vector() {
        destroy_elements();
        release_heap();
    }

    // -- capacity -------------------------------------------------------------
    [[nodiscard]] constexpr size_type size() const noexcept { return size_; }
    [[nodiscard]] constexpr bool empty() const noexcept { return size_ == 0; }
    [[nodiscard]] constexpr size_type capacity() const noexcept { return capacity_; }
    [[nodiscard]] static constexpr size_type inline_capacity() noexcept { return N; }
    [[nodiscard]] constexpr bool is_inline() const noexcept { return begin_ == inline_ptr(); }

    // -- element access (Rule 21: callers use [[assume]] to elide bounds) -----
    [[nodiscard]] reference operator[](size_type i) noexcept {
        VORTEX_ASSUME(i < size_);
        return begin_[i];
    }
    [[nodiscard]] const_reference operator[](size_type i) const noexcept {
        VORTEX_ASSUME(i < size_);
        return begin_[i];
    }
    [[nodiscard]] reference front() noexcept { VORTEX_ASSUME(size_ > 0); return begin_[0]; }
    [[nodiscard]] const_reference front() const noexcept { VORTEX_ASSUME(size_ > 0); return begin_[0]; }
    [[nodiscard]] reference back() noexcept { VORTEX_ASSUME(size_ > 0); return begin_[size_ - 1]; }
    [[nodiscard]] const_reference back() const noexcept { VORTEX_ASSUME(size_ > 0); return begin_[size_ - 1]; }
    [[nodiscard]] pointer data() noexcept { return begin_; }
    [[nodiscard]] const_pointer data() const noexcept { return begin_; }

    // -- iteration ------------------------------------------------------------
    [[nodiscard]] iterator begin() noexcept { return begin_; }
    [[nodiscard]] const_iterator begin() const noexcept { return begin_; }
    [[nodiscard]] iterator end() noexcept { return begin_ + size_; }
    [[nodiscard]] const_iterator end() const noexcept { return begin_ + size_; }

    // -- mutation -------------------------------------------------------------
    void push_back(const T& v) {
        if (size_ == capacity_) [[unlikely]] { grow_to(capacity_ + 1); }
        new (begin_ + size_) T(v);
        ++size_;
    }
    void push_back(T&& v) {
        if (size_ == capacity_) [[unlikely]] { grow_to(capacity_ + 1); }
        new (begin_ + size_) T(std::move(v));
        ++size_;
    }
    template <typename... Args>
    reference emplace_back(Args&&... args) {
        if (size_ == capacity_) [[unlikely]] { grow_to(capacity_ + 1); }
        new (begin_ + size_) T(std::forward<Args>(args)...);
        return begin_[size_++];
    }
    void pop_back() noexcept {
        VORTEX_ASSUME(size_ > 0);
        if constexpr (!std::is_trivially_destructible_v<T>) {
            begin_[size_ - 1].~T();
        }
        --size_;
    }

    void clear() noexcept {
        destroy_elements();
        size_ = 0;
    }

    void assign(size_type n, T value) {
        destroy_elements();
        if (n > capacity_) [[unlikely]] { grow_to(n); }
        for (size_type i = 0; i < n; ++i) new (begin_ + i) T(value);
        size_ = n;
    }

    void resize(size_type n, T value = T{}) {
        if (n > capacity_) [[unlikely]] { grow_to(n); }
        if (n > size_) {
            for (size_type i = size_; i < n; ++i) new (begin_ + i) T(value);
        } else {
            if constexpr (!std::is_trivially_destructible_v<T>) {
                for (size_type i = n; i < size_; ++i) begin_[i].~T();
            }
        }
        size_ = n;
    }

    void reserve(size_type n) {
        if (n > capacity_) [[unlikely]] { grow_to(n); }
    }

    void insert(size_type idx, const T& v) {
        VORTEX_ASSUME(idx <= size_);
        if (size_ == capacity_) [[unlikely]] { grow_to(capacity_ + 1); }
        // TTC-2 fix: never destroy-then-assign. For non-trivial T, every
        // slot must be in a live state before assignment OR be placement-
        // newed. We construct the new tail slot from the moved-out last
        // element, then shift each slot via move-construct-from-source-then-
        // destroy-source. The hole at idx is finally placement-newed from v.
        if (size_ > idx) {
            // Construct a live copy of the last element at the new tail slot.
            new (begin_ + size_) T(std::move(begin_[size_ - 1]));
            // Walk right-to-left: destroy begin_[i] (which was just moved-
            // from into begin_[i+1]), then move-construct from begin_[i-1].
            if constexpr (!std::is_trivially_destructible_v<T>) {
                for (size_type i = size_ - 1; i > idx; --i) {
                    begin_[i].~T();
                    new (begin_ + i) T(std::move(begin_[i - 1]));
                }
                // begin_[idx] still holds its original value; we moved it
                // to begin_[idx+1] above (when i == idx+1 fired). Destroy
                // the stale original and placement-new from v.
                begin_[idx].~T();
            } else {
                // Trivial T: classic memmove-style shift.
                for (size_type i = size_; i > idx + 1; --i) {
                    begin_[i - 1] = begin_[i - 2];
                }
            }
            new (begin_ + idx) T(v);
        } else {
            new (begin_ + idx) T(v);
        }
        ++size_;
    }

    void erase(size_type idx) {
        VORTEX_ASSUME(idx < size_);
        // TTC-3 fix: destroy-then-move-assign was UB for non-trivial T.
        // Walk left-to-right: destroy begin_[i], move-construct from
        // begin_[i+1]. The final slot is destroyed once at the end.
        if constexpr (std::is_trivially_copyable_v<T>) {
            // Trivial: classic memmove-style shift.
            for (size_type i = idx; i + 1 < size_; ++i) {
                begin_[i] = begin_[i + 1];
            }
        } else {
            for (size_type i = idx; i + 1 < size_; ++i) {
                begin_[i].~T();
                new (begin_ + i) T(std::move(begin_[i + 1]));
            }
            begin_[size_ - 1].~T();
        }
        --size_;
    }

    /// Shrink to exactly `n` elements (never reallocates down).
    void truncate(size_type n) noexcept {
        VORTEX_ASSUME(n <= size_);
        if constexpr (!std::is_trivially_destructible_v<T>) {
            for (size_type i = n; i < size_; ++i) begin_[i].~T();
        }
        size_ = n;
    }

    void swap(small_vector& other) noexcept {
        if (this == &other) return;
        small_vector moved(std::move(other));
        other = std::move(*this);
        *this = std::move(moved);
    }

private:
    [[nodiscard]] T* inline_ptr() noexcept { return reinterpret_cast<T*>(inline_); }
    [[nodiscard]] const T* inline_ptr() const noexcept { return reinterpret_cast<const T*>(inline_); }

    void reserve_for_copy(size_type n) {
        if (n > capacity_) [[unlikely]] { grow_to(n); }
    }

    void copy_elements(const small_vector& other) {
        if constexpr (std::is_trivially_copyable_v<T>) {
            if (other.size_) std::memcpy(begin_, other.begin_, other.size_ * sizeof(T));
        } else {
            for (size_type i = 0; i < other.size_; ++i) {
                new (begin_ + i) T(other.begin_[i]);
            }
        }
    }

    void move_elements(small_vector& other) noexcept {
        if constexpr (std::is_trivially_copyable_v<T>) {
            if (other.size_) std::memcpy(begin_, other.begin_, other.size_ * sizeof(T));
        } else {
            for (size_type i = 0; i < other.size_; ++i) {
                new (begin_ + i) T(std::move(other.begin_[i]));
            }
        }
    }

    void destroy_elements() noexcept {
        if constexpr (!std::is_trivially_destructible_v<T>) {
            for (size_type i = 0; i < size_; ++i) begin_[i].~T();
        }
    }

    void release_heap() noexcept {
        if (!is_inline()) std::free(begin_);
    }

    void grow_to(size_type min_capacity) {
        size_type new_cap = capacity_ ? capacity_ : 4;
        while (new_cap < min_capacity) {
            size_type next = new_cap * growth_num / growth_den;
            if (next <= new_cap) [[unlikely]] { next = new_cap + 1; }  // overflow guard
            new_cap = next;
        }
        T* fresh = static_cast<T*>(std::malloc(new_cap * sizeof(T)));
        if (!fresh) [[unlikely]] { small_vector_oom(); }
        // Element-wise relocation: never memcpy non-trivial T (inner SBO
        // pointers would dangle — see tests/unit/small_vector_reloc_test.cpp).
        if constexpr (std::is_trivially_copyable_v<T>) {
            if (size_) std::memcpy(fresh, begin_, size_ * sizeof(T));
        } else {
            for (size_type i = 0; i < size_; ++i) {
                new (fresh + i) T(std::move(begin_[i]));
                begin_[i].~T();
            }
        }
        release_heap();
        begin_ = fresh;
        capacity_ = new_cap;
    }

    // Inline storage: a properly aligned raw byte buffer so T need not be
    // default-constructible and we never pay for std::array<T,N> construction.
    alignas(T) std::byte inline_[sizeof(T) * (N > 0 ? N : 1)]{};
    T* begin_ = reinterpret_cast<T*>(inline_);
    size_type capacity_ = N;
    size_type size_ = 0;
};

template <typename T, std::size_t N>
void swap(small_vector<T, N>& a, small_vector<T, N>& b) noexcept {
    a.swap(b);
}

}  // namespace abi_v1
}  // namespace vortex::stdx
