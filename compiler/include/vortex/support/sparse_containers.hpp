// =============================================================================
// vortex/support/sparse_containers.hpp — SparseSet & BitVector (Rule 18)
//
// Purpose:
//   Pass dataflow state (liveness, dominators, visited) never uses
//   std::set / std::unordered_set / std::vector<bool>. Small dense sets use
//   SparseSet (O(1) insert/contains/clear via two arrays); large sparse sets
//   use BitVector (one bit per element, word-strided ops).
//
// Invariants:
//   - SparseSet: universe fixed at construction; contains/insert O(1);
//     clear() O(size) not O(universe).
//   - BitVector: sized to universe; and/or/andnot/complement are word-wise;
//     set/clear/test O(1).
// =============================================================================

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "vortex/stdx/small_vector.hpp"

namespace vortex {

inline namespace abi_v1 {

class SparseSet {
public:
    explicit SparseSet(std::size_t universe)
        : dense_(universe), sparse_(universe, sentinel), member_count_(0) {}

    void insert(std::size_t v) noexcept {
        // TTC-12: bounds-check the universe before indexing sparse_/dense_.
        // Without this, v >= universe writes OOB into sparse_ and dense_ —
        // silent memory corruption that ASAN catches but the project's own
        // invariants don't.
        if (v >= sparse_.size()) [[unlikely]] { return; }
        if (contains(v)) return;
        sparse_[v] = static_cast<std::uint32_t>(member_count_);
        dense_[member_count_++] = static_cast<std::uint32_t>(v);
    }
    [[nodiscard]] bool contains(std::size_t v) const noexcept {
        return v < sparse_.size() && sparse_[v] != sentinel && sparse_[v] < member_count_ &&
               dense_[sparse_[v]] == v;
    }
    void erase(std::size_t v) noexcept {
        if (!contains(v)) return;
        std::uint32_t idx = sparse_[v];
        std::uint32_t last = dense_[--member_count_];
        dense_[idx] = last;
        sparse_[last] = idx;
        sparse_[v] = sentinel;
    }
    void clear() noexcept { member_count_ = 0; }

    [[nodiscard]] std::size_t size() const noexcept { return member_count_; }
    [[nodiscard]] bool empty() const noexcept { return member_count_ == 0; }
    [[nodiscard]] const std::uint32_t* begin() const noexcept { return dense_.data(); }
    [[nodiscard]] const std::uint32_t* end() const noexcept { return dense_.data() + member_count_; }

private:
    static constexpr std::uint32_t sentinel = 0xFFFF'FFFF;
    stdx::small_vector<std::uint32_t, 16> dense_;
    stdx::small_vector<std::uint32_t, 16> sparse_;
    std::uint32_t member_count_{0};
};

class BitVector {
public:
    BitVector() = default;
    explicit BitVector(std::size_t bits) : bits_(bits) { words_.assign(word_count(bits), 0); }

    void resize(std::size_t bits) noexcept {
        bits_ = bits;
        words_.resize(word_count(bits));
    }
    void set(std::size_t i) noexcept {
        VORTEX_ASSUME(i < bits_);
        words_[i >> 6] |= (std::uint64_t{1} << (i & 63));
    }
    void clear_bit(std::size_t i) noexcept {
        VORTEX_ASSUME(i < bits_);
        words_[i >> 6] &= ~(std::uint64_t{1} << (i & 63));
    }
    [[nodiscard]] bool test(std::size_t i) const noexcept {
        VORTEX_ASSUME(i < bits_);
        return (words_[i >> 6] & (std::uint64_t{1} << (i & 63))) != 0;
    }
    void clear_all() noexcept { std::memset(words_.data(), 0, words_.size() * 8); }

    [[nodiscard]] std::size_t size() const noexcept { return bits_; }

    void operator&=(const BitVector& o) noexcept {
        VORTEX_ASSUME(o.bits_ == bits_);
        for (std::size_t w = 0; w < words_.size(); ++w) words_[w] &= o.words_[w];
    }
    void operator|=(const BitVector& o) noexcept {
        VORTEX_ASSUME(o.bits_ == bits_);
        for (std::size_t w = 0; w < words_.size(); ++w) words_[w] |= o.words_[w];
    }
    void and_not(const BitVector& o) noexcept {
        VORTEX_ASSUME(o.bits_ == bits_);
        for (std::size_t w = 0; w < words_.size(); ++w) words_[w] &= ~o.words_[w];
    }
    /// Returns true if any bit changed (classic dataflow "changed" signal).
    bool union_with(const BitVector& o) noexcept {
        // TTC-11: bounds-check the operand. Two BitVectors with different
        // sizes have different word counts; indexing o.words_[w] past its
        // size is OOB. The static contract requires same size; we still
        // belt-and-braces the bound at runtime in case a pass creates
        // asymmetric universes (e.g. liveness vs. live-in sets).
        if (o.bits_ != bits_ || o.words_.size() != words_.size()) [[unlikely]] {
            const std::size_t shared = words_.size() < o.words_.size() ? words_.size() : o.words_.size();
            bool changed = false;
            for (std::size_t w = 0; w < shared; ++w) {
                std::uint64_t merged = words_[w] | o.words_[w];
                if (merged != words_[w]) changed = true;
                words_[w] = merged;
            }
            return changed;
        }
        bool changed = false;
        for (std::size_t w = 0; w < words_.size(); ++w) {
            std::uint64_t merged = words_[w] | o.words_[w];
            if (merged != words_[w]) changed = true;
            words_[w] = merged;
        }
        return changed;
    }
    [[nodiscard]] bool operator==(const BitVector& o) const noexcept {
        if (bits_ != o.bits_) return false;
        return std::memcmp(words_.data(), o.words_.data(), words_.size() * 8) == 0;
    }
    [[nodiscard]] bool any() const noexcept {
        for (std::uint64_t w : words_)
            if (w) return true;
        return false;
    }
    [[nodiscard]] std::size_t popcount() const noexcept {
        std::size_t total = 0;
        for (std::uint64_t w : words_) total += static_cast<std::size_t>(__builtin_popcountll(w));
        return total;
    }

private:
    [[nodiscard]] static std::size_t word_count(std::size_t bits) noexcept {
        return (bits + 63) / 64;
    }
    std::size_t bits_{0};
    stdx::small_vector<std::uint64_t, 8> words_;
};

}  // namespace abi_v1
}  // namespace vortex
