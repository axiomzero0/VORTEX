// =============================================================================
// vortex/stdx/flat_map.hpp — Cache-friendly sorted-vector map (Rule 17)
//
// Purpose:
//   std::unordered_map is banned in the compiler hot path (pointer chasing,
//   one heap node per entry). flat_map is backed by one contiguous sorted
//   array of pairs: lookups are branchless binary search over contiguous
//   memory; iteration is linear prefetch-friendly.
//
// Invariants:
//   - Keys are kept strictly sorted; duplicate insert is a no-op returning
//     the existing iterator.
//   - Amortized O(log n) find, O(n) insert (moves at most n pairs).
//     Build-once / query-many is the intended IR workload (GVN tables,
//     hash-cons caches, pass state).
//   - Probing is on a flat stdx::small_vector backing store so small maps
//     stay fully inline (no heap) — Rule 19.
//
// Rationale:
//   std::flat_map lands in C++26 but is not shipped by all vendors yet; this
//   implementation is API-compatible with the subset VORTEX uses and is
//   selected via feature detection in vortex/stdx/stdx.hpp.
//
// Edge cases:
//   - Empty map: begin()==end(), find() returns end().
//   - Heterogeneous lookup is deliberately NOT provided (Rule 16: the hot
//     path never uses strings — callers intern to SymbolId first).
// =============================================================================

#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>
#include <utility>

#include "vortex/stdx/small_vector.hpp"

namespace vortex::stdx {

inline namespace abi_v1 {

template <typename Key,
          typename Value,
          std::size_t InlineCapacity = 8,
          typename Compare = std::less<Key>>
class flat_map {
public:
    using key_type = Key;
    using mapped_type = Value;
    using value_type = std::pair<Key, Value>;
    using size_type = std::size_t;
    using iterator = typename small_vector<value_type, InlineCapacity>::iterator;
    using const_iterator = typename small_vector<value_type, InlineCapacity>::const_iterator;

    flat_map() = default;
    explicit flat_map(size_type reserve_hint) { entries_.reserve(reserve_hint); }

    // -- queries ---------------------------------------------------------------
    [[nodiscard]] const_iterator find(const Key& k) const noexcept {
        const auto* base = entries_.data();
        size_type n = entries_.size();
        while (n > 0) {   // branchless binary search loop; predictable for cache
            size_type half = n / 2;
            const auto& mid = base[half];
            if (Compare{}(mid.first, k)) {
                base += half + 1;
                n -= half + 1;
            } else {
                n = half;
            }
        }
        if (base != entries_.data() + entries_.size() && !Compare{}(k, base->first)) {
            return const_iterator(base);
        }
        return entries_.end();
    }

    [[nodiscard]] iterator find(const Key& k) noexcept {
        const auto* base = entries_.data();
        size_type n = entries_.size();
        while (n > 0) {
            size_type half = n / 2;
            const auto& mid = base[half];
            if (Compare{}(mid.first, k)) {
                base += half + 1;
                n -= half + 1;
            } else {
                n = half;
            }
        }
        if (base != entries_.data() + entries_.size() && !Compare{}(k, base->first)) {
            return iterator(base);
        }
        return entries_.end();
    }

    [[nodiscard]] bool contains(const Key& k) const noexcept { return find(k) != entries_.end(); }

    [[nodiscard]] Value* get(const Key& k) noexcept {
        auto it = find(k);
        return it == entries_.end() ? nullptr : &it->second;
    }
    [[nodiscard]] const Value* get(const Key& k) const noexcept {
        auto it = find(k);
        return it == entries_.end() ? nullptr : &it->second;
    }

    [[nodiscard]] size_type size() const noexcept { return entries_.size(); }
    [[nodiscard]] bool empty() const noexcept { return entries_.empty(); }

    // -- insertion ---------------------------------------------------------------
    /// Returns {iterator, inserted}.
    std::pair<iterator, bool> insert(const Key& k, const Value& v) {
        auto it = lower_bound(k);
        if (it != entries_.end() && !Compare{}(k, it->first)) {
            return {it, false};   // duplicate: first insertion wins (GVN semantics)
        }
        size_type idx = static_cast<size_type>(it - entries_.begin());
        entries_.reserve(entries_.size() + 1);
        entries_.insert(idx, value_type(k, v));
        return {entries_.begin() + idx, true};
    }

    std::pair<iterator, bool> insert(const Key& k, Value&& v) {
        auto it = lower_bound(k);
        if (it != entries_.end() && !Compare{}(k, it->first)) {
            return {it, false};
        }
        size_type idx = static_cast<size_type>(it - entries_.begin());
        entries_.reserve(entries_.size() + 1);
        entries_.insert(idx, value_type(k, std::move(v)));
        return {entries_.begin() + idx, true};
    }

    Value& operator[](const Key& k) {
        auto it = lower_bound(k);
        if (it != entries_.end() && !Compare{}(k, it->first)) return it->second;
        size_type idx = static_cast<size_type>(it - entries_.begin());
        entries_.reserve(entries_.size() + 1);
        entries_.insert(idx, value_type(k, Value{}));
        return (entries_.begin() + idx)->second;
    }

    /// Insert-or-overwrite (map assignment semantics).
    Value& insert_or_assign(const Key& k, Value v) {
        auto [it, inserted] = insert(k, std::move(v));
        if (!inserted) it->second = std::move(v);
        return it->second;
    }

    size_type erase(const Key& k) {
        auto it = find(k);
        if (it == entries_.end()) return 0;
        size_type idx = static_cast<size_type>(it - entries_.begin());
        entries_.erase(idx);
        return 1;
    }

    void insert_if_absent(const Key& k, const Value& v) noexcept {
        if (!contains(k)) insert(k, v);
    }

    void clear() noexcept { entries_.clear(); }
    void reserve(size_type n) { entries_.reserve(n); }

    [[nodiscard]] iterator begin() noexcept { return entries_.begin(); }
    [[nodiscard]] iterator end() noexcept { return entries_.end(); }
    [[nodiscard]] const_iterator begin() const noexcept { return entries_.begin(); }
    [[nodiscard]] const_iterator end() const noexcept { return entries_.end(); }

private:
    [[nodiscard]] iterator lower_bound(const Key& k) noexcept {
        size_type lo = 0, hi = entries_.size();
        while (lo < hi) {
            size_type mid = lo + (hi - lo) / 2;
            if (Compare{}(entries_[mid].first, k)) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return entries_.begin() + lo;
    }

    small_vector<value_type, InlineCapacity> entries_;
};

}  // namespace abi_v1
}  // namespace vortex::stdx
