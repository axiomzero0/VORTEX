// =============================================================================
// vortex/support/symbol_table.cpp — interning implementation (Rule 16).
// =============================================================================

#include "vortex/support/symbol_table.hpp"

#include <cstring>

namespace vortex {
inline namespace abi_v1 {

namespace {
// FNV-1a 64: sufficient dispersion for identifier-sized keys; the table is
// open-addressing with quadratic probing (Rule 17: no node-per-entry heap).
constexpr std::uint64_t fnv_offset = 14695981039346656037ull;
constexpr std::uint64_t fnv_prime = 1099511628211ull;

std::uint64_t hash_bytes(std::string_view s) noexcept {
    std::uint64_t h = fnv_offset;
    for (char c : s) {
        h ^= static_cast<std::uint8_t>(c);
        h *= fnv_prime;
    }
    return h;
}
}  // namespace

SymbolTable::SymbolTable() {
    buckets_.assign(initial_buckets, 0);
    mask_ = initial_buckets - 1;
}

std::size_t SymbolTable::probe_index(std::string_view key, std::size_t mask) const noexcept {
    std::uint64_t h = hash_bytes(key);
    std::size_t idx = static_cast<std::size_t>(h) & mask;
    std::size_t step = 0;
    for (;;) {
        std::uint32_t entry = buckets_[idx];
        if (entry == 0) return idx;   // empty slot
        const Entry& e = entries_[entry - 1];
        std::string_view stored(reinterpret_cast<const char*>(bytes_.data()) + e.offset, e.length);
        if (stored == key) return idx;   // already interned
        // Quadratic probing.
        ++step;
        idx = (idx + step) & mask;
    }
}

SymbolId SymbolTable::intern(std::string_view text) noexcept {
    if (buckets_.empty()) [[unlikely]] {
        buckets_.assign(initial_buckets, 0);
        mask_ = initial_buckets - 1;
    }
    std::size_t idx = probe_index(text, mask_);
    if (buckets_[idx] != 0) return static_cast<SymbolId>(buckets_[idx] - 1);

    // Append bytes + entry.
    Entry e;
    e.offset = static_cast<std::uint32_t>(bytes_.size());
    e.length = static_cast<std::uint32_t>(text.size());
    for (char c : text) bytes_.push_back(c);
    bytes_.push_back('\0');
    entries_.push_back(e);
    SymbolId id = static_cast<SymbolId>(entries_.size() - 1);
    buckets_[idx] = id + 1;
    ++count_;

    if (count_ > static_cast<std::uint32_t>(buckets_.size() * max_load_factor)) [[unlikely]] {
        rehash();
    }
    return id;
}

bool SymbolTable::contains(std::string_view text) const noexcept {
    if (buckets_.empty()) return false;
    std::size_t idx = probe_index(text, mask_);
    return buckets_[idx] != 0;
}

std::string_view SymbolTable::text(SymbolId id) const noexcept {
    if (id >= entries_.size()) [[unlikely]] {
        return "<invalid-symbol>";
    }
    const Entry& e = entries_[id];
    return std::string_view(reinterpret_cast<const char*>(bytes_.data()) + e.offset, e.length);
}

void SymbolTable::rehash() noexcept {
    std::size_t new_bucket_count = buckets_.size() * 4;   // 4x: amortized cheap rehash
    buckets_.assign(new_bucket_count, 0);
    mask_ = new_bucket_count - 1;
    for (std::size_t entry_index = 0; entry_index < entries_.size(); ++entry_index) {
        const Entry& e = entries_[entry_index];
        std::string_view stored(reinterpret_cast<const char*>(bytes_.data()) + e.offset, e.length);
        std::size_t idx = probe_index(stored, mask_);
        buckets_[idx] = static_cast<std::uint32_t>(entry_index) + 1;
    }
}

SymbolTable& global_symbols() noexcept {
    static SymbolTable table;
    return table;
}

}  // namespace abi_v1
}  // namespace vortex
