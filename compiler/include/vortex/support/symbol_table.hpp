// =============================================================================
// vortex/support/symbol_table.hpp — Interned symbols (Rule 16)
//
// Purpose:
//   No std::string / std::string_view ever enters the IR or a pass. All
//   identifiers are interned once at the frontend into a global SymbolTable
//   and referenced by SymbolId (uint32_t) forever after.
//
// Invariants:
//   - SymbolId values are dense and stable for the process lifetime.
//   - Interning is idempotent: same bytes -> same SymbolId, forever.
//   - Lookup is a cache-friendly open-addressing table over stable storage;
//     no std::unordered_map (Rule 17).
//
// Rationale / Edge cases:
//   - Strings are stored out-of-line in an append-only byte arena so
//     interning never invalidates prior SymbolIds or views.
//   - The table is sized with quadratic probing and never shrinks.
// =============================================================================

#pragma once

#include <cstdint>
#include <cstring>
#include <string_view>

#include "vortex/stdx/small_vector.hpp"

namespace vortex {

inline namespace abi_v1 {

using SymbolId = std::uint32_t;
inline constexpr SymbolId invalid_symbol = 0xFFFF'FFFF;

class SymbolTable {
public:
    static constexpr std::size_t initial_buckets = 512;   // power of two
    static constexpr double max_load_factor = 0.7;

    SymbolTable();

    /// Intern `text`; returns a stable, dense SymbolId.
    SymbolId intern(std::string_view text) noexcept;

    /// Reverse lookup (diagnostics / IR printing only — never hot).
    [[nodiscard]] std::string_view text(SymbolId id) const noexcept;

    [[nodiscard]] std::uint32_t size() const noexcept { return count_; }
    [[nodiscard]] bool contains(std::string_view text) const noexcept;

private:
    struct Entry {
        std::uint32_t offset{0};
        std::uint32_t length{0};
    };

    void rehash() noexcept;
    [[nodiscard]] std::size_t probe_index(std::string_view key, std::size_t mask) const noexcept;

    // Append-only string bytes; SymbolId indexes into `entries_`.
    stdx::small_vector<char, 4096> bytes_;
    stdx::small_vector<Entry, 256> entries_;

    // Open-addressing bucket table storing entry_index+1 (0 = empty).
    stdx::small_vector<std::uint32_t, 512> buckets_;
    std::size_t mask_{0};
    std::uint32_t count_{0};
};

/// Process-wide symbol table. Initialized before any frontend runs; the
/// frontend is single-threaded by design (parser thread only), so no locking
/// is required — see docs/adr/0004-frontend-threading.md (Rule 13: compiler
/// threads only ever read frozen tables).
SymbolTable& global_symbols() noexcept;

}  // namespace abi_v1
}  // namespace vortex
