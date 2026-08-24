// =============================================================================
// vortex/rt/driver.hpp — end-to-end compilation driver
//
// Source -> lex/parse -> Pass 1 lowering (all units, BFS through nested
// functions/classes) -> Tier-0 scheduling -> linked Program -> execution.
// This is the Tier-0 baseline path; higher tiers hook in through the
// tiering daemon (jit.hpp) observing CodeUnit counters.
// =============================================================================

#pragma once

#include "vortex/rt/interp.hpp"

namespace vortex::rt {

inline namespace abi_v1 {

struct CompileOutcome {
    bool ok{false};
    std::uint32_t units_compiled{0};
    std::uint32_t module_unit_id{0};   // id of the __module__ toplevel unit
    Diagnostic diagnostic{};
};

/// Compile `source` into `vm.program`. Diagnostics carry source locations.
[[nodiscard]] CompileOutcome compile_program(Vm& vm, std::string_view source,
                                             SymbolId module_name) noexcept;

/// Compile + run: the `vortex run` fast path.
[[nodiscard]] Result<Value> run_source(Vm& vm, std::string_view source) noexcept;

}  // namespace abi_v1
}  // namespace vortex::rt
