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

/// Compilation switches the CALLER controls. Default-ON optimizations
/// live here as opt-OUT toggles — a default-constructed CompileOptions
/// runs the FULL pipeline (polyhedral included). The only legitimate
/// reason to set an opt-out flag here is compilation-time sensitivity
/// (the polyhedral analysis is linear in N for bounded D, but non-
/// trivial in absolute terms on hot-loop-heavy code).
struct CompileOptions {
    /// Pass 33 (polyhedral loop transforms). DEFAULT-ON. Maps to
    /// OptOption::DisablePolyhedral when set. The only legitimate
    /// reason to set this is compilation-time sensitivity.
    bool disable_polyhedral{false};
};

struct CompileOutcome {
    bool ok{false};
    std::uint32_t units_compiled{0};
    std::uint32_t module_unit_id{0};   // id of the __module__ toplevel unit
    Diagnostic diagnostic{};
};

/// Compile `source` into `vm.program`. Diagnostics carry source locations.
[[nodiscard]] CompileOutcome compile_program(Vm& vm, std::string_view source,
                                             SymbolId module_name,
                                             const CompileOptions& options = {}) noexcept;

/// Compile + run: the `vortex run` fast path.
[[nodiscard]] Result<Value> run_source(Vm& vm, std::string_view source,
                                       const CompileOptions& options = {}) noexcept;

}  // namespace abi_v1
}  // namespace vortex::rt
