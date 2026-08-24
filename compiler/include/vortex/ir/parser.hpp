// =============================================================================
// vortex/ir/parser.hpp — .vortex text IR parser (golden tests, Rule 35/52)
//
// Grammar (exact inverse of printer.cpp output):
//   file      := header node* frameStates?
//   header    := "fun" SYMBOL "params=" INT
//   node      := "n" INT "=" kind payload? "ins:" ("n" INT)*
//   payload   := per-kind tokens (constants, subop symbols, guards)
//   frameStates := "frame_states:" ("fs" INT "bcoff=" INT "unit=" INT
//                   "vals:" ("n" INT)* "kinds:" INT*)*
//
// The parser is diagnostics-rich (Rule 47): every syntax error carries the
// 1-based line and offending token.
// =============================================================================

#pragma once

#include <string_view>

#include "vortex/ir/graph.hpp"

namespace vortex::ir {

inline namespace abi_v1 {

/// Parse `text` into `g`. On failure returns a Diagnostic with line/column.
[[nodiscard]] Result<void> parse_graph(std::string_view text, Graph& g) noexcept;

}  // namespace abi_v1
}  // namespace vortex::ir
