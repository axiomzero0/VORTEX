// =============================================================================
// vortex/support/diagnostic.cpp — Rule 47 reporting implementation.
// =============================================================================

#include "vortex/support/assume.hpp"
#include "vortex/support/diagnostic.hpp"

#include <cstdio>

namespace vortex {
inline namespace abi_v1 {

namespace {

// Clamp-and-terminate copy for fixed-capacity diagnostic text (never heap).
void emit_field(std::FILE* out, const char* label, std::string_view value) noexcept {
    if (value.empty()) return;
    if (value.size() > diagnostic_text_capacity) {
        std::fprintf(out, "  %-9s %.*s~\n", label,
                     static_cast<int>(diagnostic_text_capacity), value.data());
    } else {
        std::fprintf(out, "  %-9s %.*s\n", label, static_cast<int>(value.size()), value.data());
    }
}

const char* severity_name(Severity s) noexcept {
    switch (s) {
        case Severity::Note: return "note";
        case Severity::Warning: return "warning";
        case Severity::Error: return "error";
        case Severity::Fatal: return "FATAL";
    }
    VORTEX_UNREACHABLE();
}

}  // namespace

void Diagnostic::report(std::FILE* out) const noexcept {
    if (where.file.empty()) {
        std::fprintf(out, "VORTEX %s [code %u]: %.*s\n", severity_name(severity), code,
                     static_cast<int>(message.size()), message.data());
    } else {
        std::fprintf(out, "VORTEX %s [code %u] %.*s:%u:%u: %.*s\n", severity_name(severity), code,
                     static_cast<int>(where.file.size()), where.file.data(), where.line,
                     where.column, static_cast<int>(message.size()), message.data());
    }
    if (!expected.empty() || !actual.empty()) {
        emit_field(out, "expected:", expected);
        emit_field(out, "actual:", actual);
    }
    emit_field(out, "fix:", fix);
}

}  // namespace abi_v1
}  // namespace vortex
