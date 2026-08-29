// =============================================================================
// vortex/ir/printer.cpp — canonical text form implementation.
// =============================================================================

#include "vortex/ir/printer.hpp"

#include <cstring>

#include "vortex/support/symbol_table.hpp"

namespace vortex::ir {
inline namespace abi_v1 {

namespace {

const char* binop_name(std::uint16_t subop) noexcept {
    switch (static_cast<BinOpKind>(subop)) {
        case BinOpKind::Add: return "+";
        case BinOpKind::Sub: return "-";
        case BinOpKind::Mul: return "*";
        case BinOpKind::TrueDiv: return "/";
        case BinOpKind::FloorDiv: return "//";
        case BinOpKind::Mod: return "%";
        case BinOpKind::Pow: return "**";
        case BinOpKind::LShift: return "<<";
        case BinOpKind::RShift: return ">>";
        case BinOpKind::BitAnd: return "&";
        case BinOpKind::BitOr: return "|";
        case BinOpKind::BitXor: return "^";
        case BinOpKind::MatMul: return "@";
    }
    VORTEX_UNREACHABLE();
}

const char* cmpop_name(std::uint16_t subop) noexcept {
    switch (static_cast<CmpOpKind>(subop)) {
        case CmpOpKind::LT: return "<";
        case CmpOpKind::LE: return "<=";
        case CmpOpKind::GT: return ">";
        case CmpOpKind::GE: return ">=";
        case CmpOpKind::EQ: return "==";
        case CmpOpKind::NE: return "!=";
        case CmpOpKind::Is: return "is";
        case CmpOpKind::IsNot: return "is-not";
        case CmpOpKind::In: return "in";
        case CmpOpKind::NotIn: return "not-in";
    }
    VORTEX_UNREACHABLE();
}

const char* guard_name(std::uint16_t subop) noexcept {
    switch (static_cast<GuardKind>(subop)) {
        case GuardKind::TypeIs: return "type_is";
        case GuardKind::ShapeIs: return "shape_is";
        case GuardKind::IntFits: return "int_fits";
        case GuardKind::NotNone: return "not_none";
        case GuardKind::Bounds: return "bounds";
        case GuardKind::NoOverflow: return "no_overflow";
        case GuardKind::AliasDisjoint: return "alias_disjoint";
        case GuardKind::ModuleVersion: return "module_version";
        case GuardKind::MonomorphicCall: return "mono_call";
    }
    VORTEX_UNREACHABLE();
}

void print_node_line(const Graph& g, NodeId id, std::FILE* out) noexcept {
    const Node& n = g.node(id);
    std::fprintf(out, "n%u = %s", id, node_kind_name(n.kind));

    // Payloads per kind.
    switch (n.kind) {
        case NodeKind::ConstInt:
            std::fprintf(out, " %lld", static_cast<long long>(n.const_value.as.i));
            break;
        case NodeKind::ConstFloat:
            std::fprintf(out, " %.17g", n.const_value.as.f);
            break;
        case NodeKind::ConstPy:
            std::fprintf(out, " tag=%d i=%lld", static_cast<int>(n.const_value.tag),
                         static_cast<long long>(n.const_value.as.i));
            break;
        case NodeKind::Parameter:
            std::fprintf(out, " %u", n.aux0);
            break;
        case NodeKind::PyBinary:
        case NodeKind::PyUnary:
            std::fprintf(out, " %s", binop_name(n.subop));
            break;
        case NodeKind::PyCompare:
            std::fprintf(out, " %s", cmpop_name(n.subop));
            break;
        case NodeKind::Guard:
            std::fprintf(out, " %s fs=%u", guard_name(n.subop), n.aux1);
            break;
        default:
            if (n.symbol != 0xFFFF'FFFF) {
                std::fprintf(out, " sym:%s", global_symbols().text(n.symbol).data());
            }
            if (n.shape_id != 0) {
                std::fprintf(out, " shape=%u", n.shape_id);
            }
            if (n.aux0 != 0 && n.kind != NodeKind::Parameter) {
                std::fprintf(out, " a0=%u", n.aux0);
            }
            break;
    }

    std::fprintf(out, " ins:");
    for (NodeId input : n.ins) {
        std::fprintf(out, " n%u", input);
    }
    std::fprintf(out, "\n");
}

}  // namespace

void print_graph(const Graph& g, std::FILE* out) noexcept {
    std::fprintf(out, "fun %s params=%u\n",
                 g.function_name == 0xFFFF'FFFF
                     ? "<anon>"
                     : global_symbols().text(g.function_name).data(),
                 g.n_parameters);
    g.for_each_live([&](NodeId id) { print_node_line(g, id, out); });

    if (g.frame_state_count() > 0) {
        std::fprintf(out, "frame_states:\n");
        for (std::uint32_t i = 0; i < g.frame_state_count(); ++i) {
            const FrameState& fs = g.frame_state(i);
            std::fprintf(out, "  fs%u bcoff=%u unit=%u vals:", i, fs.bytecode_offset,
                         fs.code_unit_id);
            for (NodeId v : fs.values) std::fprintf(out, " n%u", v);
            std::fprintf(out, " kinds:");
            for (std::uint8_t k : fs.kinds) std::fprintf(out, " %u", k);
            std::fprintf(out, "\n");
        }
    }
}

bool graph_to_string(const Graph& g, stdx::small_vector<char, 4096>& out) noexcept {
    // Render via a memory stream so tests never touch the filesystem.
    std::size_t cap = 4096 + g.node_count() * 64;
    out.clear();
    out.resize(cap);   // zero-fill full capacity so fmemopen has scratch space
    FILE* mem = ::fmemopen(out.data(), cap, "w");
    if (!mem) [[unlikely]] {
        out.clear();
        return false;
    }
    print_graph(g, mem);
    std::size_t used = static_cast<std::size_t>(std::ftell(mem));
    std::fclose(mem);
    out.truncate(used);
    return true;
}

}  // namespace abi_v1
}  // namespace vortex::ir
