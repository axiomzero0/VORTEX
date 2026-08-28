// =============================================================================
// vortex/ir/parser.cpp — .vortex text IR parser implementation.
//
// Line-oriented grammar (one node per line — matches printer.cpp exactly):
//   line 1 : fun <symbol> params=<n>
//   node   : n<id> = <kind> [payload...] ins: [n<id>...]
//   fs     : fs<i> bcoff=<u> unit=<u> vals: [n<id>...] kinds: [u...]
// =============================================================================

#include "vortex/ir/parser.hpp"

#include <cstdlib>
#include <cstring>

#include "vortex/support/symbol_table.hpp"

namespace vortex::ir {
inline namespace abi_v1 {

namespace {

/// Splits `text` into lines (handles \n; tolerates \r\n).
[[nodiscard]] stdx::small_vector<std::string_view, 64> split_lines(
    std::string_view text) noexcept {
    stdx::small_vector<std::string_view, 64> lines;
    std::size_t start = 0;
    for (std::size_t i = 0; i <= text.size(); ++i) {
        if (i == text.size() || text[i] == '\n') {
            std::string_view line = text.substr(start, i - start);
            if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
            lines.push_back(line);
            start = i + 1;
        }
    }
    return lines;
}

/// Whitespace-split of one line into tokens.
[[nodiscard]] stdx::small_vector<std::string_view, 16> split_tokens(
    std::string_view line) noexcept {
    stdx::small_vector<std::string_view, 16> out;
    std::size_t i = 0;
    while (i < line.size()) {
        while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) ++i;
        std::size_t start = i;
        while (i < line.size() && line[i] != ' ' && line[i] != '\t') ++i;
        if (i > start) out.push_back(line.substr(start, i - start));
    }
    return out;
}

[[nodiscard]] bool is_blank(const stdx::small_vector<std::string_view, 16>& toks) noexcept {
    return toks.empty() || (toks.size() == 1 && (toks[0].empty() || toks[0][0] == '#'));
}

[[nodiscard]] bool parse_int_tok(std::string_view tok, std::int64_t& out) noexcept {
    if (tok.empty()) return false;
    std::array<char, 32> buf{};
    if (tok.size() >= buf.size()) return false;
    std::memcpy(buf.data(), tok.data(), tok.size());
    char* end = nullptr;
    long long v = std::strtoll(buf.data(), &end, 10);
    if (end == buf.data()) return false;
    out = v;
    return true;
}

[[nodiscard]] bool parse_uint_tok(std::string_view tok, std::uint64_t& out) noexcept {
    std::int64_t v = 0;
    if (!parse_int_tok(tok, v) || v < 0) return false;
    out = static_cast<std::uint64_t>(v);
    return true;
}

[[nodiscard]] bool parse_double_tok(std::string_view tok, double& out) noexcept {
    if (tok.empty()) return false;
    std::array<char, 64> buf{};
    if (tok.size() >= buf.size()) return false;
    std::memcpy(buf.data(), tok.data(), tok.size());
    char* end = nullptr;
    double v = std::strtod(buf.data(), &end);
    if (end == buf.data()) return false;
    out = v;
    return true;
}

[[nodiscard]] bool binop_from_name(std::string_view t, std::uint16_t& subop) noexcept {
    struct E { const char* n; BinOpKind k; };
    static constexpr E table[] = {
        {"+", BinOpKind::Add}, {"-", BinOpKind::Sub}, {"*", BinOpKind::Mul},
        {"/", BinOpKind::TrueDiv}, {"//", BinOpKind::FloorDiv}, {"%", BinOpKind::Mod},
        {"**", BinOpKind::Pow}, {"<<", BinOpKind::LShift}, {">>", BinOpKind::RShift},
        {"&", BinOpKind::BitAnd}, {"|", BinOpKind::BitOr}, {"^", BinOpKind::BitXor},
        {"@", BinOpKind::MatMul},
    };
    for (const E& e : table) {
        if (t == e.n) { subop = static_cast<std::uint16_t>(e.k); return true; }
    }
    return false;
}

[[nodiscard]] bool cmpop_from_name(std::string_view t, std::uint16_t& subop) noexcept {
    struct E { const char* n; CmpOpKind k; };
    static constexpr E table[] = {
        {"<", CmpOpKind::LT}, {"<=", CmpOpKind::LE}, {">", CmpOpKind::GT},
        {">=", CmpOpKind::GE}, {"==", CmpOpKind::EQ}, {"!=", CmpOpKind::NE},
        {"is", CmpOpKind::Is}, {"is-not", CmpOpKind::IsNot}, {"in", CmpOpKind::In},
        {"not-in", CmpOpKind::NotIn},
    };
    for (const E& e : table) {
        if (t == e.n) { subop = static_cast<std::uint16_t>(e.k); return true; }
    }
    return false;
}

[[nodiscard]] bool guard_from_name(std::string_view t, std::uint16_t& subop) noexcept {
    struct E { const char* n; GuardKind k; };
    static constexpr E table[] = {
        {"type_is", GuardKind::TypeIs}, {"shape_is", GuardKind::ShapeIs},
        {"int_fits", GuardKind::IntFits}, {"not_none", GuardKind::NotNone},
        {"bounds", GuardKind::Bounds}, {"no_overflow", GuardKind::NoOverflow},
        {"alias_disjoint", GuardKind::AliasDisjoint},
        {"module_version", GuardKind::ModuleVersion},
        {"mono_call", GuardKind::MonomorphicCall},
    };
    for (const E& e : table) {
        if (t == e.n) { subop = static_cast<std::uint16_t>(e.k); return true; }
    }
    return false;
}

[[nodiscard]] Diagnostic syntax_error(std::uint32_t line, std::string_view tok,
                                      std::string_view expected) noexcept {
    Diagnostic d = Diagnostic::error("vortex-ir parse: unexpected token",
                                     diag_code::parse_unexpected_token);
    d.where.line = line;
    d.expected = expected;
    d.actual = tok;
    d.fix = "Check the golden .vortex grammar in vortex/ir/parser.hpp";
    return d;
}

struct ParsedNode {
    std::uint32_t id{};
    NodeKind kind{};
    std::uint16_t subop{0};
    Value const_value{};
    std::uint32_t symbol{0xFFFF'FFFF};
    std::uint32_t shape_id{0};
    std::uint32_t aux0{0};
    bool has_aux0{false};
    std::uint32_t aux1{0};
    bool has_aux1{false};
    stdx::small_vector<NodeId, 4> ins{};
};

}  // namespace

Result<void> parse_graph(std::string_view text, Graph& g) noexcept {
    auto lines = split_lines(text);
    std::uint32_t line_no = 0;
    std::uint32_t header_lines = 0;

    // --- header ------------------------------------------------------------
    std::string_view name_tok;
    std::uint64_t nparams = 0;
    {
        // skip blanks/comments
        stdx::small_vector<std::string_view, 16> toks;
        do {
            if (line_no >= lines.size()) {
                return fail_msg("vortex-ir parse: empty input", diag_code::parse_unexpected_token);
            }
            toks = split_tokens(lines[line_no]);
            ++line_no;
            ++header_lines;
        } while (is_blank(toks));
        if (toks.empty() || toks[0] != "fun") {
            return fail(syntax_error(line_no, toks.empty() ? "" : toks[0], "'fun' header"));
        }
        if (toks.size() < 3 || !toks[2].starts_with("params=")) {
            return fail(syntax_error(line_no, toks.size() < 3 ? "" : toks[2], "'params=' N"));
        }
        name_tok = toks[1];
        if (!parse_uint_tok(toks[2].substr(7), nparams)) {
            return fail(syntax_error(line_no, toks[2], "integer param count"));
        }
    }
    g.function_name = global_symbols().intern(name_tok);
    g.n_parameters = static_cast<std::uint32_t>(nparams);

    stdx::small_vector<ParsedNode, 128> parsed;
    bool in_frame_states = false;
    std::uint64_t max_id = 0;

    for (; line_no < lines.size(); ++line_no) {
        auto toks = split_tokens(lines[line_no]);
        if (is_blank(toks)) continue;
        if (toks[0] == "frame_states:") {
            in_frame_states = true;
            continue;
        }
        if (in_frame_states) {
            // fs<i> bcoff=<u> unit=<u> vals: ... kinds: ...
            if (!toks[0].starts_with("fs")) {
                return fail(syntax_error(line_no + 1, toks[0], "'fs<i>'"));
            }
            FrameState fs;
            std::size_t i = 1;
            if (i >= toks.size() || !toks[i].starts_with("bcoff=")) {
                return fail(syntax_error(line_no + 1, toks[i], "'bcoff='"));
            }
            std::uint64_t bcoff = 0;
            if (!parse_uint_tok(toks[i].substr(6), bcoff)) {
                return fail(syntax_error(line_no + 1, toks[i], "int"));
            }
            ++i;
            if (i >= toks.size() || !toks[i].starts_with("unit=")) {
                return fail(syntax_error(line_no + 1, toks[i], "'unit='"));
            }
            std::uint64_t unit = 0;
            if (!parse_uint_tok(toks[i].substr(5), unit)) {
                return fail(syntax_error(line_no + 1, toks[i], "int"));
            }
            ++i;
            fs.bytecode_offset = static_cast<std::uint32_t>(bcoff);
            fs.code_unit_id = static_cast<std::uint32_t>(unit);
            if (i >= toks.size() || toks[i] != "vals:") {
                return fail(syntax_error(line_no + 1, toks[i], "'vals:'"));
            }
            ++i;
            for (; i < toks.size() && toks[i] != "kinds:"; ++i) {
                if (!toks[i].starts_with("n") || toks[i].size() < 2) {
                    return fail(syntax_error(line_no + 1, toks[i], "node id"));
                }
                std::uint64_t vid = 0;
                if (!parse_uint_tok(toks[i].substr(1), vid)) {
                    return fail(syntax_error(line_no + 1, toks[i], "node id"));
                }
                fs.values.push_back(static_cast<NodeId>(vid));
            }
            if (i >= toks.size() || toks[i] != "kinds:") {
                return fail(syntax_error(line_no + 1, "eol", "'kinds:'"));
            }
            ++i;
            for (; i < toks.size(); ++i) {
                std::uint64_t k = 0;
                if (!parse_uint_tok(toks[i], k)) {
                    return fail(syntax_error(line_no + 1, toks[i], "kind byte"));
                }
                fs.kinds.push_back(static_cast<std::uint8_t>(k));
            }
            g.add_frame_state(fs);
            continue;
        }

        // --- node line: n<id> = <kind> [payload...] ins: [inputs...] -------------
        if (!toks[0].starts_with("n") || toks[0].size() < 2) {
            return fail(syntax_error(line_no + 1, toks[0], "'n<id> ='"));
        }
        std::uint64_t id = 0;
        if (!parse_uint_tok(toks[0].substr(1), id)) {
            return fail(syntax_error(line_no + 1, toks[0], "numeric node id"));
        }
        std::size_t i = 1;
        if (i >= toks.size() || toks[i] != "=") {
            return fail(syntax_error(line_no + 1, i < toks.size() ? toks[i] : "eol", "'='"));
        }
        ++i;
        if (i >= toks.size()) {
            return fail(syntax_error(line_no + 1, "eol", "node kind"));
        }
        std::string_view kind_tok = toks[i];
        ++i;
        NodeKind kind{};
        if (!node_kind_from_name(kind_tok.data(), kind_tok.size(), kind)) {
            return fail(syntax_error(line_no + 1, kind_tok, "known node kind"));
        }

        ParsedNode pn;
        pn.id = static_cast<std::uint32_t>(id);
        pn.kind = kind;

        // payloads until "ins:"
        for (; i < toks.size() && toks[i] != "ins:"; ++i) {
            std::string_view p = toks[i];
            switch (kind) {
                case NodeKind::ConstInt: {
                    std::int64_t v = 0;
                    if (!parse_int_tok(p, v)) return fail(syntax_error(line_no + 1, p, "int64"));
                    pn.const_value = Value::integer(v);
                    break;
                }
                case NodeKind::ConstFloat: {
                    double v = 0;
                    if (!parse_double_tok(p, v)) return fail(syntax_error(line_no + 1, p, "double"));
                    pn.const_value = Value::real(v);
                    break;
                }
                case NodeKind::ConstPy: {
                    if (p.starts_with("tag=")) {
                        std::uint64_t t = 0;
                        if (!parse_uint_tok(p.substr(4), t)) {
                            return fail(syntax_error(line_no + 1, p, "tag int"));
                        }
                        // NaN-boxed Value: set tag + zero payload
                        pn.const_value = Value::from_raw(static_cast<std::uint64_t>(t) << 48);
                    } else if (p.starts_with("i=")) {
                        std::int64_t v = 0;
                        if (!parse_int_tok(p.substr(2), v)) {
                            return fail(syntax_error(line_no + 1, p, "int64"));
                        }
                        pn.const_value = Value::integer(v);
                    }
                    break;
                }
                case NodeKind::Parameter: {
                    std::uint64_t v = 0;
                    if (!parse_uint_tok(p, v)) return fail(syntax_error(line_no + 1, p, "param idx"));
                    pn.aux0 = static_cast<std::uint32_t>(v);
                    pn.has_aux0 = true;
                    break;
                }
                case NodeKind::PyBinary:
                case NodeKind::PyUnary:
                    if (!binop_from_name(p, pn.subop)) {
                        return fail(syntax_error(line_no + 1, p, "binop"));
                    }
                    break;
                case NodeKind::PyCompare:
                    if (!cmpop_from_name(p, pn.subop)) {
                        return fail(syntax_error(line_no + 1, p, "cmpop"));
                    }
                    break;
                case NodeKind::Guard: {
                    if (p.starts_with("fs=")) {
                        std::uint64_t v = 0;
                        if (!parse_uint_tok(p.substr(3), v)) {
                            return fail(syntax_error(line_no + 1, p, "fs index"));
                        }
                        pn.aux1 = static_cast<std::uint32_t>(v);
                        pn.has_aux1 = true;
                    } else if (!guard_from_name(p, pn.subop)) {
                        return fail(syntax_error(line_no + 1, p, "guard kind"));
                    }
                    break;
                }
                default:
                    if (p.starts_with("sym:")) {
                        pn.symbol = global_symbols().intern(p.substr(4));
                    } else if (p.starts_with("shape=")) {
                        std::uint64_t v = 0;
                        if (!parse_uint_tok(p.substr(6), v)) {
                            return fail(syntax_error(line_no + 1, p, "shape id"));
                        }
                        pn.shape_id = static_cast<std::uint32_t>(v);
                    } else if (p.starts_with("a0=")) {
                        std::uint64_t v = 0;
                        if (!parse_uint_tok(p.substr(3), v)) {
                            return fail(syntax_error(line_no + 1, p, "aux0"));
                        }
                        pn.aux0 = static_cast<std::uint32_t>(v);
                        pn.has_aux0 = true;
                    } else if (p.starts_with("tag=") || p.starts_with("i=")) {
                        break;   // ConstPy reached via default arm
                    } else {
                        return fail(syntax_error(line_no + 1, p, "'ins:' or payload"));
                    }
                    break;
            }
        }
        if (i >= toks.size() || toks[i] != "ins:") {
            return fail(syntax_error(line_no + 1, "eol", "'ins:'"));
        }
        ++i;
        for (; i < toks.size(); ++i) {
            if (!toks[i].starts_with("n") || toks[i].size() < 2) {
                return fail(syntax_error(line_no + 1, toks[i], "input node id"));
            }
            std::uint64_t input_id = 0;
            if (!parse_uint_tok(toks[i].substr(1), input_id)) {
                return fail(syntax_error(line_no + 1, toks[i], "input node id"));
            }
            pn.ins.push_back(static_cast<NodeId>(input_id));
        }
        if (pn.id > max_id) max_id = pn.id;
        parsed.push_back(std::move(pn));
    }

    // --- materialize -----------------------------------------------------------
    if (max_id > 1000000) [[unlikely]] {
        return fail_msg("vortex-ir parse: node id space too large",
                        diag_code::parse_unexpected_token);
    }
    stdx::small_vector<NodeId, 256> id_map(static_cast<std::size_t>(max_id) + 1, invalid_node);
    id_map[0] = 0;   // reserved slot
    for (std::uint32_t i = 1; i <= max_id; ++i) {
        id_map[i] = g.create(NodeKind::Unreachable);
    }
    for (ParsedNode& pn : parsed) {
        Node& n = g.node(id_map[pn.id]);
        n.kind = pn.kind;
        n.subop = pn.subop;
        n.const_value = pn.const_value;
        n.symbol = pn.symbol;
        n.shape_id = pn.shape_id;
        if (pn.has_aux0) n.aux0 = pn.aux0;
        if (pn.has_aux1) n.aux1 = pn.aux1;
        if (pn.kind == NodeKind::Start) g.set_start(id_map[pn.id]);
    }
    for (ParsedNode& pn : parsed) {
        NodeId nid = id_map[pn.id];
        for (NodeId input : pn.ins) {
            if (input > max_id) {
                return fail_msg("vortex-ir parse: input id beyond declared nodes",
                                diag_code::parse_unexpected_token);
            }
            g.add_input(nid, id_map[input]);
        }
    }
    return {};
}

}  // namespace abi_v1
}  // namespace vortex::ir
