// =============================================================================
// vortex/frontend/lowering.cpp — Pass 1: AST -> Sea of Nodes implementation.
//
// Structured SSA construction:
//   - A lowering cursor holds (control, effect-tail, var map).
//   - Branches lower with per-arm var snapshots; merges create Region + Phi
//     (control input appended last) for every variable whose value differs.
//   - Loops pre-create header Phis (self-referencing backedge, patched after
//     the body) including an EffectPhi for the memory chain.
//   - break/continue/return set control "unreachable" for the remainder of
//     their block; loop exits and backedges are merged with Regions after
//     the body lowers.
//   - try bodies snapshot vars after EVERY statement; the Catch region merges
//     those raise points, giving handlers exact Python binding visibility.
//   - Closures use CPython-style cells: locals captured transitively by any
//     nested def become cells at entry, synced on every write, and passed by
//     shared reference at def time (late-binding semantics preserved).
// =============================================================================

#include "vortex/frontend/lowering.hpp"

#include "vortex/ir/node_kind.hpp"
#include "vortex/support/symbol_table.hpp"

namespace vortex::fe {
inline namespace abi_v1 {

namespace {

using vortex::ir::Graph;
using vortex::ir::Node;
using vortex::ir::NodeFlag;
using vortex::ir::NodeKind;
using vortex::ir::NodeId;

constexpr std::uint16_t bool_and = 1, bool_or = 2;
constexpr std::uint16_t un_neg = 1, un_invert = 2, un_not = 3;
constexpr SymbolId sym_invalid = 0xFFFF'FFFF;

using VarMap = stdx::flat_map<SymbolId, NodeId, 16>;

[[nodiscard]] VarMap clone_vars(const VarMap& v) noexcept { return v; }

// ---------------------------------------------------------------------------
// Name analysis (transitive free variables).
// ---------------------------------------------------------------------------
struct NameSets {
    stdx::small_vector<SymbolId, 16> bound{};
    stdx::small_vector<SymbolId, 16> referenced{};          // this body only
    stdx::small_vector<SymbolId, 16> child_referenced{};    // from nested defs (transitive)
    stdx::small_vector<SymbolId, 16> globals_declared{};
};

[[nodiscard]] bool contains_sym(const stdx::small_vector<SymbolId, 16>& v, SymbolId s) noexcept {
    for (SymbolId x : v) {
        if (x == s) return true;
    }
    return false;
}
[[nodiscard]] bool contains_sym8(const stdx::small_vector<SymbolId, 8>& v, SymbolId s) noexcept {
    for (SymbolId x : v) {
        if (x == s) return true;
    }
    return false;
}

void add_unique(stdx::small_vector<SymbolId, 16>& v, SymbolId s) noexcept {
    if (!contains_sym(v, s)) v.push_back(s);
}

/// Fold a nested scope's references into the parent's child_referenced set,
/// excluding names the nested scope binds itself.
void fold_child_refs(NameSets& parent, const NameSets& child,
                     const stdx::small_vector<SymbolId, 16>& child_bound) noexcept {
    for (SymbolId r : child.referenced) {
        if (!contains_sym(child_bound, r)) add_unique(parent.child_referenced, r);
    }
    for (SymbolId r : child.child_referenced) {
        if (!contains_sym(child_bound, r)) add_unique(parent.child_referenced, r);
    }
}

void names_of_expr(Expr* e, NameSets& out) noexcept;
void names_of_stmt(Stmt* s, NameSets& out) noexcept;
void names_of_stmts(const StmtList& body, NameSets& out) noexcept {
    for (Stmt* s : body) names_of_stmt(s, out);
}

void names_of_expr(Expr* e, NameSets& out) noexcept {
    if (!e) return;
    switch (e->kind) {
        case ExprKind::Name: out.referenced.push_back(e->name); return;
        default: break;
    }
    names_of_expr(e->sub, out);
    names_of_expr(e->index, out);
    names_of_expr(e->lower, out);
    names_of_expr(e->upper, out);
    for (Expr* a : e->args) names_of_expr(a, out);
    if (e->comp.target) names_of_expr(e->comp.target, out);
    if (e->comp.iter) names_of_expr(e->comp.iter, out);
    if (e->comp.cond) names_of_expr(e->comp.cond, out);
    for (Argument& a : e->call_args) names_of_expr(a.value, out);
}

void names_of_stmt(Stmt* s, NameSets& out) noexcept {
    if (!s) return;
    switch (s->kind) {
        case StmtKind::FunctionDef: {
            add_unique(out.bound, s->name);
            NameSets child;
            names_of_stmts(s->body, child);
            for (SymbolId p : s->params) add_unique(out.bound, p);
            // Everything the child references but does not bind itself is a
            // potential capture from THIS scope (or above): fold upward.
            fold_child_refs(out, child, child.bound);
            for (SymbolId g : child.globals_declared) add_unique(out.globals_declared, g);
            return;
        }
        case StmtKind::ClassDef: {
            add_unique(out.bound, s->name);
            if (s->base_name != sym_invalid) out.referenced.push_back(s->base_name);
            NameSets child;
            names_of_stmts(s->body, child);
            fold_child_refs(out, child, child.bound);
            return;
        }
        case StmtKind::Assign:
            for (Expr* t : s->targets) {
                if (t->kind == ExprKind::Name) add_unique(out.bound, t->name);
                else names_of_expr(t, out);
            }
            names_of_expr(s->value, out);
            return;
        case StmtKind::AugAssign:
            names_of_expr(s->targets[0], out);
            names_of_expr(s->value, out);
            return;
        case StmtKind::For:
            if (s->for_target && s->for_target->kind == ExprKind::Name) {
                add_unique(out.bound, s->for_target->name);
            } else if (s->for_target) {
                names_of_expr(s->for_target, out);
            }
            names_of_expr(s->iter, out);
            names_of_stmts(s->body, out);
            names_of_stmts(s->orelse, out);
            return;
        case StmtKind::Global:
        case StmtKind::Nonlocal:
            for (SymbolId n : s->names) add_unique(out.globals_declared, n);
            return;
        case StmtKind::Try:
            names_of_stmts(s->body, out);
            for (auto& h : s->handlers) {
                if (h.type_name != sym_invalid) out.referenced.push_back(h.type_name);
                if (h.bind_name != sym_invalid) add_unique(out.bound, h.bind_name);
                names_of_stmts(h.body, out);
            }
            names_of_stmts(s->orelse, out);
            names_of_stmts(s->finalbody, out);
            return;
        default:
            names_of_expr(s->cond, out);
            names_of_expr(s->value, out);
            names_of_expr(s->iter, out);
            names_of_expr(s->test, out);
            names_of_stmts(s->body, out);
            names_of_stmts(s->orelse, out);
            return;
    }
}

[[nodiscard]] NameSets analyze(const StmtList& body) noexcept {
    NameSets out;
    names_of_stmts(body, out);
    return out;
}

}  // namespace

stdx::small_vector<SymbolId, 16> bound_names(const StmtList& body) noexcept {
    return analyze(body).bound;
}

stdx::small_vector<SymbolId, 16> free_names(const StmtList& body) noexcept {
    NameSets n = analyze(body);
    stdx::small_vector<SymbolId, 16> out;
    for (SymbolId r : n.referenced) {
        if (contains_sym(n.bound, r)) continue;
        if (contains_sym(n.globals_declared, r)) continue;
        if (contains_sym(out, r)) continue;
        out.push_back(r);
    }
    // Names needed by transitively nested scopes (and not bound here).
    for (SymbolId r : n.child_referenced) {
        if (contains_sym(n.bound, r)) continue;
        if (contains_sym(n.globals_declared, r)) continue;
        if (contains_sym(out, r)) continue;
        out.push_back(r);
    }
    return out;
}

namespace {

// ---------------------------------------------------------------------------
// The Lowerer.
// ---------------------------------------------------------------------------
class Lowerer {
public:
    Lowerer(Module& m, LowerContext& ctx, LoweredUnit& unit, bool is_toplevel,
            bool class_body, const stdx::small_vector<SymbolId, 8>& captured)
        : mod_(m), ctx_(ctx), unit_(unit), is_toplevel_(is_toplevel), class_body_(class_body) {
        for (SymbolId c : captured) captured_.push_back(c);
    }

    Result<void> run(Stmt* def, const StmtList& body,
                     const stdx::small_vector<SymbolId, 8>& cell_vars) noexcept;

private:
    Graph& g() noexcept { return unit_.graph; }

    // --- constants ----------------------------------------------------------
    NodeId const_int(std::int64_t v) noexcept {
        NodeId n = g().create(NodeKind::ConstInt);
        Node& node = g().node(n);
        node.const_value = Value::integer(v);
        node.set_flag(NodeFlag::Pure);
        return n;
    }
    NodeId const_float(double v) noexcept {
        NodeId n = g().create(NodeKind::ConstFloat);
        Node& node = g().node(n);
        node.const_value = Value::real(v);
        node.set_flag(NodeFlag::Pure);
        return n;
    }
    NodeId const_none() noexcept {
        NodeId n = g().create(NodeKind::ConstPy);
        Node& node = g().node(n);
        node.const_value = Value::none();
        node.set_flag(NodeFlag::Pure);
        return n;
    }
    NodeId const_string(std::uint32_t offset, std::uint32_t len) noexcept {
        NodeId n = g().create(NodeKind::ConstPy);
        Node& node = g().node(n);
        node.const_value = Value::none();
        node.aux0 = offset;
        node.aux1 = len;
        node.set_flag(NodeFlag::Pure);
        return n;
    }
    NodeId const_symbol_str(SymbolId sym) noexcept {
        NodeId n = g().create(NodeKind::ConstPy);
        Node& node = g().node(n);
        node.const_value = Value::none();
        node.symbol = sym;
        node.aux0 = 0xFFFF'FFFF;
        node.set_flag(NodeFlag::Pure);
        return n;
    }

    // --- node factories -------------------------------------------------------
    NodeId effect_op(NodeKind k, std::initializer_list<NodeId> data_ins,
                     bool may_throw = true) noexcept {
        NodeId n = g().create(k);
        Node& node = g().node(n);
        node.set_flag(NodeFlag::OnEffectChain);
        if (may_throw) node.set_flag(NodeFlag::MayThrow);
        g().add_input(n, control_);
        g().add_input(n, memory_);
        for (NodeId d : data_ins) {
            if (d != vortex::ir::invalid_node) g().add_input(n, d);
        }
        memory_ = n;
        return n;
    }

    NodeId call_native(NativeHelper helper, std::initializer_list<NodeId> data_ins,
                       bool may_throw = false) noexcept {
        NodeId n = effect_op(NodeKind::CallNative, data_ins, may_throw);
        g().node(n).subop = static_cast<std::uint16_t>(helper);
        return n;
    }

    NodeId py_op(NodeKind k, std::uint16_t subop, std::initializer_list<NodeId> data_ins) noexcept {
        NodeId n = g().create(k, data_ins);
        Node& node = g().node(n);
        node.subop = subop;
        node.set_flag(NodeFlag::Pure);
        node.set_flag(NodeFlag::MayThrow);
        node.set_flag(NodeFlag::MayCall);
        return n;
    }

    // --- variables ---------------------------------------------------------------
    [[nodiscard]] NodeId read_var(SymbolId name) noexcept {
        if (NodeId* slot = vars_.get(name)) return *slot;
        if (contains_sym8(captured_, name)) {
            NodeId idx = const_int(capture_index(name));
            NodeId cell = effect_op(NodeKind::LoadIndex, {cells_param_, idx}, true);
            return call_native(NativeHelper::CellGet, {cell}, true);
        }
        if (deleted_locals_.contains(name)) {
            return call_native(NativeHelper::UnboundCheck, {}, true);
        }
        NodeId n = effect_op(NodeKind::LoadGlobal, {}, true);
        g().node(n).symbol = name;
        return n;
    }

    void write_var(SymbolId name, NodeId value) noexcept {
        vars_.insert(name, value);
        deleted_locals_.erase(name);
        if (contains_sym8(cell_vars_, name)) {
            NodeId cell = cell_of(name);
            call_native(NativeHelper::CellSet, {cell, value});
        }
        if (is_toplevel_) {
            // Module toplevel publishes bindings globally; class bodies build
            // their namespace dict at return instead (no global pollution).
            NodeId n = effect_op(NodeKind::StoreGlobal, {value});
            g().node(n).symbol = name;
        }
    }

    [[nodiscard]] std::uint32_t capture_index(SymbolId name) const noexcept {
        for (std::uint32_t i = 0; i < captured_.size(); ++i) {
            if (captured_[i] == name) return i;
        }
        return 0;
    }

    NodeId cell_of(SymbolId name) noexcept {
        if (NodeId* slot = local_cells_.get(name)) return *slot;
        // Entry-time creation covers all cell vars; reaching here means the
        // cell var was created mid-flight (def before entry loop) — make one.
        NodeId current = vars_.contains(name) ? *vars_.get(name) : const_none();
        NodeId cell = call_native(NativeHelper::MakeCell, {current});
        local_cells_.insert(name, cell);
        return cell;
    }

    // --- arm merging ---------------------------------------------------------------
    struct ArmState {
        NodeId control{vortex::ir::invalid_node};
        VarMap vars{};
    };

    NodeId merge_arms(stdx::small_vector<ArmState, 4>& arms) noexcept {
        stdx::small_vector<ArmState*, 4> live;
        for (ArmState& arm : arms) {
            if (arm.control != vortex::ir::invalid_node) live.push_back(&arm);
        }
        if (live.empty()) {
            control_ = vortex::ir::invalid_node;
            return vortex::ir::invalid_node;
        }
        if (live.size() == 1) {
            control_ = live[0]->control;
            vars_ = live[0]->vars;
            return control_;
        }
        NodeId region = g().create(NodeKind::Region);
        for (ArmState* arm : live) g().add_input(region, arm->control);

        VarMap merged = live[0]->vars;
        for (std::size_t i = 1; i < live.size(); ++i) {
            for (auto& kv : live[i]->vars) {
                if (NodeId* cur = merged.get(kv.first)) {
                    if (*cur != kv.second) {
                        NodeId phi = g().create(NodeKind::Phi);
                        g().add_input(phi, *cur);
                        g().add_input(phi, kv.second);
                        g().add_input(phi, region);
                        merged.insert(kv.first, phi);
                    }
                } else {
                    merged.insert(kv.first, kv.second);
                }
            }
        }
        control_ = region;
        vars_ = std::move(merged);
        return region;
    }

    // For statements that can appear inside try blocks: record raise snapshots.
    stdx::small_vector<ArmState, 8>* try_snapshots_{nullptr};

    // --- statements ------------------------------------------------------------------
    Result<void> lower_stmts(const StmtList& body) noexcept;
    Result<void> lower_stmts_tracked(const StmtList& body) noexcept;
    Result<void> lower_stmt(Stmt* s) noexcept;
    Result<void> lower_if(Stmt* s) noexcept;
    Result<void> lower_while(Stmt* s) noexcept;
    Result<void> lower_for(Stmt* s) noexcept;
    Result<void> lower_try(Stmt* s) noexcept;
    Result<void> lower_function_def(Stmt* s) noexcept;
    Result<void> lower_class_def(Stmt* s) noexcept;
    Result<NodeId> lower_assign_target(Expr* target, NodeId value) noexcept;

    void emit_return(NodeId value) noexcept {
        if (control_ == vortex::ir::invalid_node) return;
        g().create(NodeKind::Return, {control_, value});
        control_ = vortex::ir::invalid_node;
    }

    // --- expressions --------------------------------------------------------------------
    Result<NodeId> lower_expr(Expr* e) noexcept;
    Result<NodeId> lower_boolop(Expr* e) noexcept;
    Result<NodeId> lower_compare_chain(Expr* e) noexcept;
    Result<NodeId> lower_call(Expr* e) noexcept;
    Result<NodeId> lower_listcomp(Expr* e) noexcept;

    Module& mod_;
    LowerContext& ctx_;
    LoweredUnit& unit_;
    bool is_toplevel_;
    bool class_body_;
    stdx::small_vector<SymbolId, 8> captured_;
    stdx::small_vector<SymbolId, 8> cell_vars_{};

    NodeId control_{vortex::ir::invalid_node};
    NodeId memory_{vortex::ir::invalid_node};
    NodeId cells_param_{vortex::ir::invalid_node};

    VarMap vars_{};
    VarMap deleted_locals_{};
    VarMap local_cells_{};

    struct LoopCtx {
        stdx::small_vector<ArmState, 4> exits{};
        stdx::small_vector<ArmState, 4> backedges{};
    };
    struct LoopSkeleton {
        NodeId loop{vortex::ir::invalid_node};
        NodeId eff_phi{vortex::ir::invalid_node};
        stdx::small_vector<std::pair<SymbolId, NodeId>, 8> header_phis{};
    };
    void patch_loop(const LoopSkeleton& sk, LoopCtx& ctx) noexcept;
    stdx::small_vector<LoopCtx*, 8> loop_stack_{};

public:
    stdx::small_vector<SymbolId, 16> bound_names_saved_{};

    [[nodiscard]] std::uint32_t depth_{0};
    static constexpr std::uint32_t max_lower_depth = 64;
};

// ---------------------------------------------------------------------------
// Entry
// ---------------------------------------------------------------------------
Result<void> Lowerer::run(Stmt* def, const StmtList& body,
                          const stdx::small_vector<SymbolId, 8>& cell_vars) noexcept {
    cell_vars_ = cell_vars;
    Graph& graph = g();

    NodeId start = graph.create(NodeKind::Start);
    graph.set_start(start);
    control_ = start;
    memory_ = start;

    if (def) {
        graph.function_name = def->name;
        graph.n_parameters = static_cast<std::uint32_t>(def->params.size());
        unit_.name = def->name;
        for (std::uint32_t i = 0; i < def->params.size(); ++i) {
            NodeId p = graph.create(NodeKind::Parameter, {start});
            Node& pn = graph.node(p);
            pn.aux0 = i;
            pn.set_flag(NodeFlag::Pure);
            unit_.param_names.push_back(def->params[i]);
            if (def->params[i] != sym_invalid) vars_.insert(def->params[i], p);
        }
        unit_.has_varargs = def->has_varargs;
        unit_.has_kwargs = def->has_kwargs;
        for (Expr* d : def->defaults) unit_.default_exprs.push_back(d);
        if (!captured_.empty()) {
            NodeId p = graph.create(NodeKind::Parameter, {start});
            Node& pn = graph.node(p);
            pn.aux0 = static_cast<std::uint32_t>(def->params.size());
            pn.set_flag(NodeFlag::Pure);
            cells_param_ = p;
        }
    } else {
        graph.function_name = global_symbols().intern("__module__");
        graph.n_parameters = 0;
        unit_.name = graph.function_name;
    }

    unit_.code_unit_id = ctx_.next_code_unit_id++;
    ++ctx_.units_lowered;

    // Create cells for captured locals at entry (unbound until first write).
    for (SymbolId cv : cell_vars_) {
        NodeId unbound = g().create(NodeKind::ConstPy);
        Node& un = g().node(unbound);
        un.const_value = Value::object(nullptr);   // unbound sentinel
        un.set_flag(NodeFlag::Pure);
        NodeId cell = call_native(NativeHelper::MakeCell, {unbound});
        local_cells_.insert(cv, cell);
    }

    VORTEX_TRY_VOID(lower_stmts(body));

    if (class_body_) {
        // Class bodies return their namespace as a dict.
        if (control_ != vortex::ir::invalid_node) {
            NodeId dict = effect_op(NodeKind::NewDict, {});
            for (auto& kv : vars_) {
                effect_op(NodeKind::StoreIndex,
                          {dict, const_symbol_str(kv.first), kv.second});
            }
            emit_return(dict);
        }
        return {};
    }

    if (control_ != vortex::ir::invalid_node) {
        emit_return(const_none());
    }
    return {};
}

Result<void> Lowerer::lower_stmts(const StmtList& body) noexcept {
    for (Stmt* s : body) {
        if (control_ == vortex::ir::invalid_node) return {};
        VORTEX_TRY_VOID(lower_stmt(s));
    }
    return {};
}

// try-body lowering: snapshot (control, vars) after each statement so the
// Catch region can merge exact binding visibility.
Result<void> Lowerer::lower_stmts_tracked(const StmtList& body) noexcept {
    for (Stmt* s : body) {
        if (control_ == vortex::ir::invalid_node) return {};
        VORTEX_TRY_VOID(lower_stmt(s));
        if (control_ != vortex::ir::invalid_node && try_snapshots_) {
            try_snapshots_->push_back(ArmState{control_, clone_vars(vars_)});
        }
    }
    return {};
}

// ---------------------------------------------------------------------------
// Statements
// ---------------------------------------------------------------------------
Result<void> Lowerer::lower_stmt(Stmt* s) noexcept {
    if (++depth_ > max_lower_depth) {
        return fail_msg("lower: expression nesting too deep",
                        diag_code::parse_unexpected_token);
    }
    --depth_;

    switch (s->kind) {
        case StmtKind::Pass:
            return {};

        case StmtKind::Expr: {
            if (s->value && s->value->kind == ExprKind::Yield) {
                unit_.is_generator = true;
                NodeId v = const_none();
                if (s->value->sub) v = VORTEX_TRY(lower_expr(s->value->sub));
                effect_op(NodeKind::Yield, {v});
                return {};
            }
            if (s->value) (void)VORTEX_TRY(lower_expr(s->value));
            return {};
        }

        case StmtKind::Return: {
            NodeId v = const_none();
            if (s->value) v = VORTEX_TRY(lower_expr(s->value));
            emit_return(v);
            return {};
        }

        case StmtKind::Assign: {
            NodeId value = VORTEX_TRY(lower_expr(s->value));
            for (Expr* target : s->targets) {
                VORTEX_TRY(lower_assign_target(target, value));
            }
            return {};
        }

        case StmtKind::AugAssign: {
            Expr* target = s->targets[0];
            if (target->kind == ExprKind::Name) {
                NodeId cur = read_var(target->name);
                NodeId rhs = VORTEX_TRY(lower_expr(s->value));
                NodeId res = py_op(NodeKind::PyBinary, s->aug_op, {cur, rhs});
                write_var(target->name, res);
                return {};
            }
            if (target->kind == ExprKind::Subscript) {
                NodeId base = VORTEX_TRY(lower_expr(target->sub));
                NodeId idx = VORTEX_TRY(lower_expr(target->index));
                NodeId cur = effect_op(NodeKind::LoadIndex, {base, idx}, true);
                NodeId rhs = VORTEX_TRY(lower_expr(s->value));
                NodeId res = py_op(NodeKind::PyBinary, s->aug_op, {cur, rhs});
                effect_op(NodeKind::StoreIndex, {base, idx, res});
                return {};
            }
            if (target->kind == ExprKind::Attribute) {
                NodeId base = VORTEX_TRY(lower_expr(target->sub));
                NodeId cur = effect_op(NodeKind::LoadAttr, {base}, true);
                g().node(cur).symbol = target->attr;
                NodeId rhs = VORTEX_TRY(lower_expr(s->value));
                NodeId res = py_op(NodeKind::PyBinary, s->aug_op, {cur, rhs});
                NodeId store = effect_op(NodeKind::StoreAttr, {base, res});
                g().node(store).symbol = target->attr;
                return {};
            }
            return fail_msg("lower: augmented assignment target must be name/subscript/attr",
                            diag_code::parse_unexpected_token);
        }

        case StmtKind::If:
            return lower_if(s);
        case StmtKind::While:
            return lower_while(s);
        case StmtKind::For:
            return lower_for(s);
        case StmtKind::Try:
            return lower_try(s);
        case StmtKind::FunctionDef:
            return lower_function_def(s);
        case StmtKind::ClassDef:
            return lower_class_def(s);

        case StmtKind::Break: {
            if (loop_stack_.empty()) {
                return fail_msg("lower: 'break' outside loop", diag_code::parse_unexpected_token);
            }
            loop_stack_.back()->exits.push_back(ArmState{control_, clone_vars(vars_)});
            control_ = vortex::ir::invalid_node;
            return {};
        }

        case StmtKind::Continue: {
            if (loop_stack_.empty()) {
                return fail_msg("lower: 'continue' outside loop",
                                diag_code::parse_unexpected_token);
            }
            loop_stack_.back()->backedges.push_back(ArmState{control_, clone_vars(vars_)});
            control_ = vortex::ir::invalid_node;
            return {};
        }

        case StmtKind::Global:
        case StmtKind::Nonlocal:
            return {};   // name analysis already accounted for these

        case StmtKind::Assert: {
            NodeId cond = VORTEX_TRY(lower_expr(s->cond));
            NodeId iff = g().create(NodeKind::If, {control_, cond});
            NodeId f = g().create(NodeKind::IfFalse, {iff});
            NodeId t = g().create(NodeKind::IfTrue, {iff});

            control_ = f;
            NodeId msg = s->test ? VORTEX_TRY(lower_expr(s->test)) : const_none();
            NodeId cls = effect_op(NodeKind::LoadGlobal, {}, true);
            g().node(cls).symbol = global_symbols().intern("AssertionError");
            NodeId instance = effect_op(NodeKind::CallPy, {cls, msg}, true);
            g().create(NodeKind::Throw, {control_, instance});
            control_ = vortex::ir::invalid_node;

            control_ = t;
            return {};
        }

        case StmtKind::Raise: {
            NodeId exc = const_none();
            if (s->value) exc = VORTEX_TRY(lower_expr(s->value));
            g().create(NodeKind::Throw, {control_, exc});
            control_ = vortex::ir::invalid_node;
            return {};
        }

        case StmtKind::Del: {
            for (Expr* t : s->targets) {
                if (t->kind == ExprKind::Name) {
                    vars_.erase(t->name);
                    deleted_locals_.insert(t->name, const_none());
                } else if (t->kind == ExprKind::Subscript) {
                    NodeId base = VORTEX_TRY(lower_expr(t->sub));
                    NodeId idx = VORTEX_TRY(lower_expr(t->index));
                    call_native(NativeHelper::DelSubscript, {base, idx}, true);
                } else if (t->kind == ExprKind::Attribute) {
                    NodeId base = VORTEX_TRY(lower_expr(t->sub));
                    call_native(NativeHelper::DelAttr, {base, const_int(t->attr)}, true);
                }
            }
            return {};
        }

        case StmtKind::Import: {
            NodeId modname = const_symbol_str(s->name);
            NodeId module = call_native(NativeHelper::ImportModule, {modname}, true);
            if (s->names.empty()) {
                write_var(s->name, module);
            } else {
                for (SymbolId n : s->names) {
                    NodeId attr = effect_op(NodeKind::LoadAttr, {module}, true);
                    g().node(attr).symbol = n;
                    write_var(n, attr);
                }
            }
            return {};
        }
    }
    return fail_msg("lower: unhandled statement kind", diag_code::parse_unexpected_token);
}

Result<NodeId> Lowerer::lower_assign_target(Expr* target, NodeId value) noexcept {
    switch (target->kind) {
        case ExprKind::Name:
            write_var(target->name, value);
            return value;
        case ExprKind::Attribute: {
            NodeId base = VORTEX_TRY(lower_expr(target->sub));
            NodeId n = effect_op(NodeKind::StoreAttr, {base, value});
            g().node(n).symbol = target->attr;
            return value;
        }
        case ExprKind::Subscript: {
            NodeId base = VORTEX_TRY(lower_expr(target->sub));
            NodeId idx = VORTEX_TRY(lower_expr(target->index));
            effect_op(NodeKind::StoreIndex, {base, idx, value});
            return value;
        }
        case ExprKind::TupleLit:
        case ExprKind::ListLit: {
            NodeId n = const_int(static_cast<std::int64_t>(target->args.size()));
            NodeId packed = call_native(NativeHelper::UnpackSequence, {value, n}, true);
            for (std::uint32_t i = 0; i < target->args.size(); ++i) {
                NodeId elem = effect_op(NodeKind::LoadIndex, {packed, const_int(i)}, true);
                VORTEX_TRY(lower_assign_target(target->args[i], elem));
            }
            return value;
        }
        default:
            return fail_msg("lower: invalid assignment target",
                            diag_code::parse_unexpected_token);
    }
}

// ---------------------------------------------------------------------------
// If
// ---------------------------------------------------------------------------
Result<void> Lowerer::lower_if(Stmt* s) noexcept {
    NodeId cond = VORTEX_TRY(lower_expr(s->cond));
    NodeId iff = g().create(NodeKind::If, {control_, cond});
    NodeId t = g().create(NodeKind::IfTrue, {iff});
    NodeId f = g().create(NodeKind::IfFalse, {iff});

    control_ = t;
    VarMap entry = clone_vars(vars_);
    VORTEX_TRY_VOID(lower_stmts(s->body));
    ArmState then_arm{control_, clone_vars(vars_)};

    vars_ = entry;
    control_ = f;
    VORTEX_TRY_VOID(lower_stmts(s->orelse));
    ArmState else_arm{control_, clone_vars(vars_)};

    stdx::small_vector<ArmState, 4> arms;
    arms.push_back(std::move(then_arm));
    arms.push_back(std::move(else_arm));
    merge_arms(arms);
    return {};
}

// ---------------------------------------------------------------------------
// While / For
// ---------------------------------------------------------------------------
Result<void> Lowerer::lower_while(Stmt* s) noexcept {
    NodeId loop = g().create(NodeKind::Loop);
    g().add_input(loop, control_);
    g().add_input(loop, loop);

    LoopSkeleton sk;
    sk.loop = loop;
    VarMap entry = clone_vars(vars_);
    for (auto& kv : entry) {
        NodeId phi = g().create(NodeKind::Phi);
        g().add_input(phi, kv.second);
        g().add_input(phi, phi);
        g().add_input(phi, loop);
        vars_.insert(kv.first, phi);
        sk.header_phis.push_back({kv.first, phi});
    }
    NodeId eff_phi = g().create(NodeKind::EffectPhi);
    g().add_input(eff_phi, memory_);
    g().add_input(eff_phi, eff_phi);
    g().add_input(eff_phi, loop);
    memory_ = eff_phi;
    sk.eff_phi = eff_phi;

    LoopCtx ctx;
    loop_stack_.push_back(&ctx);

    control_ = loop;
    NodeId cond = VORTEX_TRY(lower_expr(s->cond));
    NodeId iff = g().create(NodeKind::If, {loop, cond});
    NodeId t = g().create(NodeKind::IfTrue, {iff});
    NodeId f = g().create(NodeKind::IfFalse, {iff});

    control_ = t;
    VORTEX_TRY_VOID(lower_stmts(s->body));
    if (control_ != vortex::ir::invalid_node) {
        ctx.backedges.push_back(ArmState{control_, clone_vars(vars_)});
    }

    control_ = f;
    if (!s->orelse.empty()) {
        VORTEX_TRY_VOID(lower_stmts(s->orelse));
    }
    if (control_ != vortex::ir::invalid_node) {
        ctx.exits.push_back(ArmState{control_, clone_vars(vars_)});
    }
    loop_stack_.pop_back();

    patch_loop(sk, ctx);
    return {};
}

Result<void> Lowerer::lower_for(Stmt* s) noexcept {
    NodeId iterable = VORTEX_TRY(lower_expr(s->iter));
    NodeId it = effect_op(NodeKind::Iter, {iterable}, true);

    NodeId loop = g().create(NodeKind::Loop);
    g().add_input(loop, control_);
    g().add_input(loop, loop);

    LoopSkeleton sk;
    sk.loop = loop;
    VarMap entry = clone_vars(vars_);
    for (auto& kv : entry) {
        NodeId phi = g().create(NodeKind::Phi);
        g().add_input(phi, kv.second);
        g().add_input(phi, phi);
        g().add_input(phi, loop);
        vars_.insert(kv.first, phi);
        sk.header_phis.push_back({kv.first, phi});
    }
    NodeId eff_phi = g().create(NodeKind::EffectPhi);
    g().add_input(eff_phi, memory_);
    g().add_input(eff_phi, eff_phi);
    g().add_input(eff_phi, loop);
    memory_ = eff_phi;
    sk.eff_phi = eff_phi;

    LoopCtx ctx;
    loop_stack_.push_back(&ctx);

    control_ = loop;
    NodeId more = effect_op(NodeKind::GetIterCheck, {it}, false);
    NodeId iff = g().create(NodeKind::If, {loop, more});
    NodeId t = g().create(NodeKind::IfTrue, {iff});
    NodeId f = g().create(NodeKind::IfFalse, {iff});

    control_ = t;
    NodeId value = effect_op(NodeKind::IterNext, {it}, false);
    VORTEX_TRY_VOID(lower_assign_target(s->for_target, value));
    VORTEX_TRY_VOID(lower_stmts(s->body));
    if (control_ != vortex::ir::invalid_node) {
        ctx.backedges.push_back(ArmState{control_, clone_vars(vars_)});
    }

    control_ = f;
    if (!s->orelse.empty()) {
        VORTEX_TRY_VOID(lower_stmts(s->orelse));
    }
    if (control_ != vortex::ir::invalid_node) {
        ctx.exits.push_back(ArmState{control_, clone_vars(vars_)});
    }
    loop_stack_.pop_back();

    patch_loop(sk, ctx);
    return {};
}

void Lowerer::patch_loop(const LoopSkeleton& sk, LoopCtx& ctx) noexcept {
    if (ctx.backedges.empty()) {
        // Backedge never taken: leave self-edges (loop is a straight line).
        for (auto& [name, phi] : sk.header_phis) {
            Node& ph = g().node(phi);
            if (ph.ins.size() >= 3) g().set_input(phi, 1, phi);
        }
        Node& ep = g().node(sk.eff_phi);
        if (ep.ins.size() >= 3) g().set_input(sk.eff_phi, 1, memory_);
    } else {
        NodeId backedge_ctrl = vortex::ir::invalid_node;
        VarMap backedge_vars = ctx.backedges[0].vars;
        if (ctx.backedges.size() == 1) {
            backedge_ctrl = ctx.backedges[0].control;
        } else {
            backedge_ctrl = g().create(NodeKind::Region);
            for (ArmState& arm : ctx.backedges) g().add_input(backedge_ctrl, arm.control);
            // merge vars across multiple backedges
            for (std::size_t i = 1; i < ctx.backedges.size(); ++i) {
                for (auto& kv : ctx.backedges[i].vars) {
                    if (NodeId* cur = backedge_vars.get(kv.first)) {
                        if (*cur != kv.second) {
                            NodeId phi = g().create(NodeKind::Phi);
                            g().add_input(phi, *cur);
                            g().add_input(phi, kv.second);
                            g().add_input(phi, backedge_ctrl);
                            backedge_vars.insert(kv.first, phi);
                        }
                    } else {
                        backedge_vars.insert(kv.first, kv.second);
                    }
                }
            }
        }
        g().set_input(sk.loop, 1, backedge_ctrl);
        for (auto& [name, phi] : sk.header_phis) {
            NodeId* v = backedge_vars.get(name);
            g().set_input(phi, 1, v ? *v : phi);
        }
        g().set_input(sk.eff_phi, 1, memory_);
    }

    // Merge exits into the post-loop state.
    stdx::small_vector<ArmState, 4> arms = std::move(ctx.exits);
    merge_arms(arms);
}

// ---------------------------------------------------------------------------
// Try / except / finally
// ---------------------------------------------------------------------------
Result<void> Lowerer::lower_try(Stmt* s) noexcept {
    stdx::small_vector<ArmState, 8> snapshots;
    try_snapshots_ = &snapshots;
    VORTEX_TRY_VOID(lower_stmts_tracked(s->body));
    try_snapshots_ = nullptr;

    ArmState normal{control_, clone_vars(vars_)};

    if (!s->handlers.empty()) {
        // Catch region merging every statement-level raise point.
        NodeId catch_region = g().create(NodeKind::Catch);
        stdx::small_vector<ArmState, 8> catch_arms;
        for (ArmState& snap : snapshots) {
            if (snap.control != vortex::ir::invalid_node) {
                g().add_input(catch_region, snap.control);
                catch_arms.push_back(std::move(snap));
            }
        }

        control_ = catch_region;
        // Values visible on the catch path = phi over snapshots.
        VarMap catch_vars;
        if (!catch_arms.empty()) {
            catch_vars = catch_arms[0].vars;
            for (std::size_t i = 1; i < catch_arms.size(); ++i) {
                for (auto& kv : catch_arms[i].vars) {
                    if (NodeId* cur = catch_vars.get(kv.first)) {
                        if (*cur != kv.second) {
                            NodeId phi = g().create(NodeKind::Phi);
                            g().add_input(phi, *cur);
                            g().add_input(phi, kv.second);
                            g().add_input(phi, catch_region);
                            catch_vars.insert(kv.first, phi);
                        }
                    } else {
                        catch_vars.insert(kv.first, kv.second);
                    }
                }
            }
        }
        vars_ = std::move(catch_vars);

        NodeId exc_value = call_native(NativeHelper::GetCurrentException, {}, false);

        NodeId matched_exit = vortex::ir::invalid_node;
        VarMap matched_vars{};
        NodeId unmatched_ctrl = control_;
        VarMap unmatched_vars = clone_vars(vars_);

        for (ExceptClause& h : s->handlers) {
            if (unmatched_ctrl == vortex::ir::invalid_node) break;
            control_ = unmatched_ctrl;
            vars_ = unmatched_vars;
            NodeId matches = const_int(1);
            if (h.type_name != sym_invalid) {
                NodeId tyname = const_symbol_str(h.type_name);
                matches = call_native(NativeHelper::IsInstance, {exc_value, tyname}, false);
            }
            NodeId iff = g().create(NodeKind::If, {control_, matches});
            NodeId t = g().create(NodeKind::IfTrue, {iff});
            NodeId f = g().create(NodeKind::IfFalse, {iff});

            control_ = t;
            if (h.bind_name != sym_invalid) {
                write_var(h.bind_name, exc_value);
            }
            VORTEX_TRY_VOID(lower_stmts(h.body));
            if (matched_exit == vortex::ir::invalid_node) {
                matched_exit = control_;
                matched_vars = clone_vars(vars_);
            } else if (control_ != vortex::ir::invalid_node) {
                NodeId region = g().create(NodeKind::Region);
                g().add_input(region, matched_exit);
                g().add_input(region, control_);
                for (auto& kv : vars_) {
                    if (NodeId* old = matched_vars.get(kv.first)) {
                        if (*old != kv.second) {
                            NodeId phi = g().create(NodeKind::Phi);
                            g().add_input(phi, *old);
                            g().add_input(phi, kv.second);
                            g().add_input(phi, region);
                            matched_vars.insert(kv.first, phi);
                        }
                    }
                }
                matched_exit = region;
            }
            unmatched_ctrl = f;
        }

        if (unmatched_ctrl != vortex::ir::invalid_node) {
            control_ = unmatched_ctrl;
            g().create(NodeKind::Throw, {control_, exc_value});
        }

        stdx::small_vector<ArmState, 4> arms;
        arms.push_back(std::move(normal));
        arms.push_back(ArmState{matched_exit, matched_vars});
        merge_arms(arms);
    } else {
        control_ = normal.control;
        vars_ = normal.vars;
    }

    if (!s->finalbody.empty() && control_ != vortex::ir::invalid_node) {
        // finally on the merged path; the rethrow path already carries it via
        // the handler chain (documented subset deviation for return-in-finally).
        VORTEX_TRY_VOID(lower_stmts(s->finalbody));
    }
    return {};
}

// ---------------------------------------------------------------------------
// def / class
// ---------------------------------------------------------------------------
Result<void> Lowerer::lower_function_def(Stmt* s) noexcept {
    // Transitive captures: child free names ∩ my bound names.
    stdx::small_vector<SymbolId, 16> child_free = free_names(s->body);
    stdx::small_vector<SymbolId, 8> captures;
    for (SymbolId f : child_free) {
        bool is_param = false;
        for (SymbolId p : s->params) {
            if (p == f) { is_param = true; break; }
        }
        if (is_param) continue;
        if (contains_sym(bound_names_saved_, f)) {
            if (!contains_sym8(captures, f)) captures.push_back(f);
        }
    }

    PendingFunction pf{s->name, s, ctx_.next_code_unit_id};
    unit_.children.push_back(pf);

    stdx::small_vector<NodeId, 4> default_values;
    for (Expr* d : s->defaults) {
        default_values.push_back(VORTEX_TRY(lower_expr(d)));
    }

    // cells tuple (shared references — late binding preserved)
    NodeId cells = const_none();
    if (!captures.empty()) {
        cells = g().create(NodeKind::NewTuple);
        Node& tn = g().node(cells);
        tn.set_flag(NodeFlag::OnEffectChain);
        g().add_input(cells, control_);
        g().add_input(cells, memory_);
        for (SymbolId c : captures) {
            NodeId cell = cell_of(c);
            g().add_input(cells, cell);
        }
        memory_ = cells;
    }

    NodeId fn = g().create(NodeKind::CallNative);
    Node& fnode = g().node(fn);
    fnode.subop = static_cast<std::uint16_t>(NativeHelper::MakeFunction);
    fnode.set_flag(NodeFlag::OnEffectChain);
    g().add_input(fn, control_);
    g().add_input(fn, memory_);
    g().add_input(fn, const_int(pf.code_unit_hint));
    g().add_input(fn, const_int(static_cast<std::int64_t>(default_values.size())));
    for (NodeId d : default_values) g().add_input(fn, d);
    g().add_input(fn, cells);
    memory_ = fn;

    write_var(s->name, fn);
    return {};
}

Result<void> Lowerer::lower_class_def(Stmt* s) noexcept {
    // class body is its own unit returning the namespace dict
    PendingFunction pf{s->name, s, ctx_.next_code_unit_id};
    unit_.children.push_back(pf);

    NodeId fn = g().create(NodeKind::CallNative);
    Node& fnode = g().node(fn);
    fnode.subop = static_cast<std::uint16_t>(NativeHelper::MakeFunction);
    fnode.set_flag(NodeFlag::OnEffectChain);
    g().add_input(fn, control_);
    g().add_input(fn, memory_);
    g().add_input(fn, const_int(pf.code_unit_hint));
    g().add_input(fn, const_int(0));
    g().add_input(fn, const_none());
    memory_ = fn;

    NodeId ns = effect_op(NodeKind::CallPy, {fn}, true);

    NodeId base = const_none();
    if (s->base_name != sym_invalid) {
        base = effect_op(NodeKind::LoadGlobal, {}, true);
        g().node(base).symbol = s->base_name;
    }

    NodeId cls = call_native(NativeHelper::MakeClass,
                             {const_symbol_str(s->name), ns, base}, false);
    write_var(s->name, cls);
    return {};
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------
Result<NodeId> Lowerer::lower_expr(Expr* e) noexcept {
    if (++depth_ > max_lower_depth) {
        return fail_msg("lower: expression nesting too deep",
                        diag_code::parse_unexpected_token);
    }
    struct DepthGuard {
        Lowerer* l;
        ~DepthGuard() { --(l->depth_); }
    } guard{this};

    switch (e->kind) {
        case ExprKind::IntLit: return const_int(e->int_value);
        case ExprKind::FloatLit: return const_float(e->float_value);
        case ExprKind::StrLit: return const_string(e->str_offset, e->str_length);
        case ExprKind::BoolLit: {
            NodeId n = g().create(NodeKind::ConstPy);
            g().node(n).const_value = Value::boolean(e->bool_value);
            g().node(n).set_flag(NodeFlag::Pure);
            return n;
        }
        case ExprKind::NoneLit: return const_none();
        case ExprKind::Name: return read_var(e->name);

        case ExprKind::BinOp:
            return py_op(NodeKind::PyBinary, e->op,
                         {VORTEX_TRY(lower_expr(e->args[0])),
                          VORTEX_TRY(lower_expr(e->args[1]))});

        case ExprKind::UnaryOp: {
            NodeId v = VORTEX_TRY(lower_expr(e->sub));
            return py_op(NodeKind::PyUnary, e->op, {v});
        }

        case ExprKind::BoolOp:
            return lower_boolop(e);

        case ExprKind::Compare:
            return lower_compare_chain(e);

        case ExprKind::Call:
            return lower_call(e);

        case ExprKind::Attribute: {
            NodeId base = VORTEX_TRY(lower_expr(e->sub));
            NodeId n = effect_op(NodeKind::LoadAttr, {base}, true);
            g().node(n).symbol = e->attr;
            return n;
        }

        case ExprKind::Subscript: {
            NodeId base = VORTEX_TRY(lower_expr(e->sub));
            NodeId idx = VORTEX_TRY(lower_expr(e->index));
            return effect_op(NodeKind::LoadIndex, {base, idx}, true);
        }

        case ExprKind::ListLit: {
            NodeId n = g().create(NodeKind::NewList);
            Node& ln = g().node(n);
            ln.set_flag(NodeFlag::OnEffectChain);
            ln.set_flag(NodeFlag::MayThrow);
            g().add_input(n, control_);
            g().add_input(n, memory_);
            for (Expr* el : e->args) {
                g().add_input(n, VORTEX_TRY(lower_expr(el)));
            }
            memory_ = n;
            return n;
        }

        case ExprKind::TupleLit: {
            NodeId n = g().create(NodeKind::NewTuple);
            Node& tn = g().node(n);
            tn.set_flag(NodeFlag::OnEffectChain);
            tn.set_flag(NodeFlag::MayThrow);
            g().add_input(n, control_);
            g().add_input(n, memory_);
            for (Expr* el : e->args) {
                g().add_input(n, VORTEX_TRY(lower_expr(el)));
            }
            memory_ = n;
            return n;
        }

        case ExprKind::DictLit: {
            NodeId n = g().create(NodeKind::NewDict);
            Node& dn = g().node(n);
            dn.set_flag(NodeFlag::OnEffectChain);
            dn.set_flag(NodeFlag::MayThrow);
            g().add_input(n, control_);
            g().add_input(n, memory_);
            for (std::uint32_t i = 0; i + 1 < e->args.size(); i += 2) {
                g().add_input(n, VORTEX_TRY(lower_expr(e->args[i])));
                g().add_input(n, VORTEX_TRY(lower_expr(e->args[i + 1])));
            }
            memory_ = n;
            return n;
        }

        case ExprKind::ListComp:
            return lower_listcomp(e);

        case ExprKind::Lambda: {
            // Build an implicit child function `lambda`: params + body.
            // We synthesize a Stmt* def on the module arena.
            Stmt* def = mod_.make<Stmt>();
            def->kind = StmtKind::FunctionDef;
            def->name = global_symbols().intern("<lambda>");
            def->line = e->line;
            for (SymbolId p : e->lambda_params) def->params.push_back(p);
            Stmt* ret = mod_.make<Stmt>();
            ret->kind = StmtKind::Return;
            ret->line = e->line;
            ret->value = e->sub;
            def->body.push_back(ret);
            // reuse the function-def machinery via a synthetic statement
            Lowerer* self = this;
            // (lower_function_def operates on this->bound_names_saved_)
            return [self, def]() -> Result<NodeId> {
                VORTEX_TRY_VOID(self->lower_function_def(def));
                // value = the just-bound function name
                return self->read_var(def->name);
            }();
        }

        case ExprKind::IfExp: {
            NodeId cond = VORTEX_TRY(lower_expr(e->lower));
            NodeId iff = g().create(NodeKind::If, {control_, cond});
            NodeId t = g().create(NodeKind::IfTrue, {iff});
            NodeId f = g().create(NodeKind::IfFalse, {iff});

            control_ = t;
            NodeId tv = VORTEX_TRY(lower_expr(e->sub));
            ArmState then_arm{control_, clone_vars(vars_)};

            control_ = f;
            NodeId fv = VORTEX_TRY(lower_expr(e->upper));
            ArmState else_arm{control_, clone_vars(vars_)};

            stdx::small_vector<ArmState, 4> arms;
            arms.push_back(std::move(then_arm));
            arms.push_back(std::move(else_arm));
            NodeId region = merge_arms(arms);
            if (region == vortex::ir::invalid_node) {
                return fail_msg("lower: both conditional-expression arms exited",
                                diag_code::parse_unexpected_token);
            }
            NodeId phi = g().create(NodeKind::Phi);
            g().add_input(phi, tv);
            g().add_input(phi, fv);
            g().add_input(phi, region);
            return phi;
        }

        case ExprKind::Yield: {
            unit_.is_generator = true;
            NodeId v = const_none();
            if (e->sub) v = VORTEX_TRY(lower_expr(e->sub));
            return effect_op(NodeKind::Yield, {v});
        }
    }
    return fail_msg("lower: unhandled expression kind", diag_code::parse_unexpected_token);
}

Result<NodeId> Lowerer::lower_boolop(Expr* e) noexcept {
    // Short-circuit via control flow; result is the last evaluated operand
    // (Python semantics: `a or b` yields a if truthy else b).
    const bool is_or = e->op == bool_or;
    NodeId lhs = VORTEX_TRY(lower_expr(e->args[0]));
    NodeId rhs = VORTEX_TRY(lower_expr(e->args[1]));

    NodeId cond = py_op(NodeKind::PyUnary, is_or ? 4 : 5, {lhs});   // truth-test op
    NodeId iff = g().create(NodeKind::If, {control_, cond});
    NodeId t = g().create(NodeKind::IfTrue, {iff});
    NodeId f = g().create(NodeKind::IfFalse, {iff});

    // or: True -> lhs, False -> rhs. and: True -> rhs, False -> lhs.
    NodeId shortv = is_or ? lhs : lhs;
    NodeId longv = rhs;

    control_ = is_or ? t : f;
    ArmState short_arm{control_, clone_vars(vars_)};
    control_ = is_or ? f : t;
    ArmState long_arm{control_, clone_vars(vars_)};

    stdx::small_vector<ArmState, 4> arms;
    arms.push_back(std::move(short_arm));
    arms.push_back(std::move(long_arm));
    NodeId region = merge_arms(arms);
    NodeId phi = g().create(NodeKind::Phi);
    g().add_input(phi, shortv);
    g().add_input(phi, longv);
    g().add_input(phi, region);
    return phi;
}

Result<NodeId> Lowerer::lower_compare_chain(Expr* e) noexcept {
    if (e->cmp_ops.size() == 1) {
        NodeId lhs = VORTEX_TRY(lower_expr(e->args[0]));
        NodeId rhs = VORTEX_TRY(lower_expr(e->args[1]));
        return py_op(NodeKind::PyCompare, e->cmp_ops[0], {lhs, rhs});
    }
    // a op1 b op2 c  ==  (a op1 b) and (b op2 c) with short-circuit.
    NodeId lhs = VORTEX_TRY(lower_expr(e->args[0]));
    NodeId result = const_int(1);
    for (std::uint32_t i = 0; i < e->cmp_ops.size(); ++i) {
        NodeId rhs = VORTEX_TRY(lower_expr(e->args[i + 1]));
        NodeId cmp = py_op(NodeKind::PyCompare, e->cmp_ops[i], {lhs, rhs});
        if (i == 0) {
            result = cmp;
        } else {
            NodeId cond = py_op(NodeKind::PyUnary, 5, {result});   // truth-test
            NodeId iff = g().create(NodeKind::If, {control_, cond});
            NodeId t = g().create(NodeKind::IfTrue, {iff});
            NodeId f = g().create(NodeKind::IfFalse, {iff});
            control_ = t;
            NodeId short_false = const_int(0);
            ArmState short_arm{control_, clone_vars(vars_)};
            control_ = f;
            ArmState long_arm{control_, clone_vars(vars_)};
            stdx::small_vector<ArmState, 4> arms;
            arms.push_back(std::move(short_arm));
            arms.push_back(std::move(long_arm));
            NodeId region = merge_arms(arms);
            NodeId phi = g().create(NodeKind::Phi);
            g().add_input(phi, short_false);
            g().add_input(phi, cmp);
            g().add_input(phi, region);
            result = phi;
        }
        lhs = rhs;
    }
    return result;
}

Result<NodeId> Lowerer::lower_call(Expr* e) noexcept {
    NodeId callee = VORTEX_TRY(lower_expr(e->sub));
    stdx::small_vector<NodeId, 8> arg_values;
    for (Argument& a : e->call_args) {
        arg_values.push_back(VORTEX_TRY(lower_expr(a.value)));
    }
    NodeId n = g().create(NodeKind::CallPy);
    Node& cn = g().node(n);
    cn.set_flag(NodeFlag::OnEffectChain);
    cn.set_flag(NodeFlag::MayThrow);
    cn.set_flag(NodeFlag::MayCall);
    g().add_input(n, control_);
    g().add_input(n, memory_);
    g().add_input(n, callee);
    for (NodeId v : arg_values) g().add_input(n, v);
    // keyword layout: ConstPy payload carries kw count; kw symbols packed in
    // node symbols list — we encode kw args as trailing (symbol, value) pairs
    // with aux0 = positional count. Runtime reads aux0; pairs follow.
    std::uint32_t positional = 0;
    std::uint32_t flags = 0;
    for (Argument& a : e->call_args) {
        if (a.keyword == arg_invalid) ++positional;
        else if (a.keyword == arg_star) flags |= 1;
        else if (a.keyword == arg_kwargs) flags |= 2;
    }
    cn.aux0 = positional;
    cn.aux1 = flags;
    // keyword names: stored in a trailing ConstPy tuple node
    std::uint32_t kw_count = 0;
    for (Argument& a : e->call_args) {
        if (a.keyword != arg_invalid && a.keyword != arg_star && a.keyword != arg_kwargs) {
            ++kw_count;
        }
    }
    if (kw_count > 0 || (flags & 3)) {
        NodeId kwn = g().create(NodeKind::NewTuple);
        Node& kn = g().node(kwn);
        kn.set_flag(NodeFlag::OnEffectChain);
        g().add_input(kwn, control_);
        g().add_input(kwn, memory_);
        for (Argument& a : e->call_args) {
            if (a.keyword != arg_invalid && a.keyword != arg_star && a.keyword != arg_kwargs) {
                g().add_input(kwn, const_int(a.keyword));
            }
        }
        g().add_input(n, kwn);
        memory_ = n;
    }
    memory_ = n;
    return n;
}

Result<NodeId> Lowerer::lower_listcomp(Expr* e) noexcept {
    // [elt for target in iter (if cond)] -> loop appending to a new list.
    // Genexps use the same lowering (documented eager-materialization).
    NodeId result = effect_op(NodeKind::NewList, {}, true);

    NodeId iterable = VORTEX_TRY(lower_expr(e->comp.iter));
    NodeId it = effect_op(NodeKind::Iter, {iterable}, true);

    NodeId loop = g().create(NodeKind::Loop);
    g().add_input(loop, control_);
    g().add_input(loop, loop);

    LoopSkeleton sk;
    sk.loop = loop;
    VarMap entry = clone_vars(vars_);
    for (auto& kv : entry) {
        NodeId phi = g().create(NodeKind::Phi);
        g().add_input(phi, kv.second);
        g().add_input(phi, phi);
        g().add_input(phi, loop);
        vars_.insert(kv.first, phi);
        sk.header_phis.push_back({kv.first, phi});
    }
    NodeId eff_phi = g().create(NodeKind::EffectPhi);
    g().add_input(eff_phi, memory_);
    g().add_input(eff_phi, eff_phi);
    g().add_input(eff_phi, loop);
    memory_ = eff_phi;
    sk.eff_phi = eff_phi;

    LoopCtx ctx;
    loop_stack_.push_back(&ctx);

    control_ = loop;
    NodeId more = effect_op(NodeKind::GetIterCheck, {it}, false);
    NodeId iff = g().create(NodeKind::If, {loop, more});
    NodeId t = g().create(NodeKind::IfTrue, {iff});
    NodeId f = g().create(NodeKind::IfFalse, {iff});

    control_ = t;
    NodeId value = effect_op(NodeKind::IterNext, {it}, false);
    VORTEX_TRY_VOID(lower_assign_target(e->comp.target, value));

    NodeId append = VORTEX_TRY(lower_expr(e->args[0]));
    if (e->comp.cond) {
        NodeId c = VORTEX_TRY(lower_expr(e->comp.cond));
        NodeId cif = g().create(NodeKind::If, {control_, c});
        NodeId ct = g().create(NodeKind::IfTrue, {cif});
        NodeId cf = g().create(NodeKind::IfFalse, {cif});
        control_ = ct;
        effect_op(NodeKind::ListAppend, {result, append}, true);
        stdx::small_vector<ArmState, 4> arms;
        arms.push_back(ArmState{control_, clone_vars(vars_)});
        arms.push_back(ArmState{cf, clone_vars(vars_)});
        merge_arms(arms);
    } else {
        effect_op(NodeKind::ListAppend, {result, append}, true);
    }

    if (control_ != vortex::ir::invalid_node) {
        ctx.backedges.push_back(ArmState{control_, clone_vars(vars_)});
    }
    control_ = f;
    ctx.exits.push_back(ArmState{control_, clone_vars(vars_)});
    loop_stack_.pop_back();

    patch_loop(sk, ctx);
    return result;
}

}  // namespace

// ---------------------------------------------------------------------------
// Public entry: lower one unit.
// ---------------------------------------------------------------------------
Result<LoweredUnit> lower_unit(Module& module, LowerContext& ctx, Stmt* def, SymbolId name,
                               stdx::small_vector<SymbolId, 8>& captured_names,
                               bool class_body) noexcept {
    LoweredUnit unit;
    StmtList body_storage;
    if (!def) {
        for (Stmt* st : module.body) body_storage.push_back(st);
    }
    const StmtList& body = def ? def->body : body_storage;

    // Locals captured (transitively) by nested functions become cells.
    NameSets sets = analyze(body);
    stdx::small_vector<SymbolId, 8> cell_vars;
    for (SymbolId b : sets.bound) {
        if (contains_sym(sets.child_referenced, b)) {
            cell_vars.push_back(b);
        }
    }

    Lowerer lowerer(module, ctx, unit, def == nullptr, class_body, captured_names);
    lowerer.bound_names_saved_ = sets.bound;
    Result<void> ok = lowerer.run(def, body, cell_vars);
    if (!ok) return std::unexpected(ok.error());
    return unit;
}

}  // namespace abi_v1
}  // namespace vortex::fe
