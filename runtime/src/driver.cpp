// =============================================================================
// vortex/rt/driver.cpp — compile driver implementation.
// =============================================================================

#include "vortex/rt/driver.hpp"

#include <cstring>

#include "vortex/frontend/lowering.hpp"
#include "vortex/frontend/parser.hpp"
#include "vortex/passes/pass_pipeline.hpp"
#include "vortex/pipeline/scheduler.hpp"
#include "vortex/support/config.hpp"
#include "vortex/support/arena.hpp"
#include "vortex/support/symbol_table.hpp"

namespace vortex::rt {
inline namespace abi_v1 {

CompileOutcome compile_program(Vm& vm, std::string_view source,
                               SymbolId module_name) noexcept {
    CompileOutcome outcome;
    Runtime& rt = Runtime::instance();

    // Frontend: source -> AST (arena owns the tree for the whole program).
    BumpArena module_arena;
    Result<fe::Module*> ast = fe::compile_to_ast(module_arena, source);
    if (!ast) {
        outcome.diagnostic = ast.error();
        return outcome;
    }

    // Pass 1 lowering: BFS over units. The module toplevel is unit 1
    // (id 0 reserved). Children are lowered in DISCOVERY order so the
    // unit ids pre-assigned during the parent's lowering stay valid —
    // a LIFO stack scrambled them (the MakeFunction unit-id mismatch).
    fe::LowerContext lower_ctx;
    struct Pending {
        fe::Stmt* def;
        stdx::small_vector<SymbolId, 8> captures;
        bool class_body;
        std::uint32_t reserved_id{0xFFFFFFFF};
    };
    stdx::small_vector<Pending, 16> queue;
    {
        Pending p;
        p.def = nullptr;
        p.class_body = false;
        queue.push_back(p);
    }

    std::uint32_t module_unit_id = 0xFFFFFFFFu;
    std::size_t queue_head = 0;   // FIFO: discovery order == id order
    while (queue_head < queue.size()) {
        Pending p = queue[queue_head++];

        SymbolId name = p.def ? p.def->name : module_name;
        // Module toplevel self-assigns id 1; children use the id reserved
        // at discovery (p.def != null means the hint lives in p.reserved_id).
        Result<fe::LoweredUnit> unit_res =
            fe::lower_unit(**ast, lower_ctx, p.def, name, p.captures, p.class_body,
                           p.def ? p.reserved_id : 0xFFFFFFFF);
        if (!unit_res) {
            outcome.diagnostic = unit_res.error();
            return outcome;
        }
        fe::LoweredUnit& lowered = *unit_res;

        // Ensure the program's unit table is large enough; ids are assigned
        // by lowering in discovery order, so extend to match next_code_unit_id.
        while (vm.program.units.size() < lower_ctx.next_code_unit_id) {
            vm.program.units.push_back(nullptr);
        }

        // Run the optimizing pipeline (Tier 2 mode: unified passes,
        // budget-guarded) before scheduling. The passes only remove or
        // forward provably-dead/constant code, so execution of the
        // optimized graph must match the unoptimized one — the
        // differential test suite verifies this (Rule 36).
        {
            passes::PassContext pctx;
            pctx.tier = passes::TierMode::Tier2;
            pctx.node_budget = cfg::tier2_node_budget;
            pctx.code_unit_id = lowered.code_unit_id;
            pctx.telemetry = &vm.telemetry;
            Result<void> optimized = passes::optimize(lowered.graph, pctx);
            if (!optimized) {
                outcome.diagnostic = optimized.error();
                return outcome;
            }
        }

        // Schedule to Tier-0 bytecode.
        auto* cu = new CodeUnit();
        cu->id = lowered.code_unit_id;
        cu->has_varargs = lowered.has_varargs;
        cu->has_kwargs = lowered.has_kwargs;

        // Parameter registers: node ids of the Parameter nodes (aux0 order).
        stdx::small_vector<std::uint32_t, 8> param_regs;
        stdx::small_vector<SymbolId, 8> param_names;
        lowered.graph.for_each_live([&](ir::NodeId id) {
            const ir::Node& n = lowered.graph.node(id);
            if (n.kind != ir::NodeKind::Parameter) return;
            std::uint32_t idx = n.aux0;
            if (idx + 1 > param_regs.size()) param_regs.resize(idx + 1, 0);
            param_regs[idx] = id;
            if (lowered.param_names.size() > idx) {
                if (idx + 1 > param_names.size()) param_names.resize(idx + 1, 0xFFFF'FFFF);
                param_names[idx] = lowered.param_names[idx];
            }
        });

        Result<void> scheduled =
            pipeline::schedule_unit(lowered.graph, *cu, (**ast).string_pool, param_regs,
                                    param_names, lowered.is_generator);
        if (!scheduled) {
            delete cu;
            outcome.diagnostic = scheduled.error();
            return outcome;
        }
        vm.program.units[cu->id] = cu;
        if (p.def == nullptr) module_unit_id = cu->id;

        // Queue children (reverse order so the first child compiles first).
        for (std::size_t i = lowered.children.size(); i-- > 0;) {
            fe::PendingFunction& child = lowered.children[i];
            Pending cp;
            cp.def = child.def;
            cp.class_body = (child.def->kind == fe::StmtKind::ClassDef);
            cp.reserved_id = child.code_unit_hint;
            for (SymbolId cap : child.captures) cp.captures.push_back(cap);
            queue.push_back(cp);
        }
    }

    // Link check: every unit id has a CodeUnit (id 0 is reserved-unused).
    for (std::size_t i = 1; i < vm.program.units.size(); ++i) {
        if (!vm.program.units[i]) {
            Diagnostic d = Diagnostic::error("code unit not compiled during link",
                                             diag_code::runtime_value_error);
            d.actual = "missing unit";
            outcome.diagnostic = d;
            return outcome;
        }
    }

    outcome.ok = true;
    outcome.module_unit_id = module_unit_id;
    outcome.units_compiled = static_cast<std::uint32_t>(vm.program.units.size());
    return outcome;
}

Result<Value> run_source(Vm& vm, std::string_view source) noexcept {
    SymbolId module_name = global_symbols().intern("__main__");
    CompileOutcome outcome = compile_program(vm, source, module_name);
    if (!outcome.ok) {
        Diagnostic d = outcome.diagnostic;
        if (d.fix.empty()) {
            d.fix = "Fix the source line reported above (VORTEX Python subset)";
        }
        return fail(d);
    }
    CodeUnit* toplevel = outcome.module_unit_id < vm.program.units.size()
                             ? vm.program.units[outcome.module_unit_id]
                             : nullptr;
    if (!toplevel) {
        return fail_msg("no module toplevel unit", diag_code::runtime_value_error);
    }
    return vm.run_module(toplevel);
}

}  // namespace abi_v1
}  // namespace vortex::rt
