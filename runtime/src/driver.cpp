// =============================================================================
// vortex/rt/driver.cpp — compile driver implementation.
// =============================================================================

#include "vortex/rt/driver.hpp"

#include <cstring>
#include <sys/mman.h>
#include <unistd.h>

#include "vortex/backend/codegen.hpp"
#include "vortex/backend/target.hpp"
#include "vortex/frontend/lowering.hpp"
#include "vortex/frontend/parser.hpp"
#include "vortex/passes/pass_pipeline.hpp"
#include "vortex/pipeline/scheduler.hpp"
#include "vortex/support/config.hpp"
#include "vortex/support/arena.hpp"
#include "vortex/support/symbol_table.hpp"

namespace vortex::rt {
inline namespace abi_v1 {

namespace {
/// Task 24: page-aligned RWX buffer for JIT-compiled machine code.
/// Allocated via mmap; freed by CodeUnit's destructor via the
/// vortex_rt_munmap_jit_buffer shim (see runtime/src/jit.cpp).
/// Returns nullptr on failure (the driver logs and falls back to
/// Tier-0 execution for this unit).
[[nodiscard]] std::byte* make_jit_buffer(std::size_t bytes) noexcept {
    long pagesz = sysconf(_SC_PAGESIZE);
    if (pagesz <= 0) pagesz = 4096;
    std::size_t mapped = ((bytes + pagesz - 1) / pagesz) * pagesz;
    void* p = mmap(nullptr, mapped, PROT_READ | PROT_WRITE | PROT_EXEC,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return nullptr;
    return static_cast<std::byte*>(p);
}
}  // namespace

CompileOutcome compile_program(Vm& vm, std::string_view source,
                               SymbolId module_name,
                               const CompileOptions& options) noexcept {
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
            // Machine facts come from the probed host descriptor — never
            // from cfg constants (Rule 24/27).
            pctx.target = &backend::host_target();
            // Module string pool: the only source for string-literal
            // bytes (Rule 5). Passes that fold string constants (Pass 45)
            // read from this pool.
            pctx.string_pool = &(**ast).string_pool;
            // Opt-out switches (Pass 33 polyhedral is ON unless disabled).
            // The only legitimate reason to opt out is compilation-time
            // sensitivity (see p33_polyhedral.cpp header for rationale).
            if (options.disable_polyhedral) {
                pctx.options.set(passes::OptOption::DisablePolyhedral);
            }
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

        // Task 24: JIT-compile the unit eagerly. We attempt this for
        // EVERY unit (module toplevel included) and let the backend
        // decide via has_dynamic_ops whether the result is safe to
        // invoke from the CALL handler. Module-toplevel units rarely
        // JIT cleanly because they typically contain print()/CallPy
        // ops, but functions with provably-typed arithmetic DO.
        //
        // We don't gate on backedge_count/call_count yet — that's the
        // tiering trigger work (Path A v2). For now: every unit gets
        // JIT'd at compile time. The CALL handler falls back to
        // Tier-0 (exec_frame) when has_dynamic_ops is true.
        //
        // The IR graph (lowered.graph) is alive for the duration of
        // this loop iteration; compile_unit takes a const ref so it
        // can be safely borrowed here.
        if (!lowered.is_generator) {
            constexpr std::size_t kJitCodeCapacity = 64 * 1024;  // 64 KB per unit
            std::byte* jit_buf = make_jit_buffer(kJitCodeCapacity);
            if (jit_buf) {
                backend::CompiledCode cc =
                    backend::compile_unit(lowered.graph, cu->id, jit_buf,
                                          kJitCodeCapacity,
                                          backend::host_target());
                if (cc.valid) {
                    // Install: the function pointer is the start of
                    // the buffer (the codegen emits a prologue at
                    // offset 0). Store the buffer/capacity so the
                    // CodeUnit destructor can munmap it.
                    cu->jit_code_buffer = jit_buf;
                    cu->jit_code_capacity = cc.code_size > 0
                                               ? cc.code_size
                                               : kJitCodeCapacity;
                    cu->jit_entry.store(jit_buf, std::memory_order_release);
                    cu->current_tier.store(1, std::memory_order_release);
                    cu->has_dynamic_ops = cc.has_dynamic_ops;
                    // Task 24: the JIT's home slots are IR NodeIds
                    // (per backend/lowering.cpp: `std::uint32_t home = id;`).
                    // The scheduler's `n_registers` is set independently
                    // based on what IT considers live — for parsed while
                    // loops with Phis the JIT's frame_slots (max NodeId+1)
                    // can exceed n_registers, and the JIT would write past
                    // the regs array. Bump n_registers to cover.
                    if (cc.frame_slots > cu->n_registers) {
                        cu->n_registers = cc.frame_slots;
                    }
                    // Task 24 (candidate j): record each Phi's entry
                    // value so the CALL handler can initialize the
                    // home slot before calling jit_entry. Without
                    // this, the JIT would read Value::none() from Phi
                    // slots (the Frame constructor fills with none();
                    // only bind_parameters populates Parameter slots,
                    // never Phi slots) and every GUARD_INT / GUARD_FLOAT
                    // would fail on the first read.
                    //
                    // Handles ConstInt (-> Value::integer) and ConstFloat
                    // (-> Value::real, via bit-reinterpret of the int64
                    // transport). Parameters / arbitrary entry values
                    // need a richer table (future task).
                    //
                    // Task 24 (candidate j): the backend now emits the
                    // backedge MOVrr at the JUMP site (see
                    // backend/lowering.cpp's Phi backedge emission),
                    // so the Phi home slot is correctly updated at the
                    // end of each loop iteration. The previous
                    // has_loop_phis downgrade (force has_dynamic_ops =
                    // true to skip jit_entry) is REMOVED — parsed
                    // while loops with ConstInt/ConstFloat entry Phis
                    // now JIT correctly.
                    lowered.graph.for_each_live([&](ir::NodeId id) {
                        const ir::Node& n = lowered.graph.node(id);
                        if (n.kind != ir::NodeKind::Phi) return;
                        if (n.ins.size() < 2) return;
                        const ir::Node& entry = lowered.graph.node(n.ins[0]);
                        if (entry.kind == ir::NodeKind::ConstInt) {
                            cu->phi_init_node_ids.push_back(
                                static_cast<std::uint32_t>(id));
                            cu->phi_init_values.push_back(
                                entry.const_value.as.i);
                            cu->phi_init_is_float.push_back(0);
                        } else if (entry.kind == ir::NodeKind::ConstFloat) {
                            cu->phi_init_node_ids.push_back(
                                static_cast<std::uint32_t>(id));
                            // Bit-reinterpret the double as int64 for
                            // transport; the CALL handler reinterprets
                            // back to double when writing Value::real.
                            std::int64_t bits;
                            std::memcpy(&bits, &entry.const_value.as.f,
                                        sizeof(double));
                            cu->phi_init_values.push_back(bits);
                            cu->phi_init_is_float.push_back(1);
                        }
                        // Else: Parameter / arbitrary entry — phi_init
                        // doesn't handle yet (the entry value depends
                        // on runtime context). Such Phis would deopt
                        // on first read; the unit either runs Tier-0
                        // (via has_dynamic_ops from elsewhere) or the
                        // deopt path handles it (once safepoint_pcs
                        // is populated — candidate k).
                    });
                    // Task 24: safepoint_pcs population is NOT yet
                    // wired (would need a NodeId→Tier-0 PC map from
                    // the scheduler). For units with
                    // has_dynamic_ops == false, no CALLri exists and
                    // no bridge call can fire, so safepoint_pcs
                    // stays empty safely. For units with
                    // has_dynamic_ops == true, the CALL handler
                    // skips jit_entry anyway (falls back to Tier-0).
                    // Future task: populate safepoint_pcs so units
                    // with dynamic ops can also be JIT'd.
                } else {
                    // compile_unit reported the graph couldn't be
                    // lowered (likely contains ops the backend's MIR
                    // table doesn't handle yet, or the codegen ran
                    // out of buffer space). Fall back to Tier-0;
                    // munmap the unused buffer.
                    vortex_rt_munmap_jit_buffer(jit_buf, kJitCodeCapacity);
                }
            }
            // If make_jit_buffer returned nullptr (mmap failed), the
            // unit runs Tier-0 only — no JIT, but no fault either.
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

Result<Value> run_source(Vm& vm, std::string_view source,
                         const CompileOptions& options) noexcept {
    SymbolId module_name = global_symbols().intern("__main__");
    CompileOutcome outcome = compile_program(vm, source, module_name, options);
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
