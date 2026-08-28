// =============================================================================
// vortex/rt/interp.cpp — Tier-0 direct-threaded interpreter implementation.
// =============================================================================

#include "vortex/rt/interp.hpp"

#include <cctype>
#include <cmath>
#include <cstring>

#include "vortex/backend/codegen.hpp"   // Task 24: JitEntryFn for CALL fast path

#include "vortex/frontend/lowering.hpp"
#include "vortex/support/config.hpp"
#include "vortex/support/symbol_table.hpp"

namespace vortex::rt {
inline namespace abi_v1 {

using vortex::fe::NativeHelper;
using vortex::ir::BinOpKind;
using vortex::ir::CmpOpKind;

Program::~Program() {
    Runtime& rt = Runtime::instance();
    for (CodeUnit* u : units) delete u;
    if (globals) rt.decref(reinterpret_cast<PyObj*>(globals));
    for (PyModuleObj* m : native_modules) {
        rt.decref(reinterpret_cast<PyObj*>(m));
    }
    for (auto& kv : sym_key_cache) {
        rt.decref(reinterpret_cast<PyObj*>(kv.second));
    }
}

// =============================================================================
// Helpers
// =============================================================================
void Vm::set_pending(Value exc_owned) noexcept {
    Runtime& rt = Runtime::instance();
    if (pending_exception_.tag() == Tag::Obj && pending_exception_.as_obj()) {
        rt.decref(pending_exception_.as_obj());
    }
    pending_exception_ = exc_owned;
}

Value Vm::take_pending() noexcept {
    Value v = pending_exception_;
    pending_exception_ = Value::none();
    return v;
}

void Vm::raise_builtin(PyTypeObj* type, const char* message) noexcept {
    Runtime& rt = Runtime::instance();
    auto* msg = rt.new_str(message);
    Value arg = Value::object(reinterpret_cast<PyObj*>(msg));
    PyInstanceObj* exc = rt.new_exception(type, &arg, 1);
    set_pending(Value::object(reinterpret_cast<PyObj*>(exc)));
}

bool Vm::unwind_to_handler(Frame& f) noexcept {
    // Find innermost try range containing current pc.
    for (std::size_t i = f.handlers.size(); i-- > 0;) {
        const Frame::Handler& h = f.handlers[i];
        if (f.pc >= h.try_pc_start && f.pc < h.try_pc_end) {
            f.pc = h.handler_pc;
            return true;
        }
    }
    return false;
}

bool Vm::get_global(std::uint32_t symbol, Value& out) noexcept {
    Value key = Value::integer(symbol);   // symbols hash as ints; dict lookup
    // NOTE: globals keys are symbol-string objects created at install time.
    // We look up by interned string identity via symbol-int packed keys.
    // Store convention: globals dict keys are PyStr created from symbols —
    // pre-intern a lookup key per symbol for O(1) reuse.
    Runtime& rt = Runtime::instance();
    // Per-program key cache: owned refs, cleared on Program teardown (the
    // strings must outlive every dict they probe).
    PyStrObj* keystr = nullptr;
    if (PyStrObj** slot = program.sym_key_cache.get(symbol)) {
        keystr = *slot;
    } else {
        keystr = rt.new_str(global_symbols().text(symbol));
        rt.incref(reinterpret_cast<PyObj*>(keystr));
        program.sym_key_cache.insert(symbol, keystr);
    }
    if (dict_get(program.globals, Value::object(reinterpret_cast<PyObj*>(keystr)), out)) {
        return true;
    }
    // fall back: symbol-int keys (module versioning fast path writes these)
    if (dict_get(program.globals, key, out)) return true;
    stdx::small_vector<char, 128> msg;
    for (char c : "name '") msg.push_back(c);
    for (char c : global_symbols().text(symbol)) msg.push_back(c);
    for (char c : "' is not defined") msg.push_back(c);
    msg.push_back('\0');
    raise_builtin(Runtime::instance().type_name_error, msg.data());
    return false;
}

bool Vm::get_attr(const Value& obj, std::uint32_t symbol, Value& out) noexcept {
    Runtime& rt = Runtime::instance();
    if (obj.tag() != Tag::Obj || !obj.as_obj()) {
        raise_builtin(rt.type_attribute_error, "attribute lookup on non-object");
        return false;
    }
    PyObj* o = obj.as_obj();
    switch (o->tag) {
        case ObjTag::Module: {
            auto* m = static_cast<PyModuleObj*>(o);
            Value key = Value::integer(symbol);
            if (!dict_get(m->ns, key, out)) {
                // try string key
                PyStrObj* keystr = rt.new_str(global_symbols().text(symbol));
                if (dict_get(m->ns, Value::object(reinterpret_cast<PyObj*>(keystr)), out)) {
                    rt.decref(reinterpret_cast<PyObj*>(keystr));
                    return true;
                }
                rt.decref(reinterpret_cast<PyObj*>(keystr));
                raise_builtin(rt.type_attribute_error, "module attribute not found");
                return false;
            }
            return true;
        }
        case ObjTag::Instance: {
            auto* inst = static_cast<PyInstanceObj*>(o);
            std::uint32_t slot = 0;
            if (rt.shape_find(inst->shape, symbol, slot) && slot < inst->slot_capacity) {
                out = inst->slots[slot];
                return true;
            }
            // walk type chain for methods
            for (PyTypeObj* t = inst->type; t; t = t->base) {
                Value key = Value::integer(symbol);
                if (dict_get(t->dict, key, out)) {
                    // bind method
                    if (out.tag() == Tag::Obj && out.as_obj() &&
                        out.as_obj()->tag == ObjTag::Function) {
                        auto* bm = static_cast<PyBoundMethodObj*>(
                            std::malloc(sizeof(PyBoundMethodObj)));
                        bm->tag = ObjTag::BoundMethod;
                        bm->flags = 0;
                        bm->refcount = 1;
                        bm->func = out;
                        bm->recv = obj;
                        rt.incref(out.as_obj());
                        rt.incref(obj.as_obj());
                        out = Value::object(reinterpret_cast<PyObj*>(bm));
                    }
                    return true;
                }
            }
            raise_builtin(rt.type_attribute_error, "attribute not found");
            return false;
        }
        case ObjTag::Type: {
            auto* t = static_cast<PyTypeObj*>(o);
            Value key = Value::integer(symbol);
            for (PyTypeObj* tt = t; tt; tt = tt->base) {
                if (dict_get(tt->dict, key, out)) return true;
            }
            raise_builtin(rt.type_attribute_error, "class attribute not found");
            return false;
        }
        case ObjTag::List: {
            // bound sequence methods
            const char* name = nullptr;
            std::string_view sym = global_symbols().text(symbol);
            if (sym == "append") name = "append";
            if (sym == "pop") name = "pop";
            if (name) {
                auto* bm = static_cast<PyBoundMethodObj*>(std::malloc(sizeof(PyBoundMethodObj)));
                bm->tag = ObjTag::BoundMethod;
                bm->flags = 0;
                bm->refcount = 1;
                bm->func = Value::integer(0x100 + (sym == "append" ? 0 : 1));  // marker
                bm->recv = obj;
                rt.incref(obj.as_obj());
                out = Value::object(reinterpret_cast<PyObj*>(bm));
                return true;
            }
            raise_builtin(rt.type_attribute_error, "list attribute not found");
            return false;
        }
        case ObjTag::Dict: {
            std::string_view sym = global_symbols().text(symbol);
            if (sym == "get" || sym == "keys" || sym == "values" || sym == "items") {
                auto* bm = static_cast<PyBoundMethodObj*>(std::malloc(sizeof(PyBoundMethodObj)));
                bm->tag = ObjTag::BoundMethod;
                bm->flags = 0;
                bm->refcount = 1;
                bm->func = Value::integer(0x200 + (sym == "get" ? 0 : sym == "keys" ? 1
                                                  : sym == "values" ? 2 : 3));
                bm->recv = obj;
                rt.incref(obj.as_obj());
                out = Value::object(reinterpret_cast<PyObj*>(bm));
                return true;
            }
            raise_builtin(rt.type_attribute_error, "dict attribute not found");
            return false;
        }
        case ObjTag::Str: {
            std::string_view sym = global_symbols().text(symbol);
            struct StrMethod { const char* name; std::uint64_t kind; };
            static const StrMethod str_methods[] = {
                {"upper", 0x300},    {"lower", 0x301},   {"split", 0x302},
                {"startswith", 0x303}, {"endswith", 0x304}, {"strip", 0x305},
                {"replace", 0x306},  {"join", 0x307},
            };
            for (const StrMethod& m : str_methods) {
                if (sym == m.name) {
                    auto* bm = static_cast<PyBoundMethodObj*>(std::malloc(sizeof(PyBoundMethodObj)));
                    bm->tag = ObjTag::BoundMethod;
                    bm->flags = 0;
                    bm->refcount = 1;
                    bm->func = Value::integer(static_cast<std::int64_t>(m.kind));
                    bm->recv = obj;
                    rt.incref(obj.as_obj());
                    out = Value::object(reinterpret_cast<PyObj*>(bm));
                    return true;
                }
            }
            raise_builtin(rt.type_attribute_error, "str attribute not found");
            return false;
        }
        default:
            raise_builtin(rt.type_attribute_error, "object has no such attribute");
            return false;
    }
}

bool Vm::set_attr(const Value& obj, std::uint32_t symbol, Value value) noexcept {
    Runtime& rt = Runtime::instance();
    if (obj.tag() != Tag::Obj || !obj.as_obj()) {
        raise_builtin(rt.type_attribute_error, "attribute store on non-object");
        return false;
    }
    PyObj* o = obj.as_obj();
    if (o->tag == ObjTag::Instance) {
        auto* inst = static_cast<PyInstanceObj*>(o);
        std::uint32_t slot = 0;
        if (!rt.shape_find(inst->shape, symbol, slot)) {
            ShapeNode* extended = rt.shape_add_attr(inst->shape, symbol);
            slot = extended->slot;
            inst->shape = extended;
        }
        if (slot >= inst->slot_capacity) {
            std::uint32_t new_cap = slot + 1;
            Value* fresh = static_cast<Value*>(std::calloc(new_cap, sizeof(Value)));
            if (!fresh) return false;
            if (inst->slots) {
                std::memcpy(fresh, inst->slots, sizeof(Value) * inst->slot_capacity);
                std::free(inst->slots);
            }
            inst->slots = fresh;
            inst->slot_capacity = new_cap;
        }
        // store: incref new, decref old
        if (value.tag() == Tag::Obj && value.as_obj()) rt.incref(value.as_obj());
        Value& slot_ref = inst->slots[slot];
        if (slot_ref.tag() == Tag::Obj && slot_ref.as_obj()) rt.decref(slot_ref.as_obj());
        slot_ref = value;
        return true;
    }
    if (o->tag == ObjTag::Module) {
        auto* m = static_cast<PyModuleObj*>(o);
        return dict_set(m->ns, Value::integer(symbol), value);
    }
    if (o->tag == ObjTag::Type) {
        auto* t = static_cast<PyTypeObj*>(o);
        return dict_set(t->dict, Value::integer(symbol), value);
    }
    raise_builtin(rt.type_attribute_error, "object does not support attribute assignment");
    return false;
}

bool Vm::get_iter(const Value& obj, Value& out) noexcept {
    Runtime& rt = Runtime::instance();
    if (obj.tag() == Tag::Obj && obj.as_obj()) {
        switch (obj.as_obj()->tag) {
            case ObjTag::RangeIter:
            case ObjTag::ListIter:
            case ObjTag::StrIter:
            case ObjTag::DictIter:
            case ObjTag::Generator:
                // Identity: return an OWNED reference (the register slot will
                // take ownership; the source register keeps its own).
                if (obj.tag() == Tag::Obj && obj.as_obj()) rt.incref(obj.as_obj());
                out = obj;
                return true;
            case ObjTag::List: {
                auto* li = static_cast<PyListIterObj*>(std::malloc(sizeof(PyListIterObj)));
                li->tag = ObjTag::ListIter;
                li->flags = 0;
                li->refcount = 1;
                li->list = static_cast<PyListObj*>(obj.as_obj());
                li->index = 0;
                out = Value::object(reinterpret_cast<PyObj*>(li));
                return true;
            }
            case ObjTag::Str: {
                auto* si = static_cast<PyStrIterObj*>(std::malloc(sizeof(PyStrIterObj)));
                si->tag = ObjTag::StrIter;
                si->flags = 0;
                si->refcount = 1;
                si->str = static_cast<PyStrObj*>(obj.as_obj());
                si->index = 0;
                out = Value::object(reinterpret_cast<PyObj*>(si));
                return true;
            }
            case ObjTag::Dict: {
                auto* di = static_cast<PyDictIterObj*>(std::malloc(sizeof(PyDictIterObj)));
                di->tag = ObjTag::DictIter;
                di->flags = 0;
                di->refcount = 1;
                di->dict = static_cast<PyDictObj*>(obj.as_obj());
                di->slot = 0;   // seq cursor (insertion order)
                di->remaining = static_cast<PyDictObj*>(obj.as_obj())->count;
                out = Value::object(reinterpret_cast<PyObj*>(di));
                return true;
            }
            default:
                break;
        }
    }
    raise_builtin(rt.type_type_error, "object is not iterable");
    return false;
}

bool Vm::iter_check(Value& it, bool& more) noexcept {
    Runtime& rt = Runtime::instance();
    if (it.tag() != Tag::Obj || !it.as_obj()) {
        raise_builtin(rt.type_type_error, "iterator protocol on non-object");
        return false;
    }
    PyObj* o = it.as_obj();
    switch (o->tag) {
        case ObjTag::RangeIter: {
            auto* r = static_cast<PyRangeIterObj*>(o);
            more = r->step > 0 ? r->current < r->stop
                               : (r->step < 0 ? r->current > r->stop : false);
            return true;
        }
        case ObjTag::ListIter: {
            auto* l = static_cast<PyListIterObj*>(o);
            more = l->list && l->index < l->list->length;
            return true;
        }
        case ObjTag::StrIter: {
            auto* s = static_cast<PyStrIterObj*>(o);
            more = s->str && s->index < s->str->length;
            return true;
        }
        case ObjTag::DictIter: {
            auto* d = static_cast<PyDictIterObj*>(o);
            more = d->remaining > 0;
            return true;
        }
        case ObjTag::Generator: {
            auto* g = static_cast<PyGeneratorObj*>(o);
            if (g->exhausted) {
                more = false;
                return true;
            }
            Value out;
            bool has = false;
            if (!generator_step(g, has, out)) return false;
            // cache the value inside the generator (values tuple-less: use a
            // small heap slot via the frame's first scratch register)
            if (has) {
                // OWNED reference into the cache slot: incref the new value,
                // decref whatever was cached before (frame teardown decrefs
                // every slot exactly once). The old code stored a borrowed
                // ref -> refcount underflow -> premature free -> UAF.
                Value& slot =
                    g->frame->regs[g->frame->n_regs > 0 ? g->frame->n_regs - 1 : 0];
                if (slot.tag() == Tag::Obj && slot.as_obj()) rt.decref(slot.as_obj());
                slot = out;   // `out` is an owned ref (L_YIELD incref); transfer
                more = true;
            } else {
                more = false;
            }
            return true;
        }
        default:
            raise_builtin(rt.type_type_error, "object is not an iterator");
            return false;
    }
}

bool Vm::iter_next(const Value& it, Value& out) noexcept {
    Runtime& rt = Runtime::instance();
    if (it.tag() != Tag::Obj || !it.as_obj()) {
        raise_builtin(rt.type_type_error, "iterator protocol on non-object");
        return false;
    }
    PyObj* o = it.as_obj();
    switch (o->tag) {
        case ObjTag::RangeIter: {
            auto* r = static_cast<PyRangeIterObj*>(o);
            out = Value::integer(r->current);
            r->current += r->step;
            return true;
        }
        case ObjTag::ListIter: {
            auto* l = static_cast<PyListIterObj*>(o);
            // INT-3 fix: return an OWNED ref (incref). The previous code
            // returned l->list->items[l->index++] directly — a BORROWED
            // ref. The caller's slot teardown decrefs out, dropping the
            // list's reference and freeing the value while the list still
            // references it (over-decref / UAF).
            out = l->list->items[l->index++];
            if (out.tag() == Tag::Obj && out.as_obj()) rt.incref(out.as_obj());
            return true;
        }
        case ObjTag::StrIter: {
            auto* s = static_cast<PyStrIterObj*>(o);
            // INT-3 fix: new_str returns an owned ref already (no incref needed).
            out = Value::object(reinterpret_cast<PyObj*>(rt.new_str(
                std::string_view(s->str->data() + s->index, 1))));
            ++s->index;
            return true;
        }
        case ObjTag::DictIter: {
            // Insertion-order iteration (CPython 3.7+ guarantee): repeatedly
            // select the live entry with the smallest seq >= cursor.
            // OBJ-16 fix: skip tombstones (used=true, key.tag()=None). With
            // backward-shift deletion in dict_del, tombstones shouldn't be
            // present in the live table, but a table may have been built
            // before this fix shipped — defensive guard prevents the iter
            // from returning None for deleted entries.
            auto* di = static_cast<PyDictIterObj*>(o);
            PyDictObj* dict = di->dict;
            if (di->remaining == 0 || !dict) {
                out = Value::none();
                return true;
            }
            std::uint32_t best = 0xFFFFFFFFu;
            std::uint32_t best_slot = 0;
            for (std::uint32_t i = 0; i < dict->capacity; ++i) {
                const DictEntry& e = dict->entries[i];
                if (!e.used || e.seq < di->slot) continue;   // already emitted
                if (e.key.tag() == Tag::None) continue;        // tombstone
                if (e.seq < best) {
                    best = e.seq;
                    best_slot = i;
                }
            }
            if (best == 0xFFFFFFFFu) {
                out = Value::none();
                return true;
            }
            // OBJ-15-style fix: return an OWNED reference (incref) so the
            // caller's decref balances. Previously iter_next returned a
            // borrowed ref from the dict's slot — the caller's slot
            // teardown would drop the dict's only reference, freeing the
            // value while the dict still referenced it.
            out = dict->entries[best_slot].key;
            if (out.tag() == Tag::Obj && out.as_obj()) rt.incref(out.as_obj());
            di->slot = best + 1;   // next scan starts after this seq
            --di->remaining;
            return true;
        }
        case ObjTag::Generator: {
            auto* g = static_cast<PyGeneratorObj*>(o);
            // cached by iter_check in the last register slot.
            // INT-3 fix: return an OWNED ref (incref). The previous code
            // returned the cached slot value directly — a BORROWED ref.
            // The caller's slot teardown decrefs out, dropping the
            // generator's reference and freeing the value while the
            // generator's frame still references it (over-decref / UAF).
            out = g->frame->regs[g->frame->n_regs > 0 ? g->frame->n_regs - 1 : 0];
            if (out.tag() == Tag::Obj && out.as_obj()) rt.incref(out.as_obj());
            return true;
        }
        default:
            raise_builtin(rt.type_type_error, "object is not an iterator");
            return false;
    }
}

bool Vm::generator_step(PyGeneratorObj* gen, bool& has_value, Value& out) noexcept {
    has_value = false;
    if (gen->exhausted || !gen->frame) {
        gen->exhausted = true;
        return true;
    }
    ExecStatus st = exec_frame(*gen->frame);
    if (st == ExecStatus::Suspended) {
        out = frame_return_;
        frame_return_ = Value::none();
        has_value = true;
        return true;
    }
    if (st == ExecStatus::Raised) {
        gen->exhausted = true;
        return false;   // pending_exception_ carries the raise
    }
    // Returned: generator finished. StopIteration semantics handled by caller
    // setting exhausted.
    gen->exhausted = true;
    has_value = false;
    return true;
}

// =============================================================================
// Parameter binding & calls
// =============================================================================
bool Vm::bind_parameters(Frame& f, PyFuncObj* fn, Value* args, std::uint32_t argc,
                          PyTupleObj* kw_names, std::uint32_t nkw) noexcept {
    Runtime& rt = Runtime::instance();
    CodeUnit* unit = f.unit;
    std::uint32_t n_params = static_cast<std::uint32_t>(unit->param_regs.size());

    // The hidden closure-cells parameter occupies the LAST slot (appended by
    // lowering after the declared params). Exclude it from plain binding.
    std::uint32_t declared = n_params;
    if (fn && fn->cells && declared > 0) declared -= 1;

    // varargs / kwargs packing: trailing params
    std::uint32_t plain = declared;
    if (unit->has_varargs && plain > 0) plain -= 1;
    if (unit->has_kwargs && plain > 0) plain -= 1;

    if (argc > plain) {
        if (!unit->has_varargs) {
            raise_builtin(rt.type_type_error, "too many positional arguments");
            return false;
        }
    }

    // positional
    std::uint32_t i = 0;
    for (; i < argc && i < plain; ++i) {
        Value v = args[i];
        if (v.tag() == Tag::Obj && v.as_obj()) rt.incref(v.as_obj());
        f.regs[unit->param_regs[i]] = v;
    }
    // defaults for missing positionals
    std::uint32_t ndefaults = fn && fn->defaults ? fn->defaults->length : 0;
    for (; i < plain; ++i) {
        bool used_default = false;
        if (fn && fn->defaults && ndefaults > 0 && i + ndefaults >= plain) {
            // default index: ndefaults - (plain - i)
            std::uint32_t di = ndefaults - (plain - i);
            if (di < ndefaults) {
                Value dv = fn->defaults->items[di];
                if (dv.tag() == Tag::Obj && dv.as_obj()) rt.incref(dv.as_obj());
                f.regs[unit->param_regs[i]] = dv;
                used_default = true;
            }
        }
        if (!used_default) {
            raise_builtin(rt.type_type_error, "missing required argument");
            return false;
        }
    }
    // varargs tuple
    if (unit->has_varargs && n_params >= 1) {
        std::uint32_t var_reg = unit->param_regs[plain];
        PyTupleObj* tup = rt.new_tuple(argc > plain ? argc - plain : 0);
        for (std::uint32_t k = plain; k < argc; ++k) {
            tup->items[k - plain] = args[k];
            if (args[k].tag() == Tag::Obj && args[k].as_obj()) rt.incref(args[k].as_obj());
        }
        f.regs[var_reg] = Value::object(reinterpret_cast<PyObj*>(tup));
    }
    // kwargs dict from kw_names (handles def f(**kwargs) and def f(*args, **kwargs)
    // and the partial-positional-with-kwargs case).
    // INT-5 fix: the previous code computed kw_reg = param_regs[plain + 1],
    // which assumed has_varargs (param_regs[plain] is varargs, [plain+1] is
    // kwargs). When has_kwargs without has_varargs, kwargs is at param_regs[plain],
    // not plain+1. INT-7 fix: def f(**kwargs) has n_params == 1 (just kwargs);
    // the previous code required n_params >= 2. INT-6 fix: in the has_kwargs
    // branch, also try to bind kwargs to matching positional params first
    // (Python: f(a, **kwargs) called as f(a=1) binds a=1 and kwargs={}).
    std::uint32_t kw_param_index = unit->has_varargs ? (plain + 1) : plain;
    if (unit->has_kwargs && kw_param_index < n_params) {
        std::uint32_t kw_reg = unit->param_regs[kw_param_index];
        PyDictObj* d = rt.new_dict();
        // INT-6: bind matching kwargs to positional params first.
        stdx::small_vector<bool, 8> consumed_kw(nkw, false);
        for (std::uint32_t k = 0; k < nkw && kw_names; ++k) {
            Value key = kw_names->items[k];
            if (key.tag() != Tag::Int) continue;
            for (std::uint32_t p = 0; p < plain; ++p) {
                if (unit->param_names.size() > p &&
                    unit->param_names[p] == static_cast<std::uint32_t>(key.as_i())) {
                    // Bind this kwarg to positional param p (overrides any
                    // positional binding — Python: kwarg wins).
                    Value val = args[argc + k];
                    if (val.tag() == Tag::Obj && val.as_obj()) rt.incref(val.as_obj());
                    // Decref the old positional binding if it was set.
                    if (f.regs[unit->param_regs[p]].tag() == Tag::Obj &&
                        f.regs[unit->param_regs[p]].as_obj()) {
                        rt.decref(f.regs[unit->param_regs[p]].as_obj());
                    }
                    f.regs[unit->param_regs[p]] = val;
                    consumed_kw[k] = true;
                    break;
                }
            }
        }
        // Remaining kwargs go into the **kwargs dict.
        for (std::uint32_t k = 0; k < nkw && kw_names; ++k) {
            if (consumed_kw.size() > k && consumed_kw[k]) continue;
            Value key = kw_names->items[k];
            Value val = argc + k < argc + nkw ? args[argc + k] : Value::none();
            if (!dict_set(d, key, val)) {
                raise_builtin(rt.type_memory_error, "kwargs dict allocation failed");
                return false;
            }
        }
        f.regs[kw_reg] = Value::object(reinterpret_cast<PyObj*>(d));
    } else if (nkw > 0 && kw_names && !unit->has_kwargs) {
        // No **kwargs declared: every kwarg must match a positional param.
        for (std::uint32_t k = 0; k < nkw; ++k) {
            Value key = kw_names->items[k];
            if (key.tag() != Tag::Int) continue;
            bool matched = false;
            for (std::uint32_t p = 0; p < plain; ++p) {
                if (unit->param_names.size() > p &&
                    unit->param_names[p] == static_cast<std::uint32_t>(key.as_i())) {
                    Value val = args[argc + k];
                    if (val.tag() == Tag::Obj && val.as_obj()) rt.incref(val.as_obj());
                    // Decref the old positional binding if it was set.
                    if (f.regs[unit->param_regs[p]].tag() == Tag::Obj &&
                        f.regs[unit->param_regs[p]].as_obj()) {
                        rt.decref(f.regs[unit->param_regs[p]].as_obj());
                    }
                    f.regs[unit->param_regs[p]] = val;
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                raise_builtin(rt.type_type_error, "unexpected keyword argument");
                return false;
            }
        }
    }

    // closure cells: hidden last param
    if (fn && fn->cells && !unit->param_regs.empty()) {
        std::uint32_t cells_reg = unit->param_regs.back();
        Value cv = Value::object(reinterpret_cast<PyObj*>(fn->cells));
        rt.incref(reinterpret_cast<PyObj*>(fn->cells));
        f.regs[cells_reg] = cv;
    }
    return true;
}

bool Vm::builtin_call(PyNativeFnObj* fn, Value* args, std::uint32_t argc, Value& out) noexcept {
    out = fn->fn(fn->user, args, argc);
    if (out.tag() == Tag::Obj && out.as_obj() == nullptr) {
        out = Value::none();
        return false;   // null signals error (pending set by helper)
    }
    return true;
}

bool Vm::call_value(const Value& callee, Value* args, std::uint32_t argc, Value& out) noexcept {
    return call_value_kw(callee, args, argc, nullptr, 0, out);
}

bool Vm::call_value_kw(const Value& callee, Value* args, std::uint32_t argc,
                       PyTupleObj* kw_names, std::uint32_t nkw, Value& out) noexcept {
    Runtime& rt = Runtime::instance();

    if (callee.tag() != Tag::Obj || !callee.as_obj()) {
        raise_builtin(rt.type_type_error, "call of non-callable value");
        return false;
    }
    PyObj* target = callee.as_obj();

    if (call_depth >= cfg::max_call_depth) {
        raise_builtin(rt.type_runtime_error, "maximum recursion depth exceeded");
        return false;
    }

    switch (target->tag) {
        case ObjTag::NativeFn: {
            auto* fn = static_cast<PyNativeFnObj*>(target);
            return builtin_call(fn, args, argc, out);
        }
        case ObjTag::BoundMethod: {
            auto* bm = static_cast<PyBoundMethodObj*>(target);
            // builtin bound methods on list/dict/str
            if (bm->func.tag() == Tag::Int) {
                std::uint64_t kind = static_cast<std::uint64_t>(bm->func.as_i());
                return builtin_bound_method(kind, bm->recv, args, argc, out);
            }
            // general: call func with recv prepended
            stdx::small_vector<Value, 8> full;
            full.push_back(bm->recv);
            for (std::uint32_t i = 0; i < argc; ++i) full.push_back(args[i]);
            bool ok = call_value_kw(bm->func, full.data(), argc + 1, kw_names, nkw, out);
            return ok;
        }
        case ObjTag::Function: {
            auto* fn = static_cast<PyFuncObj*>(target);
            CodeUnit* unit = nullptr;
            if (fn->code_unit_id < program.units.size()) {
                unit = program.units[fn->code_unit_id];
            }
            if (!unit) {
                raise_builtin(rt.type_runtime_error, "code unit not linked");
                return false;
            }
            // Telemetry: call_count. Skip the atomic RMW in the hot path —
            // it's ~10ns per call (relaxed atomic is still a serialization
            // point on x86). Sampled lazily if needed for tiering decisions.
            // Re-enable when the tiering daemon actually reads it.
            // unit->call_count.fetch_add(1, std::memory_order_relaxed);

            if (unit->is_generator) {
                auto* gen = static_cast<PyGeneratorObj*>(std::malloc(sizeof(PyGeneratorObj)));
                gen->tag = ObjTag::Generator;
                gen->flags = 0;
                gen->refcount = 1;
                gen->frame = new Frame(unit);
                gen->code_unit_id = fn->code_unit_id;
                gen->exhausted = false;
                if (!bind_parameters(*gen->frame, fn, args, argc, kw_names, nkw)) {
                    delete gen->frame;
                    std::free(gen);
                    return false;
                }
                out = Value::object(reinterpret_cast<PyObj*>(gen));
                return true;
            }

            Frame f(unit);
            if (!bind_parameters(f, fn, args, argc, kw_names, nkw)) return false;

            // JIT fast path: call jit_entry when available.
            // The bridge executes ONE dynamic op via step_one and returns.
            //
            // GATE: has_dynamic_ops gates the JIT because the JIT's register
            // allocator doesn't always write live values to home slots before
            // a CALLri. step_one reads from regs[cur->a] (home slots), but
            // the value might be in a physical register only. The fix is to
            // force write-back of all live values before each CALLri in the
            // codegen. Until that's done, the gate prevents wrong results.
            void* entry_raw = unit->jit_entry.load(std::memory_order_acquire);
            // TEMPORARY: JIT disabled while NaN-boxing codegen is being fixed.
            // The stage_rax loads the full NaN-boxed word but ALU ops need the
            // payload masked. Until the codegen masks payloads after load, the
            // JIT produces wrong results. Rule 120: fall back to Tier-0.
            if (entry_raw && !unit->has_dynamic_ops && !unit->is_generator &&
                !jit_disabled_in_bridge && false) {  // TEMPORARY: always skip JIT
                // Task 24: initialize Phi home slots from their entry
                // values before calling jit_entry. The Tier-0
                // interpreter's prologue (LOAD_CONST + MOVE) does
                // this for the bytecode path; the JIT bypasses that
                // prologue, so we replicate it here. Without this,
                // every GUARD_INT would fail on the first read
                // (Phis start as Value::none()). The driver
                // populates phi_init_node_ids/values during
                // compile_program by walking the IR for Phis with
                // ConstInt entry inputs.
                const std::size_t n_init = unit->phi_init_node_ids.size();
                for (std::size_t i = 0; i < n_init; ++i) {
                    std::uint32_t node_id = unit->phi_init_node_ids[i];
                    if (node_id < f.n_regs) {
                        // Task 24 (candidate j): ConstFloat entries
                        // are bit-reinterpreted back to double here.
                        // The driver pushes int64 bits for transport;
                        // we reinterpret to double and wrap in
                        // Value::real (tag=Tag::Float).
                        const bool is_float =
                            i < unit->phi_init_is_float.size() &&
                            unit->phi_init_is_float[i] != 0;
                        if (is_float) {
                            double fv;
                            std::memcpy(&fv, &unit->phi_init_values[i],
                                        sizeof(double));
                            f.regs[node_id] = Value::real(fv);
                        } else {
                            f.regs[node_id] = Value::integer(
                                unit->phi_init_values[i]);
                        }
                    }
                }
                auto entry = reinterpret_cast<backend::JitEntryFn>(entry_raw);
                ++call_depth;
                Value rv = entry(f.regs);
                --call_depth;
                // The bridge returns the function's return value. If the
                // bridge's step_one failed (unhandled op or exception),
                // the bridge itself falls back to exec_frame and returns
                // the actual result — so rv is always the function's return
                // value on success, or none() with a pending exception on
                // failure. There is NO silent failure: if rv is none(),
                // either the function returned None, or the bridge handled
                // the exception (pending is set).
                if (rv.tag() == Tag::None && has_pending()) {
                    return false;   // propagate exception to caller
                }
                out = rv;
                return true;
            }

            ++call_depth;
            ExecStatus st = exec_frame(f);
            --call_depth;
            if (st == ExecStatus::Returned) {
                out = frame_return_;
                frame_return_ = Value::none();
                return true;
            }
            return false;   // Raised (pending set) or Suspended (invalid here)
        }
        case ObjTag::Type: {
            auto* t = static_cast<PyTypeObj*>(target);
            PyInstanceObj* inst = rt.new_instance(t);
            out = Value::object(reinterpret_cast<PyObj*>(inst));
            rt.incref(reinterpret_cast<PyObj*>(inst));
            // __init__? walk chain
            Value init;
            for (PyTypeObj* tt = t; tt; tt = tt->base) {
                if (dict_get(tt->dict, Value::integer(
                        global_symbols().intern("__init__")), init)) {
                    break;
                }
                init = Value::none();
            }
            if (init.tag() == Tag::Obj && init.as_obj()) {
                stdx::small_vector<Value, 8> full;
                full.push_back(out);   // self
                for (std::uint32_t i = 0; i < argc; ++i) full.push_back(args[i]);
                Value ignored;
                if (!call_value_kw(init, full.data(), argc + 1, kw_names, nkw, ignored)) {
                    rt.decref(reinterpret_cast<PyObj*>(inst));
                    return false;
                }
            }
            return true;
        }
        default:
            raise_builtin(rt.type_type_error, "object is not callable");
            return false;
    }
}

bool Vm::builtin_bound_method(std::uint64_t kind, const Value& recv, Value* args,
                              std::uint32_t argc, Value& out) noexcept {
    Runtime& rt = Runtime::instance();
    if (kind >= 0x100 && kind < 0x200) {
        // list methods
        auto* l = static_cast<PyListObj*>(recv.as_obj());
        if (kind == 0x100) {   // append
            if (argc != 1) {
                raise_builtin(rt.type_type_error, "append takes exactly one argument");
                return false;
            }
            Value v = args[0];
            if (v.tag() == Tag::Obj && v.as_obj()) rt.incref(v.as_obj());
            if (!list_push(l, v)) {
                raise_builtin(rt.type_memory_error, "list allocation failed");
                return false;
            }
            out = Value::none();
            return true;
        }
        if (kind == 0x101) {   // pop([index])
            if (l->length == 0) {
                raise_builtin(rt.type_index_error, "pop from empty list");
                return false;
            }
            // OBJ-22 fix: honor the optional index argument. The previous
            // code always popped the last element, so l.pop(0) returned
            // the last element instead of the first. Python: pop(i)
            // removes and returns the item at index i (negative indices
            // count from the end).
            std::int64_t idx = static_cast<std::int64_t>(l->length) - 1;
            if (argc >= 1) {
                if (!as_i64(args[0], idx)) {
                    raise_builtin(rt.type_type_error, "pop index must be an integer");
                    return false;
                }
                if (idx < 0) idx += static_cast<std::int64_t>(l->length);
            }
            if (idx < 0 || idx >= static_cast<std::int64_t>(l->length)) {
                raise_builtin(rt.type_index_error, "pop index out of range");
                return false;
            }
            std::uint32_t u_idx = static_cast<std::uint32_t>(idx);
            out = l->items[u_idx];
            // incref out since the caller treats it as owned (was borrowed).
            if (out.tag() == Tag::Obj && out.as_obj()) rt.incref(out.as_obj());
            // shift remaining elements left
            std::memmove(l->items + u_idx, l->items + u_idx + 1,
                         sizeof(Value) * (l->length - u_idx - 1));
            l->length--;
            return true;
        }
    }
    if (kind >= 0x200 && kind < 0x300) {
        // dict methods: get/keys/values/items
        auto* d = static_cast<PyDictObj*>(recv.as_obj());
        if (kind == 0x200) {   // get(key[, default])
            if (argc == 0) {
                raise_builtin(rt.type_type_error, "get needs a key");
                return false;
            }
            if (!dict_get(d, args[0], out)) {
                out = argc > 1 ? args[1] : Value::none();
            }
            return true;
        }
        // keys/values/items -> list
        auto* result = rt.new_list();
        for (std::uint32_t i = 0; i < d->capacity; ++i) {
            // OBJ-7 fix: skip tombstones. used=true with key.tag()=None is a
            // tombstone from the old deletion scheme (current dict_del
            // uses backward-shift, so this is defensive — but the table
            // may have been built by external code with tombstones).
            if (!d->entries[i].used) continue;
            if (d->entries[i].key.tag() == Tag::None) continue;
            Value v;
                if (kind == 0x201) v = d->entries[i].key;
            else if (kind == 0x202) v = d->entries[i].value;
            else {
                auto* pair = rt.new_tuple(2);
                pair->items[0] = d->entries[i].key;
                pair->items[1] = d->entries[i].value;
                if (pair->items[0].tag() == Tag::Obj) rt.incref(pair->items[0].as_obj());
                if (pair->items[1].tag() == Tag::Obj) rt.incref(pair->items[1].as_obj());
                v = Value::object(reinterpret_cast<PyObj*>(pair));
                // For pair: list_push adopts (incref's), but we own a
                // separate ref from new_tuple. Drop ours.
                list_push(result, v);
                rt.decref(reinterpret_cast<PyObj*>(pair));
                continue;
            }
            // OBJ-7 fix: the previous code incref'd v before list_push
            // (which also incref's), leaking +1 per entry. list_push
            // handles the incref for us; just push the borrowed value.
            list_push(result, v);
        }
        out = Value::object(reinterpret_cast<PyObj*>(result));
        return true;
    }
    if (kind >= 0x300) {
        // Str bound methods. Kinds assigned at LOAD_ATTR:
        //   0x300 upper, 0x301 lower, 0x302 split, 0x303 startswith,
        //   0x304 endswith, 0x305 strip, 0x306 replace, 0x307 join.
        auto* str = static_cast<PyStrObj*>(recv.as_obj());
        std::string_view sv(str->data(), str->length);
        switch (kind) {
            case 0x300: {   // upper
                stdx::small_vector<char, 128> buf;
                for (char c : sv) buf.push_back(static_cast<char>(std::toupper((unsigned char)c)));
                out = Value::object(reinterpret_cast<PyObj*>(rt.new_str(
                    std::string_view(buf.data(), buf.size()))));
                return true;
            }
            case 0x301: {   // lower
                stdx::small_vector<char, 128> buf;
                for (char c : sv) buf.push_back(static_cast<char>(std::tolower((unsigned char)c)));
                out = Value::object(reinterpret_cast<PyObj*>(rt.new_str(
                    std::string_view(buf.data(), buf.size()))));
                return true;
            }
            case 0x302: {   // split(sep)
                if (argc < 1 || args[0].tag() != Tag::Obj || !args[0].as_obj() ||
                    args[0].as_obj()->tag != ObjTag::Str) {
                    raise_builtin(rt.type_type_error, "split expects a string separator");
                    return false;
                }
                auto* sep = static_cast<PyStrObj*>(args[0].as_obj());
                auto* result = rt.new_list();
                std::string_view sepv(sep->data(), sep->length);
                if (sepv.empty()) {
                    // no separator: split on whitespace (subset: whole string)
                    auto* piece = rt.new_str(sv);
                    list_push(result, Value::object(reinterpret_cast<PyObj*>(piece)));
                } else {
                    std::size_t start = 0;
                    for (;;) {
                        std::size_t at = sv.find(sepv, start);
                        if (at == std::string_view::npos) {
                            auto* piece = rt.new_str(sv.substr(start));
                            list_push(result, Value::object(reinterpret_cast<PyObj*>(piece)));
                            break;
                        }
                        auto* piece = rt.new_str(sv.substr(start, at - start));
                        list_push(result, Value::object(reinterpret_cast<PyObj*>(piece)));
                        start = at + sepv.size();
                    }
                }
                out = Value::object(reinterpret_cast<PyObj*>(result));
                return true;
            }
            case 0x303: {   // startswith
                if (argc < 1 || args[0].tag() != Tag::Obj || !args[0].as_obj() ||
                    args[0].as_obj()->tag != ObjTag::Str) {
                    raise_builtin(rt.type_type_error, "startswith expects a string");
                    return false;
                }
                auto* pre = static_cast<PyStrObj*>(args[0].as_obj());
                std::string_view pv(pre->data(), pre->length);
                out = Value::boolean(sv.size() >= pv.size() && sv.substr(0, pv.size()) == pv);
                return true;
            }
            case 0x304: {   // endswith
                if (argc < 1 || args[0].tag() != Tag::Obj || !args[0].as_obj() ||
                    args[0].as_obj()->tag != ObjTag::Str) {
                    raise_builtin(rt.type_type_error, "endswith expects a string");
                    return false;
                }
                auto* suf = static_cast<PyStrObj*>(args[0].as_obj());
                std::string_view pv(suf->data(), suf->length);
                out = Value::boolean(sv.size() >= pv.size() && sv.substr(sv.size() - pv.size()) == pv);
                return true;
            }
            case 0x305: {   // strip
                std::size_t b = 0, e = sv.size();
                while (b < e && std::isspace((unsigned char)sv[b])) ++b;
                while (e > b && std::isspace((unsigned char)sv[e - 1])) --e;
                out = Value::object(reinterpret_cast<PyObj*>(
                    rt.new_str(sv.substr(b, e - b))));
                return true;
            }
            case 0x306: {   // replace(old, new)
                if (argc < 2 || args[0].tag() != Tag::Obj || !args[0].as_obj() ||
                    args[0].as_obj()->tag != ObjTag::Str || args[1].tag() != Tag::Obj ||
                    !args[1].as_obj() || args[1].as_obj()->tag != ObjTag::Str) {
                    raise_builtin(rt.type_type_error, "replace expects two strings");
                    return false;
                }
                auto* old_s = static_cast<PyStrObj*>(args[0].as_obj());
                auto* new_s = static_cast<PyStrObj*>(args[1].as_obj());
                std::string_view ov(old_s->data(), old_s->length);
                std::string_view nv(new_s->data(), new_s->length);
                stdx::small_vector<char, 128> buf;
                if (ov.empty()) {
                    out = Value::object(reinterpret_cast<PyObj*>(rt.new_str(sv)));
                    return true;
                }
                std::size_t start = 0;
                for (;;) {
                    std::size_t at = sv.find(ov, start);
                    if (at == std::string_view::npos) {
                        for (std::size_t k = start; k < sv.size(); ++k) buf.push_back(sv[k]);
                        break;
                    }
                    for (std::size_t k = start; k < at; ++k) buf.push_back(sv[k]);
                    for (char c : nv) buf.push_back(c);
                    start = at + ov.size();
                }
                out = Value::object(reinterpret_cast<PyObj*>(rt.new_str(
                    std::string_view(buf.data(), buf.size()))));
                return true;
            }
            case 0x307: {   // join(iterable of str)
                if (argc < 1 || args[0].tag() != Tag::Obj || !args[0].as_obj() ||
                    args[0].as_obj()->tag != ObjTag::List) {
                    raise_builtin(rt.type_type_error, "join expects a list");
                    return false;
                }
                auto* parts = static_cast<PyListObj*>(args[0].as_obj());
                stdx::small_vector<char, 128> buf;
                for (std::uint32_t i = 0; i < parts->length; ++i) {
                    if (i) for (char c : sv) buf.push_back(c);
                    Value& p = parts->items[i];
                    if (p.tag() == Tag::Obj && p.as_obj() && p.as_obj()->tag == ObjTag::Str) {
                        auto* ps = static_cast<PyStrObj*>(p.as_obj());
                        for (std::uint32_t k = 0; k < ps->length; ++k) buf.push_back(ps->data()[k]);
                    }
                }
                out = Value::object(reinterpret_cast<PyObj*>(rt.new_str(
                    std::string_view(buf.data(), buf.size()))));
                return true;
            }
            default:
                raise_builtin(rt.type_not_implemented_error, "unknown str method");
                return false;
        }
    }
    raise_builtin(rt.type_type_error, "unknown bound method");
    return false;
}

// =============================================================================
// Native helpers (CallNative ops)
// =============================================================================
bool Vm::native_helper(std::uint16_t helper, Value* args, std::uint32_t argc,
                       Value& out) noexcept {
    Runtime& rt = Runtime::instance();
    switch (static_cast<NativeHelper>(helper)) {
        case NativeHelper::MakeFunction: {
            // args: [code_unit_id, ndefaults, defaults..., cells]
            if (argc < 3) {
                raise_builtin(rt.type_runtime_error, "MakeFunction arity");
                return false;
            }
            std::uint32_t unit_id = static_cast<std::uint32_t>(args[0].as_i());
            std::uint32_t ndefaults = static_cast<std::uint32_t>(args[1].as_i());
            // INT-11 fix: bound-check the defaults copy against argc.
            // The previous code read args[2 + i] unconditionally for
            // i in [0, ndefaults); if ndefaults > argc - 2 (caller passed
            // fewer args than claimed defaults), it read past the args
            // array. Cap ndefaults at argc - 2.
            if (ndefaults > argc - 2) {
                raise_builtin(rt.type_runtime_error,
                              "MakeFunction: ndefaults exceeds available args");
                return false;
            }
            PyTupleObj* defaults = nullptr;
            if (ndefaults > 0) {
                defaults = rt.new_tuple(ndefaults);
                for (std::uint32_t i = 0; i < ndefaults; ++i) {
                    defaults->items[i] = args[2 + i];
                    if (args[2 + i].tag() == Tag::Obj && args[2 + i].as_obj()) {
                        rt.incref(args[2 + i].as_obj());
                    }
                }
            }
            Value cells_v = args[2 + ndefaults];
            PyTupleObj* cells = nullptr;
            if (cells_v.tag() == Tag::Obj && cells_v.as_obj() &&
                cells_v.as_obj()->tag == ObjTag::Tuple) {
                cells = static_cast<PyTupleObj*>(cells_v.as_obj());
            }
            CodeUnit* unit = unit_id < program.units.size() ? program.units[unit_id] : nullptr;
            if (!unit) {
                raise_builtin(rt.type_runtime_error, "MakeFunction: unknown code unit");
                return false;
            }
            SymbolId name = unit->name;
            auto* fn = rt.new_func(unit_id, name, defaults, cells,
                                   static_cast<std::uint32_t>(unit->param_regs.size()),
                                   unit->has_varargs, unit->has_kwargs);
            out = Value::object(reinterpret_cast<PyObj*>(fn));
            return true;
        }
        case NativeHelper::MakeCell: {
            PyCellObj* cell = rt.new_cell(args[0]);
            out = Value::object(reinterpret_cast<PyObj*>(cell));
            return true;
        }
        case NativeHelper::CellGet: {
            auto* cell = static_cast<PyCellObj*>(args[0].as_obj());
            if (cell->flags & PyCellObj::kUnbound) {
                raise_builtin(rt.type_runtime_error,
                              "cannot read unbound closure variable (UnboundLocalError)");
                return false;
            }
            out = cell->value;
            return true;
        }
        case NativeHelper::CellSet: {
            auto* cell = static_cast<PyCellObj*>(args[0].as_obj());
            Value v = args[1];
            if (v.tag() == Tag::Obj && v.as_obj()) rt.incref(v.as_obj());
            if (cell->value.tag() == Tag::Obj && cell->value.as_obj()) {
                rt.decref(cell->value.as_obj());
            }
            cell->value = v;
            cell->flags &= ~PyCellObj::kUnbound;
            out = Value::none();
            return true;
        }
        case NativeHelper::ImportModule: {
            std::uint32_t sym = 0;
            // arg is a symbol-string const: Value integer symbol or str object
            if (args[0].tag() == Tag::Int) {
                sym = static_cast<std::uint32_t>(args[0].as_i());
            } else if (args[0].tag() == Tag::Obj && args[0].as_obj() &&
                       args[0].as_obj()->tag == ObjTag::Str) {
                sym = global_symbols().intern(
                    static_cast<PyStrObj*>(args[0].as_obj())->view());
            }
            PyModuleObj* mod = load_native_module(sym);
            if (!mod) {
                stdx::small_vector<char, 128> msg;
                for (char c : "no module named '") msg.push_back(c);
                for (char c : global_symbols().text(sym)) msg.push_back(c);
                msg.push_back('\'');
                msg.push_back('\0');
                raise_builtin(rt.type_runtime_error, msg.data());
                return false;
            }
            out = Value::object(reinterpret_cast<PyObj*>(mod));
            return true;
        }
        case NativeHelper::MakeClass: {
            // args: [name_str, namespace_dict, base_or_none]
            std::string_view name;
            if (args[0].tag() == Tag::Int) {
                name = global_symbols().text(static_cast<std::uint32_t>(args[0].as_i()));
            } else if (args[0].tag() == Tag::Obj && args[0].as_obj() &&
                       args[0].as_obj()->tag == ObjTag::Str) {
                name = static_cast<PyStrObj*>(args[0].as_obj())->view();
            }
            PyDictObj* ns = nullptr;
            if (args[1].tag() == Tag::Obj && args[1].as_obj() &&
                args[1].as_obj()->tag == ObjTag::Dict) {
                ns = static_cast<PyDictObj*>(args[1].as_obj());
            }
            PyTypeObj* base = nullptr;
            if (args[2].tag() == Tag::Obj && args[2].as_obj() &&
                args[2].as_obj()->tag == ObjTag::Type) {
                base = static_cast<PyTypeObj*>(args[2].as_obj());
            }
            PyTypeObj* cls = rt.new_type(global_symbols().intern(name), base, ns);
            // Inherit base attributes: own (ns) entries WIN — copy base
            // entries only where the subclass did not define them.
            if (base && base->dict) {
                for (std::uint32_t i = 0; i < base->dict->capacity; ++i) {
                    if (!base->dict->entries[i].used) continue;
                    Value existing;
                    if (dict_get(cls->dict, base->dict->entries[i].key, existing)) {
                        continue;   // subclass override stays
                    }
                    if (!dict_set(cls->dict, base->dict->entries[i].key,
                                  base->dict->entries[i].value)) {
                        raise_builtin(rt.type_memory_error, "class dict copy failed");
                        return false;
                    }
                }
            }
            out = Value::object(reinterpret_cast<PyObj*>(cls));
            return true;
        }
        case NativeHelper::UnpackSequence: {
            // args: [value, n]
            std::uint32_t n = static_cast<std::uint32_t>(args[1].as_i());
            PyTupleObj* tup = rt.new_tuple(n);
            bool ok = false;
            if (args[0].tag() == Tag::Obj && args[0].as_obj()) {
                PyObj* src = args[0].as_obj();
                if (src->tag == ObjTag::List) {
                    auto* l = static_cast<PyListObj*>(src);
                    if (l->length == n) {
                        for (std::uint32_t i = 0; i < n; ++i) {
                            tup->items[i] = l->items[i];
                            if (l->items[i].tag() == Tag::Obj) rt.incref(l->items[i].as_obj());
                        }
                        ok = true;
                    }
                } else if (src->tag == ObjTag::Tuple) {
                    auto* t = static_cast<PyTupleObj*>(src);
                    if (t->length == n) {
                        for (std::uint32_t i = 0; i < n; ++i) {
                            tup->items[i] = t->items[i];
                            if (t->items[i].tag() == Tag::Obj) rt.incref(t->items[i].as_obj());
                        }
                        ok = true;
                    }
                } else if (src->tag == ObjTag::Str) {
                    auto* s = static_cast<PyStrObj*>(src);
                    if (s->length == n) {
                        for (std::uint32_t i = 0; i < n; ++i) {
                            auto* ch = rt.new_str(std::string_view(s->data() + i, 1));
                            tup->items[i] = Value::object(reinterpret_cast<PyObj*>(ch));
                        }
                        ok = true;
                    }
                }
            }
            if (!ok) {
                rt.decref(reinterpret_cast<PyObj*>(tup));
                raise_builtin(rt.type_value_error, "cannot unpack sequence to requested length");
                return false;
            }
            out = Value::object(reinterpret_cast<PyObj*>(tup));
            return true;
        }
        case NativeHelper::UnboundCheck: {
            raise_builtin(rt.type_runtime_error,
                          "local variable referenced before assignment (UnboundLocalError)");
            return false;
        }
        case NativeHelper::DelSubscript: {
            if (args[0].tag() == Tag::Obj && args[0].as_obj() &&
                args[0].as_obj()->tag == ObjTag::Dict) {
                if (!dict_del(static_cast<PyDictObj*>(args[0].as_obj()), args[1])) {
                    raise_builtin(rt.type_key_error, "del: key not found");
                    return false;
                }
                out = Value::none();
                return true;
            }
            if (args[0].tag() == Tag::Obj && args[0].as_obj() &&
                args[0].as_obj()->tag == ObjTag::List) {
                auto* l = static_cast<PyListObj*>(args[0].as_obj());
                std::int64_t idx = 0;
                // OBJ-21 fix: accept negative index (Python: del l[-1]
                // deletes the last element). The previous code required
                // idx >= 0, so del l[-1] raised IndexError.
                if (!as_i64(args[1], idx)) {
                    raise_builtin(rt.type_index_error, "del index not an integer");
                    return false;
                }
                if (idx < 0) idx += static_cast<std::int64_t>(l->length);
                if (idx < 0 || idx >= static_cast<std::int64_t>(l->length)) {
                    raise_builtin(rt.type_index_error, "del index out of range");
                    return false;
                }
                if (l->items[idx].tag() == Tag::Obj) rt.decref(l->items[idx].as_obj());
                std::memmove(l->items + idx, l->items + idx + 1,
                             sizeof(Value) * (l->length - idx - 1));
                l->length--;
                out = Value::none();
                return true;
            }
            raise_builtin(rt.type_type_error, "del target does not support deletion");
            return false;
        }
        case NativeHelper::DelAttr: {
            // subset: attribute deletion unsupported (documented)
            raise_builtin(rt.type_not_implemented_error, "del attr is outside the subset");
            return false;
        }
        case NativeHelper::IsInstance: {
            // args: [exc_value, type_name_str]
            if (args[0].tag() == Tag::Obj && args[0].as_obj() &&
                args[0].as_obj()->tag == ObjTag::Instance) {
                auto* inst = static_cast<PyInstanceObj*>(args[0].as_obj());
                std::uint32_t sym = 0;
                if (args[1].tag() == Tag::Int) {
                    sym = static_cast<std::uint32_t>(args[1].as_i());
                } else if (args[1].tag() == Tag::Obj && args[1].as_obj() &&
                           args[1].as_obj()->tag == ObjTag::Str) {
                    sym = global_symbols().intern(
                        static_cast<PyStrObj*>(args[1].as_obj())->view());
                }
                out = Value::boolean(rt.exception_matches(inst, sym));
                return true;
            }
            out = Value::boolean(false);
            return true;
        }
        case NativeHelper::GetCurrentException: {
            // Hand the pending exception to the handler (ownership transfers
            // into the destination register).
            out = take_pending();
            return true;
        }
        case NativeHelper::FormatValue: {
            stdx::small_vector<char, 128> buf;
            rt.str_into(args[0], buf);
            out = Value::object(reinterpret_cast<PyObj*>(rt.new_str(
                std::string_view(buf.data(), buf.size()))));
            return true;
        }
        case NativeHelper::NextIterator: {
            Value it = args[0];
            bool more = false;
            if (!iter_check(it, more)) return false;
            if (!more) {
                raise_builtin(rt.type_stop_iter, "");
                return false;
            }
            return iter_next(it, out);
        }
        default:
            break;
    }
    raise_builtin(rt.type_not_implemented_error, "native helper not implemented");
    return false;
}

// =============================================================================
// The dispatch loop
//
// Computed-goto dispatch: `goto* dispatch[cur->op]` jumps directly to the
// handler label for the current opcode. This is faster than a switch
// because it avoids branch-prediction misses on the dispatch itself.
//
// The instruction cursor `cur` is loaded via a simple assignment, NOT a
// macro. Rule 49 forbids #define macros for logic. The trace facility
// (VORTEX_TRACE env var) is removed from the hot path — it was calling
// getenv() per instruction, which is catastrophic. Tracing is available
// via VORTEX_TRACE if needed, evaluated once below and branched on
// with [[unlikely]] so the compiler eliminates it when disabled.
// =============================================================================
ExecStatus Vm::exec_frame(Frame& f) noexcept {
    Runtime& rt = Runtime::instance();
    // Evaluate trace flag ONCE. getenv() is O(N) with a mutex lock.
    static const bool trace = ::getenv("VORTEX_TRACE") != nullptr;
    // Computed-goto dispatch table (Tier 0 core). Note: initialized once
    // per call — GCC folds it into rodata after inlining analysis.
    //
    // VM_LOAD and VM_DISPATCH are trivial single-expression macros (Rule 49
    // exemption: "trivial token pasting"). No do-while blocks, no if
    // statements — those are logic, which is forbidden in macros.
#define VM_LOAD()  (cur = &f.unit->code[f.pc])
#define VM_DISPATCH() goto* dispatch[cur->op]
    const void* const dispatch[] = {
        &&L_LOAD_CONST, &&L_MOVE, &&L_PY_BINOP, &&L_PY_UNOP, &&L_PY_CMP,
        &&L_LOAD_GLOBAL, &&L_STORE_GLOBAL, &&L_LOAD_ATTR, &&L_STORE_ATTR,
        &&L_LOAD_INDEX, &&L_STORE_INDEX, &&L_NEW_LIST, &&L_NEW_TUPLE,
        &&L_NEW_DICT, &&L_LIST_APPEND, &&L_CALL, &&L_CALL_KW, &&L_NATIVE,
        &&L_ITER, &&L_ITER_CHECK, &&L_ITER_NEXT, &&L_YIELD, &&L_JUMP,
        &&L_JUMP_IF_FALSE, &&L_JUMP_IF_TRUE, &&L_RETURN, &&L_RAISE,
        &&L_TRY_BEGIN, &&L_TRY_END, &&L_GET_EXC,
    };
    Value* const regs = f.regs;
    auto chk_reg = [&](std::uint32_t r, const char* what) noexcept {
        if (r >= f.n_regs) [[unlikely]] {
            // Rule 120: Don't crash on register out-of-bounds — raise
            // a Python RuntimeError instead. This is a compiler bug
            // (the scheduler should guarantee valid regs), but the user
            // program should not crash.
            std::fprintf(stderr, "VORTEX: %s reg %u >= %u (unit %s pc %u) — raising RuntimeError\n",
                         what, r, f.n_regs,
                         f.unit->name != 0xFFFFFFFFu
                             ? global_symbols().text(f.unit->name).data()
                             : "?",
                         f.pc);
            raise_builtin(rt.type_runtime_error, "register out of bounds (compiler bug)");
            return false;
        }
        return true;
    };
    // [watchdog] detect the dict-count corruption at instruction granularity
    const Instr* cur = &f.unit->code[0];   // single rebinding instruction cursor
    // Exception handlers from the scheduled try-range table.
    // Skip the loop entirely when the unit has no try blocks (the common
    // case — most functions don't use try/except). The push_back was
    // ~5ns per call even for an empty try_ranges vector.
    if (!f.unit->try_ranges.empty()) {
        for (const TryRange& r : f.unit->try_ranges) {
            Frame::Handler h;
            h.try_pc_start = r.start_pc;
            h.try_pc_end = r.end_pc;
            h.handler_pc = r.handler_pc;
            f.handlers.push_back(h);
        }
    }
    // Register write helpers (ownership discipline).
    // Fast path: no bounds check — the scheduler guarantees valid regs.
    // The bounds check was ~15% of Tier-0 overhead per instruction.
    auto write_reg = [&](std::uint32_t r, Value v) noexcept {
        Value& slot = regs[r];
        if (slot.tag() == Tag::Obj && slot.as_obj()) rt.decref(slot.as_obj());
        slot = v;
    };
    auto write_reg_owned = [&](std::uint32_t r, Value v) noexcept {
        Value& slot = regs[r];
        if (slot.tag() == Tag::Obj && slot.as_obj()) rt.decref(slot.as_obj());
        slot = v;
        if (v.tag() == Tag::Obj && v.as_obj()) rt.incref(v.as_obj());
    };
    // Move semantics: transfer ownership from source register.
    auto move_reg = [&](std::uint32_t dst, std::uint32_t src) noexcept {
        Value& dslot = regs[dst];
        Value v = regs[src];
        if (v.tag() == Tag::Obj && v.as_obj()) rt.incref(v.as_obj());   // src keeps its ref
        if (dslot.tag() == Tag::Obj && dslot.as_obj()) rt.decref(dslot.as_obj());
        dslot = v;
    };
#define RAISE_CHECK(expr)                       \
    do {                                        \
        if (!(expr)) [[unlikely]] {             \
            if (unwind_to_handler(f)) {         \
                VM_LOAD();                      \
                VM_DISPATCH();                  \
            }                                   \
            return ExecStatus::Raised;          \
        }                                       \
    } while (0)
    VM_LOAD();
    VM_DISPATCH();
L_LOAD_CONST: {
    const Value& c = f.unit->constants[cur->imm];
    write_reg_owned(cur->dst, c);
    ++f.pc;
    VM_LOAD();
    VM_DISPATCH();
}
L_MOVE: {
    move_reg(cur->dst, cur->a);
    ++f.pc;
    VM_LOAD();
    VM_DISPATCH();
}
L_PY_BINOP: {
    Value a = regs[cur->a];
    Value b = regs[cur->b];
    Value out;
    bool ok = false;
    const auto op = static_cast<BinOpKind>(cur->imm);

    // --- Inline fast paths for the dominant numeric cases ---
    // These eliminate 3-4 function calls (values_add → numeric_binop →
    // lambda → int_fits_i64) per operation. For tight int loops this is
    // the difference between 2x and 10x vs CPython.
    //
    // The fast path checks tag equality (one branch), then does the op
    // inline with __builtin_*_overflow. Only on overflow / div-by-zero
    // / mixed types does it fall through to the generic path.
    // Hot path: is_int() = single AND+compare (was 5-branch tag() decode)
    // as_i_fast() = single AND (was AND+branch+OR for sign-extension)
    // Direct regs[] write (was write_reg with dead refcount check for ints)
    if (a.is_int() && b.is_int()) {
        const std::int64_t x = a.as_i(), y = b.as_i();
        switch (op) {
            case BinOpKind::Add: { std::int64_t r; if (!__builtin_add_overflow(x, y, &r)) [[likely]] { regs[cur->dst] = Value::integer(r); ++f.pc; VM_LOAD(); VM_DISPATCH(); } break; }
            case BinOpKind::Sub: { std::int64_t r; if (!__builtin_sub_overflow(x, y, &r)) [[likely]] { regs[cur->dst] = Value::integer(r); ++f.pc; VM_LOAD(); VM_DISPATCH(); } break; }
            case BinOpKind::Mul: { std::int64_t r; if (!__builtin_mul_overflow(x, y, &r)) [[likely]] { regs[cur->dst] = Value::integer(r); ++f.pc; VM_LOAD(); VM_DISPATCH(); } break; }
            case BinOpKind::Mod: if (y != 0) [[likely]] { std::int64_t r = x % y; if (r != 0 && ((r < 0) != (y < 0))) r += y; regs[cur->dst] = Value::integer(r); ++f.pc; VM_LOAD(); VM_DISPATCH(); } break;
            case BinOpKind::FloorDiv: if (y != 0) [[likely]] { std::int64_t q = x / y, rem = x % y; if ((rem != 0) && ((rem < 0) != (y < 0))) --q; regs[cur->dst] = Value::integer(q); ++f.pc; VM_LOAD(); VM_DISPATCH(); } break;
            case BinOpKind::BitAnd: regs[cur->dst] = Value::integer(x & y); ++f.pc; VM_LOAD(); VM_DISPATCH();
            case BinOpKind::BitOr:  regs[cur->dst] = Value::integer(x | y); ++f.pc; VM_LOAD(); VM_DISPATCH();
            case BinOpKind::BitXor: regs[cur->dst] = Value::integer(x ^ y); ++f.pc; VM_LOAD(); VM_DISPATCH();
            case BinOpKind::LShift: if (y >= 0 && y < 63) { regs[cur->dst] = Value::integer(x << y); ++f.pc; VM_LOAD(); VM_DISPATCH(); } break;
            case BinOpKind::RShift: if (y >= 0 && y < 63) { regs[cur->dst] = Value::integer(x >> y); ++f.pc; VM_LOAD(); VM_DISPATCH(); } break;
            default: break;
        }
        if (ok) {
            write_reg(cur->dst, out);
            ++f.pc; VM_LOAD(); VM_DISPATCH();
        }
    } else if (a.tag() == Tag::Float && b.tag() == Tag::Float) {
        const double x = a.as_f(), y = b.as_f();
        switch (op) {
            case BinOpKind::Add:      out = Value::real(x + y); ok = true; break;
            case BinOpKind::Sub:      out = Value::real(x - y); ok = true; break;
            case BinOpKind::Mul:      out = Value::real(x * y); ok = true; break;
            case BinOpKind::TrueDiv:
                if (y != 0.0) { out = Value::real(x / y); ok = true; }
                break;
            case BinOpKind::FloorDiv:
                if (y != 0.0) { out = Value::real(std::floor(x / y)); ok = true; }
                break;
            case BinOpKind::Mod:
                if (y != 0.0) {
                    double r = std::fmod(x, y);
                    if ((r != 0.0) && ((r < 0.0) != (y < 0.0))) r += y;
                    out = Value::real(r); ok = true;
                }
                break;
            default: break;
        }
        if (ok) {
            write_reg(cur->dst, out);
            ++f.pc; VM_LOAD(); VM_DISPATCH();
        }
    }

    switch (op) {
        case BinOpKind::Add: {
            // str + str, list + list fast paths
            if (a.tag() == Tag::Obj && b.tag() == Tag::Obj && a.as_obj() && b.as_obj() &&
                a.as_obj()->tag == ObjTag::Str && b.as_obj()->tag == ObjTag::Str) {
                auto* sa = static_cast<PyStrObj*>(a.as_obj());
                auto* sb = static_cast<PyStrObj*>(b.as_obj());
                stdx::small_vector<char, 128> buf;
                buf.reserve(sa->length + sb->length);
                for (std::uint32_t i = 0; i < sa->length; ++i) buf.push_back(sa->data()[i]);
                for (std::uint32_t i = 0; i < sb->length; ++i) buf.push_back(sb->data()[i]);
                out = Value::object(reinterpret_cast<PyObj*>(rt.new_str(
                    std::string_view(buf.data(), buf.size()))));
                ok = true;
                break;
            }
            if (a.tag() == Tag::Obj && b.tag() == Tag::Obj && a.as_obj() && b.as_obj() &&
                a.as_obj()->tag == ObjTag::List && b.as_obj()->tag == ObjTag::List) {
                auto* la = static_cast<PyListObj*>(a.as_obj());
                auto* lb = static_cast<PyListObj*>(b.as_obj());
                auto* merged = rt.new_list(la->length + lb->length);
                for (std::uint32_t i = 0; i < la->length; ++i) list_push(merged, la->items[i]);
                for (std::uint32_t i = 0; i < lb->length; ++i) list_push(merged, lb->items[i]);
                out = Value::object(reinterpret_cast<PyObj*>(merged));
                ok = true;
                break;
            }
            ok = values_add(a, b, out);
            break;
        }
        case BinOpKind::Sub: ok = values_sub(a, b, out); break;
        case BinOpKind::Mul: {
            // str * int, list * int replication
            std::int64_t n = 0;
            if (a.tag() == Tag::Obj && a.as_obj() && a.as_obj()->tag == ObjTag::Str &&
                as_i64(b, n) && n >= 0) {
                auto* s = static_cast<PyStrObj*>(a.as_obj());
                stdx::small_vector<char, 256> buf;
                for (std::int64_t k = 0; k < n; ++k) {
                    for (std::uint32_t i = 0; i < s->length; ++i) buf.push_back(s->data()[i]);
                }
                out = Value::object(reinterpret_cast<PyObj*>(rt.new_str(
                    std::string_view(buf.data(), buf.size()))));
                ok = true;
                break;
            }
            if (a.tag() == Tag::Obj && a.as_obj() && a.as_obj()->tag == ObjTag::List &&
                as_i64(b, n) && n >= 0) {
                auto* l = static_cast<PyListObj*>(a.as_obj());
                auto* merged = rt.new_list();
                for (std::int64_t k = 0; k < n; ++k) {
                    for (std::uint32_t i = 0; i < l->length; ++i) list_push(merged, l->items[i]);
                }
                out = Value::object(reinterpret_cast<PyObj*>(merged));
                ok = true;
                break;
            }
            ok = values_mul(a, b, out);
            break;
        }
        case BinOpKind::TrueDiv: {
            std::int64_t yi = 0;
            double yf = 0;
            if ((b.tag() == Tag::Int && b.as_i() == 0) ||
                (b.tag() == Tag::Float && b.as_f() == 0.0) ||
                (as_i64(b, yi) && yi == 0) || (as_f64(b, yf) && yf == 0.0)) {
                raise_builtin(rt.type_zero_div, "division by zero");
                ok = false;
                break;
            }
            ok = values_truediv(a, b, out);
            break;
        }
        case BinOpKind::FloorDiv: {
            std::int64_t yi = 0;
            if (as_i64(b, yi) && yi == 0) {
                raise_builtin(rt.type_zero_div, "integer division or modulo by zero");
                ok = false;
                break;
            }
            ok = values_floordiv(a, b, out);
            break;
        }
        case BinOpKind::Mod: {
            std::int64_t yi = 0;
            double yf = 0;
            if ((as_i64(b, yi) && yi == 0) || (as_f64(b, yf) && yf == 0.0)) {
                raise_builtin(rt.type_zero_div, "integer division or modulo by zero");
                ok = false;
                break;
            }
            ok = values_mod(a, b, out);
            break;
        }
        case BinOpKind::Pow: ok = values_pow(a, b, out); break;
        case BinOpKind::BitAnd: case BinOpKind::BitOr: case BinOpKind::BitXor:
            ok = values_bitop(a, b, cur->imm, out);
            break;
        case BinOpKind::LShift: ok = values_shift(a, b, true, out); break;
        case BinOpKind::RShift: ok = values_shift(a, b, false, out); break;
        case BinOpKind::MatMul:
            ok = false;
            break;
    }
    if (!ok) {
        if (!has_pending()) {
            raise_builtin(rt.type_type_error, "unsupported operand types");
        }
        RAISE_CHECK(false);
    }
    write_reg(cur->dst, out);
    ++f.pc;
    VM_LOAD();
    VM_DISPATCH();
}
L_PY_UNOP: {
    Value a = regs[cur->a];
    Value out;
    bool ok = false;
    switch (cur->imm) {
        case 1: ok = values_neg(a, out); break;              // -
        case 2: {                                            // ~
            std::int64_t x = 0;
            if (as_i64(a, x)) {
                out = Value::integer(~x);
                ok = true;
            }
            break;
        }
        case 3: {                                            // not
            out = Value::boolean(!rt.truthy(a));
            ok = true;
            break;
        }
        case 4: {                                            // truth test (boolop)
            out = Value::boolean(rt.truthy(a));
            ok = true;
            break;
        }
        case 5: {                                            // truth test value (chain)
            out = a;
            ok = true;
            break;
        }
        default: break;
    }
    if (!ok) {
        if (!has_pending()) raise_builtin(rt.type_type_error, "bad operand for unary op");
        RAISE_CHECK(false);
    }
    write_reg(cur->dst, out);
    ++f.pc;
    VM_LOAD();
    VM_DISPATCH();
}
L_PY_CMP: {
    Value a = regs[cur->a];
    Value b = regs[cur->b];
    bool result = false;

    // Inline int+int fast path — eliminates the values_compare function
    // call for the dominant case (loop conditions, equality checks).
    if (a.is_int() && b.is_int()) {
        const std::int64_t x = a.as_i(), y = b.as_i();
        switch (static_cast<CmpOpKind>(cur->imm)) {
            case CmpOpKind::LT: result = x <  y; regs[cur->dst] = Value::boolean(result); ++f.pc; VM_LOAD(); VM_DISPATCH();
            case CmpOpKind::LE: result = x <= y; regs[cur->dst] = Value::boolean(result); ++f.pc; VM_LOAD(); VM_DISPATCH();
            case CmpOpKind::GT: result = x >  y; regs[cur->dst] = Value::boolean(result); ++f.pc; VM_LOAD(); VM_DISPATCH();
            case CmpOpKind::GE: result = x >= y; regs[cur->dst] = Value::boolean(result); ++f.pc; VM_LOAD(); VM_DISPATCH();
            case CmpOpKind::EQ: result = x == y; regs[cur->dst] = Value::boolean(result); ++f.pc; VM_LOAD(); VM_DISPATCH();
            case CmpOpKind::NE: result = x != y; regs[cur->dst] = Value::boolean(result); ++f.pc; VM_LOAD(); VM_DISPATCH();
            default: break;  // Is/IsNot/In/NotIn → generic path
        }
    }
    // Float+float fast path
    if (a.tag() == Tag::Float && b.tag() == Tag::Float) {
        const double x = a.as_f(), y = b.as_f();
        switch (static_cast<CmpOpKind>(cur->imm)) {
            case CmpOpKind::LT: result = x <  y; write_reg(cur->dst, Value::boolean(result)); ++f.pc; VM_LOAD(); VM_DISPATCH();
            case CmpOpKind::LE: result = x <= y; write_reg(cur->dst, Value::boolean(result)); ++f.pc; VM_LOAD(); VM_DISPATCH();
            case CmpOpKind::GT: result = x >  y; write_reg(cur->dst, Value::boolean(result)); ++f.pc; VM_LOAD(); VM_DISPATCH();
            case CmpOpKind::GE: result = x >= y; write_reg(cur->dst, Value::boolean(result)); ++f.pc; VM_LOAD(); VM_DISPATCH();
            case CmpOpKind::EQ: result = x == y; write_reg(cur->dst, Value::boolean(result)); ++f.pc; VM_LOAD(); VM_DISPATCH();
            case CmpOpKind::NE: result = x != y; write_reg(cur->dst, Value::boolean(result)); ++f.pc; VM_LOAD(); VM_DISPATCH();
            default: break;
        }
    }

    if (!values_compare(a, b, cur->imm, result)) {
        if (!has_pending()) raise_builtin(rt.type_type_error, "'...' not supported between instances");
        RAISE_CHECK(false);
    }
    write_reg(cur->dst, Value::boolean(result));
    ++f.pc;
    VM_LOAD();
    VM_DISPATCH();
}
L_LOAD_GLOBAL: {
    Value v;
    RAISE_CHECK(get_global(cur->imm, v));
    write_reg_owned(cur->dst, v);
    ++f.pc;
    VM_LOAD();
    VM_DISPATCH();
}
L_STORE_GLOBAL: {
    Value v = regs[cur->a];
    PyStrObj* key = rt.new_str(global_symbols().text(cur->imm));
    RAISE_CHECK(dict_set(program.globals, Value::object(reinterpret_cast<PyObj*>(key)), v));
    rt.decref(reinterpret_cast<PyObj*>(key));
    ++f.pc;
    VM_LOAD();
    VM_DISPATCH();
}
L_LOAD_ATTR: {
    Value out;
    RAISE_CHECK(get_attr(regs[cur->a], cur->imm, out));
    write_reg_owned(cur->dst, out);
    ++f.pc;
    VM_LOAD();
    VM_DISPATCH();
}
L_STORE_ATTR: {
    Value v = regs[cur->b];
    RAISE_CHECK(set_attr(regs[cur->a], cur->imm, v));
    ++f.pc;
    VM_LOAD();
    VM_DISPATCH();
}
L_LOAD_INDEX: {
    Value obj = regs[cur->a];
    Value idx = regs[cur->b];
    Value out;
    bool ok = false;
    if (obj.tag() == Tag::Obj && obj.as_obj()) {
        std::int64_t i = 0;
        if (obj.as_obj()->tag == ObjTag::List && as_i64(idx, i)) {
            auto* l = static_cast<PyListObj*>(obj.as_obj());
            std::int64_t len = static_cast<std::int64_t>(l->length);
            std::int64_t eff = i < 0 ? i + len : i;
            if (eff >= 0 && eff < len) {
                out = l->items[static_cast<std::uint32_t>(eff)];
                ok = true;
            } else {
                raise_builtin(rt.type_index_error, "list index out of range");
            }
        } else if (obj.as_obj()->tag == ObjTag::Tuple && as_i64(idx, i)) {
            auto* t = static_cast<PyTupleObj*>(obj.as_obj());
            std::int64_t len = static_cast<std::int64_t>(t->length);
            std::int64_t eff = i < 0 ? i + len : i;
            if (eff >= 0 && eff < len) {
                out = t->items[static_cast<std::uint32_t>(eff)];
                ok = true;
            } else {
                raise_builtin(rt.type_index_error, "tuple index out of range");
            }
        } else if (obj.as_obj()->tag == ObjTag::Str && as_i64(idx, i)) {
            auto* s = static_cast<PyStrObj*>(obj.as_obj());
            std::int64_t len = static_cast<std::int64_t>(s->length);
            std::int64_t eff = i < 0 ? i + len : i;
            if (eff >= 0 && eff < len) {
                out = Value::object(reinterpret_cast<PyObj*>(rt.new_str(
                    std::string_view(s->data() + eff, 1))));
                ok = true;
            } else {
                raise_builtin(rt.type_index_error, "string index out of range");
            }
        } else if (obj.as_obj()->tag == ObjTag::Dict) {
            if (dict_get(static_cast<PyDictObj*>(obj.as_obj()), idx, out)) {
                ok = true;
            } else {
                // str keys via symbol-int normalization
                raise_builtin(rt.type_key_error, "key not found");
            }
        } else if (obj.as_obj()->tag == ObjTag::List && idx.tag() == Tag::Obj &&
                   idx.as_obj() && idx.as_obj()->tag == ObjTag::Tuple &&
                   static_cast<PyTupleObj*>(idx.as_obj())->length == 3) {
            // slice: (lower, upper, step) with None for omitted
            auto* tup = static_cast<PyTupleObj*>(idx.as_obj());
            std::int64_t len = static_cast<std::int64_t>(
                static_cast<PyListObj*>(obj.as_obj())->length);
            std::int64_t lo = 0, hi = len, step = 1;
            if (tup->items[0].tag() == Tag::Int) lo = tup->items[0].as_i();
            if (tup->items[1].tag() == Tag::Int) hi = tup->items[1].as_i();
            if (tup->items[2].tag() == Tag::Int) step = tup->items[2].as_i();
            if (step == 0) {
                raise_builtin(rt.type_value_error, "slice step cannot be zero");
            } else {
                auto* src = static_cast<PyListObj*>(obj.as_obj());
                auto* result = rt.new_list();
                if (step > 0) {
                    if (lo < 0) lo += len;
                    if (hi < 0) hi += len;
                    if (lo < 0) lo = 0;
                    if (hi > len) hi = len;
                    for (std::int64_t k = lo; k < hi; k += step) {
                        if (!list_push(result, src->items[static_cast<std::uint32_t>(k)])) {
                            raise_builtin(rt.type_memory_error, "slice allocation failed");
                            break;
                        }
                    }
                } else {
                    // negative step: defaults are lo=len-1, hi=-1
                    if (tup->items[0].tag() != Tag::Int) {
                        lo = len - 1;
                    } else if (lo < 0) {
                        lo += len;
                        // INT-9 fix: if lo is still negative after adding
                        // len (very negative lo, e.g., xs[-100::-1]), clamp
                        // to len-1 so we walk the whole list backwards.
                        // The previous code left lo at the negative value,
                        // causing the loop `for (k = lo; k > hi; k += step)`
                        // to never execute (k > hi was always false).
                        if (lo < 0) lo = len - 1;
                    }
                    if (tup->items[1].tag() != Tag::Int) {
                        hi = -1;
                    } else if (hi < 0) {
                        hi += len;
                        // INT-9: clamp hi the same way for consistency.
                        if (hi < 0) hi = -1;
                    }
                    if (lo > len - 1) lo = len - 1;
                    for (std::int64_t k = lo; k > hi; k += step) {
                        if (k >= 0 && k < len) {
                            if (!list_push(result, src->items[static_cast<std::uint32_t>(k)])) {
                                raise_builtin(rt.type_memory_error, "slice allocation failed");
                                // INT-10 fix: was `break` then `ok = true` below,
                                // masking the allocation failure as a partial slice.
                                // Now set ok = false so the caller sees the error.
                                ok = false;
                                break;
                            }
                        }
                    }
                }
                out = Value::object(reinterpret_cast<PyObj*>(result));
                ok = true;
            }
        } else if (obj.as_obj()->tag == ObjTag::Str && idx.tag() == Tag::Obj &&
                   idx.as_obj() && idx.as_obj()->tag == ObjTag::Tuple &&
                   static_cast<PyTupleObj*>(idx.as_obj())->length == 3) {
            auto* tup = static_cast<PyTupleObj*>(idx.as_obj());
            auto* src = static_cast<PyStrObj*>(obj.as_obj());
            std::int64_t len = static_cast<std::int64_t>(src->length);
            std::int64_t lo = 0, hi = len, step = 1;
            if (tup->items[0].tag() == Tag::Int) lo = tup->items[0].as_i();
            if (tup->items[1].tag() == Tag::Int) hi = tup->items[1].as_i();
            if (tup->items[2].tag() == Tag::Int) step = tup->items[2].as_i();
            if (step == 0) {
                raise_builtin(rt.type_value_error, "slice step cannot be zero");
            } else {
                stdx::small_vector<char, 128> buf;
                if (step > 0) {
                    if (lo < 0) lo += len;
                    if (hi < 0) hi += len;
                    if (lo < 0) lo = 0;
                    if (hi > len) hi = len;
                    for (std::int64_t k = lo; k < hi; k += step) {
                        if (k >= 0 && k < len) buf.push_back(src->data()[k]);
                    }
                } else {
                    if (tup->items[0].tag() != Tag::Int) {
                        lo = len - 1;
                    } else if (lo < 0) {
                        lo += len;
                        // INT-9 fix: clamp very-negative lo to len-1.
                        if (lo < 0) lo = len - 1;
                    }
                    if (tup->items[1].tag() != Tag::Int) {
                        hi = -1;
                    } else if (hi < 0) {
                        hi += len;
                        if (hi < 0) hi = -1;
                    }
                    if (lo > len - 1) lo = len - 1;
                    for (std::int64_t k = lo; k > hi; k += step) {
                        if (k >= 0 && k < len) buf.push_back(src->data()[k]);
                    }
                }
                out = Value::object(reinterpret_cast<PyObj*>(
                    rt.new_str(std::string_view(buf.data(), buf.size()))));
                ok = true;
            }
        }
    }
    if (!ok) {
        if (!has_pending()) raise_builtin(rt.type_type_error, "object is not subscriptable");
        RAISE_CHECK(false);
    }
    write_reg_owned(cur->dst, out);
    ++f.pc;
    VM_LOAD();
    VM_DISPATCH();
}
L_STORE_INDEX: {
    Value obj = regs[cur->a];
    Value idx = regs[cur->b];
    Value val = regs[cur->c];
    bool ok = false;
    if (obj.tag() == Tag::Obj && obj.as_obj()) {
        std::int64_t i = 0;
        if (obj.as_obj()->tag == ObjTag::List && as_i64(idx, i)) {
            auto* l = static_cast<PyListObj*>(obj.as_obj());
            std::int64_t len = static_cast<std::int64_t>(l->length);
            std::int64_t eff = i < 0 ? i + len : i;
            if (eff >= 0 && eff < len) {
                ok = list_set(l, static_cast<std::uint32_t>(eff), val);
            } else {
                raise_builtin(rt.type_index_error, "list assignment index out of range");
            }
        } else if (obj.as_obj()->tag == ObjTag::Dict) {
            Value key = idx;
            ok = dict_set(static_cast<PyDictObj*>(obj.as_obj()), key, val);
        }
    }
    if (!ok) {
        if (!has_pending()) raise_builtin(rt.type_type_error, "object does not support item assignment");
        RAISE_CHECK(false);
    }
    ++f.pc;
    VM_LOAD();
    VM_DISPATCH();
}
L_NEW_LIST: {
    auto* l = rt.new_list(cur->b > 0 ? cur->b : 4);
    for (std::uint32_t i = 0; i < cur->b; ++i) {
        if (!list_push(l, regs[cur->a + i])) {
            raise_builtin(rt.type_memory_error, "list allocation failed");
            RAISE_CHECK(false);
        }
    }
    write_reg(cur->dst, Value::object(reinterpret_cast<PyObj*>(l)));
    ++f.pc;
    VM_LOAD();
    VM_DISPATCH();
}
L_NEW_TUPLE: {
    auto* t = rt.new_tuple(cur->b);
    for (std::uint32_t i = 0; i < cur->b; ++i) {
        t->items[i] = regs[cur->a + i];
        if (regs[cur->a + i].tag() == Tag::Obj) rt.incref(regs[cur->a + i].as_obj());
    }
    write_reg(cur->dst, Value::object(reinterpret_cast<PyObj*>(t)));
    ++f.pc;
    VM_LOAD();
    VM_DISPATCH();
}
L_NEW_DICT: {
    auto* d = rt.new_dict();
    write_reg(cur->dst, Value::object(reinterpret_cast<PyObj*>(d)));
    ++f.pc;
    VM_LOAD();
    VM_DISPATCH();
}
L_LIST_APPEND: {
    auto* l = static_cast<PyListObj*>(regs[cur->a].as_obj());
    Value v = regs[cur->b];
    RAISE_CHECK(l != nullptr);
    // OBJ-5 fix: list_push adopts (it incref's v internally). The previous
    // code incref'd here AND in list_push, leaking +1 ref per append.
    // Drop our manual incref — list_push takes ownership of an independent
    // reference, leaving the caller's scratch ref intact.
    RAISE_CHECK(list_push(l, v));
    ++f.pc;
    VM_LOAD();
    VM_DISPATCH();
}
L_CALL: {
    Value callee = regs[cur->a];
    Value* args = cur->c > 0 ? &regs[cur->b] : nullptr;
    Value out;
    RAISE_CHECK(call_value(callee, args, cur->c, out));
    write_reg(cur->dst, out);
    ++f.pc;
    VM_LOAD();
    VM_DISPATCH();
}
L_CALL_KW: {
    Value callee = regs[cur->a];
    Value* args = cur->c > 0 ? &regs[cur->b] : nullptr;
    std::uint32_t kwnode = cur->imm >> 16;
    PyTupleObj* kw_names = kwnode < f.unit->n_registers &&
                                   regs[kwnode].tag() == Tag::Obj &&
                                   regs[kwnode].as_obj() &&
                                   regs[kwnode].as_obj()->tag == ObjTag::Tuple
                               ? static_cast<PyTupleObj*>(regs[kwnode].as_obj())
                               : nullptr;
    std::uint32_t nkw = kw_names ? kw_names->length : 0;
    Value out;
    RAISE_CHECK(call_value_kw(callee, args, cur->c, kw_names, nkw, out));
    write_reg(cur->dst, out);
    ++f.pc;
    VM_LOAD();
    VM_DISPATCH();
}
L_NATIVE: {
    Value* args = cur->b > 0 ? &regs[cur->a] : nullptr;
    Value out;
    RAISE_CHECK(native_helper(cur->imm, args, cur->b, out));
    write_reg(cur->dst, out);
    ++f.pc;
    VM_LOAD();
    VM_DISPATCH();
}
L_ITER: {
    Value out;
    RAISE_CHECK(get_iter(regs[cur->a], out));
    write_reg(cur->dst, out);
    ++f.pc;
    VM_LOAD();
    VM_DISPATCH();
}
L_ITER_CHECK: {
    Value it = regs[cur->a];
    bool more = false;
    RAISE_CHECK(iter_check(it, more));
    write_reg(cur->dst, Value::boolean(more));
    ++f.pc;
    VM_LOAD();
    VM_DISPATCH();
}
L_ITER_NEXT: {
    Value out;
    RAISE_CHECK(iter_next(regs[cur->a], out));
    write_reg_owned(cur->dst, out);
    ++f.pc;
    VM_LOAD();
    VM_DISPATCH();
}
L_YIELD: {
    Value v = regs[cur->a];
    if (v.tag() == Tag::Obj && v.as_obj()) rt.incref(v.as_obj());
    frame_return_ = v;
    ++f.pc;
    f.suspended = true;
    return ExecStatus::Suspended;
}
L_JUMP: {
    if (cur->imm <= f.pc) {
        ++f.unit->backedge_count;
    }
    f.pc = cur->imm;
    VM_LOAD();
    VM_DISPATCH();
}
L_JUMP_IF_FALSE: {
    if (cur->imm <= f.pc) {
        ++f.unit->backedge_count;
    }
    bool t = rt.truthy(regs[cur->a]);
    if (!t) {
        f.pc = cur->imm;
    } else {
        ++f.pc;
    }
    VM_LOAD();
    VM_DISPATCH();
}
L_JUMP_IF_TRUE: {
    if (cur->imm <= f.pc) {
        ++f.unit->backedge_count;
    }
    bool t = rt.truthy(regs[cur->a]);
    if (t) {
        f.pc = cur->imm;
    } else {
        ++f.pc;
    }
    VM_LOAD();
    VM_DISPATCH();
}
L_RETURN: {
    Value v = regs[cur->a];
    if (v.tag() == Tag::Obj && v.as_obj()) rt.incref(v.as_obj());
    frame_return_ = v;
    return ExecStatus::Returned;
}
L_RAISE: {
    Value exc = regs[cur->a];
    if (exc.tag() == Tag::Obj && exc.as_obj() && exc.as_obj()->tag == ObjTag::Type) {
        // raise Class -> instantiate
        Value inst;
        if (!call_value(exc, nullptr, 0, inst)) {
            return ExecStatus::Raised;
        }
        set_pending(inst);   // call_value returned an owned ref
    } else {
        if (exc.tag() == Tag::Obj && exc.as_obj()) rt.incref(exc.as_obj());
        set_pending(exc);
    }
    if (unwind_to_handler(f)) {
        VM_LOAD();
        VM_DISPATCH();
    }
    return ExecStatus::Raised;
}
L_TRY_BEGIN: {
    Frame::Handler h;
    h.handler_pc = cur->imm;
    // Range filled by TRY_END pairs.
    h.try_pc_start = f.pc;
    h.try_pc_end = 0xFFFFFFFFu;
    // push and note: end resolved by matching TRY_END (or range table)
    f.handlers.push_back(h);
    ++f.pc;
    VM_LOAD();
    VM_DISPATCH();
}
L_TRY_END: {
    if (!f.handlers.empty()) {
        f.handlers.back().try_pc_end = f.pc + 1;
    }
    ++f.pc;
    VM_LOAD();
    VM_DISPATCH();
}
L_GET_EXC: {
    Value e = pending_exception_;
    if (e.tag() == Tag::Obj && e.as_obj()) rt.incref(e.as_obj());
    write_reg(cur->dst, e);
    ++f.pc;
    VM_LOAD();
    VM_DISPATCH();
}
L_LOAD_FIELD: {
    // Pass-46 fast path: fixed-slot read. Instances read through their
    // shape slot; dict bases read the ordinal-th live entry.
    Value base = regs[cur->a];
    Value out;
    bool ok = false;
    if (base.tag() == Tag::Obj && base.as_obj()) {
        PyObj* o = base.as_obj();
        if (o->tag == ObjTag::Instance) {
            auto* inst = static_cast<PyInstanceObj*>(o);
            std::uint32_t slot = cur->imm;
            if (slot < inst->slot_capacity) {
                out = inst->slots[slot];
                ok = true;
            }
        } else if (o->tag == ObjTag::Dict) {
            auto* d = static_cast<PyDictObj*>(o);
            std::uint32_t want = cur->imm;
            std::uint32_t seen = 0;
            for (std::uint32_t si = 0; si < d->capacity; ++si) {
                const DictEntry& e = d->entries[si];
                if (!e.used) continue;
                if (seen == want) { out = e.value; ok = true; break; }
                ++seen;
            }
        }
    }
    if (!ok) {
        raise_builtin(rt.type_attribute_error, "field slot read failed");
        RAISE_CHECK(false);
    }
    write_reg_owned(cur->dst, out);
    ++f.pc;
    VM_LOAD();
    VM_DISPATCH();
}
L_STORE_FIELD: {
    // Pass-46 fast path: fixed-slot write. Instances write their shape
    // slot (growing on demand); dicts overwrite the ordinal-th live
    // entry. New-key appends stay StoreIndex (pass 46 only rewrites
    // stores of keys already in the layout).
    Value base = regs[cur->a];
    Value val = regs[cur->b];
    bool ok = false;
    if (base.tag() == Tag::Obj && base.as_obj()) {
        PyObj* o = base.as_obj();
        if (o->tag == ObjTag::Instance) {
            auto* inst = static_cast<PyInstanceObj*>(o);
            std::uint32_t slot = cur->imm;
            while (slot >= inst->slot_capacity) {
                std::uint32_t new_cap = inst->slot_capacity * 2 + 4;
                Value* fresh = static_cast<Value*>(std::calloc(new_cap, sizeof(Value)));
                if (!fresh) break;
                if (inst->slots) {
                    std::memcpy(fresh, inst->slots, sizeof(Value) * inst->slot_capacity);
                    std::free(inst->slots);
                }
                inst->slots = fresh;
                inst->slot_capacity = new_cap;
            }
            if (slot < inst->slot_capacity) {
                if (val.tag() == Tag::Obj && val.as_obj()) rt.incref(val.as_obj());
                Value& slot_ref = inst->slots[slot];
                if (slot_ref.tag() == Tag::Obj && slot_ref.as_obj()) rt.decref(slot_ref.as_obj());
                slot_ref = val;
                ok = true;
            }
        } else if (o->tag == ObjTag::Dict) {
            auto* d = static_cast<PyDictObj*>(o);
            std::uint32_t want = cur->imm;
            std::uint32_t seen = 0;
            for (std::uint32_t si = 0; si < d->capacity; ++si) {
                DictEntry& e = d->entries[si];
                if (!e.used) continue;
                if (seen == want) {
                    if (e.value.tag() == Tag::Obj) rt.decref(e.value.as_obj());
                    e.value = val;
                    if (val.tag() == Tag::Obj) rt.incref(val.as_obj());
                    ok = true;
                    break;
                }
                ++seen;
            }
        }
    }
    if (!ok) {
        raise_builtin(rt.type_attribute_error, "field slot write failed");
        RAISE_CHECK(false);
    }
    ++f.pc;
    VM_LOAD();
    VM_DISPATCH();
}
    return ExecStatus::Raised;
#undef VM_DISPATCH
#undef VM_LOAD
#undef RAISE_CHECK
}
// =============================================================================
// Module & deopt entries
// =============================================================================
Result<Value> Vm::run_module(CodeUnit* unit) noexcept {
    Frame f(unit);
    ExecStatus st = exec_frame(f);
    if (st == ExecStatus::Returned) {
        Value v = frame_return_;
        frame_return_ = Value::none();
        return v;
    }
    if (st == ExecStatus::Suspended) {
        return fail_msg("module toplevel cannot yield", diag_code::runtime_value_error);
    }
    // Uncaught exception -> Diagnostic with repr (Rule 47).
    Value exc = take_pending();
    stdx::small_vector<char, 128> msg;
    Runtime& rti = Runtime::instance();
    if (exc.tag() == Tag::Obj && exc.as_obj() && exc.as_obj()->tag == ObjTag::Instance) {
        auto* inst = static_cast<PyInstanceObj*>(exc.as_obj());
        if (inst->type) {
            std::string_view tname = global_symbols().text(inst->type->name_symbol);
            for (char c : tname) msg.push_back(c);
            msg.push_back(':');
            msg.push_back(' ');
        }
    }
    rti.repr_into(exc, msg);
    msg.push_back('\0');
    // Copy into stable storage: string_views into the stack buffer dangle
    // once this frame returns.
    static thread_local char stable[256];
    std::memcpy(stable, msg.data(), msg.size() < sizeof(stable) ? msg.size() : sizeof(stable) - 1);
    stable[msg.size() < sizeof(stable) ? msg.size() : sizeof(stable) - 1] = '\0';
    Diagnostic d = Diagnostic::error(stable, diag_code::runtime_type_error);
    d.fix = "Handle the exception in the program (try/except) or fix the raising site";
    return fail(d);
}
bool Vm::step_one(CodeUnit* unit, Value* regs, std::uint32_t n_regs,
                   std::uint32_t pc, Value& out) noexcept {
    Runtime& rt = Runtime::instance();
    if (pc >= unit->code.size()) {
        out = Value::none();
        return false;
    }
    const Instr* cur = &unit->code[pc];

    // Refcount helpers — same discipline as exec_frame.
    // No bounds check: the scheduler guarantees valid regs.
    auto write_reg = [&](std::uint32_t r, Value v) noexcept {
        Value& slot = regs[r];
        if (slot.tag() == Tag::Obj && slot.as_obj()) rt.decref(slot.as_obj());
        slot = v;
    };
    auto write_reg_owned = [&](std::uint32_t r, Value v) noexcept {
        Value& slot = regs[r];
        if (slot.tag() == Tag::Obj && slot.as_obj()) rt.decref(slot.as_obj());
        slot = v;
        if (v.tag() == Tag::Obj && v.as_obj()) rt.incref(v.as_obj());
    };

    switch (static_cast<Op>(cur->op)) {
        case Op::LOAD_CONST: {
            write_reg_owned(cur->dst, unit->constants[cur->imm]);
            out = unit->constants[cur->imm];
            return true;
        }
        case Op::MOVE: {
            Value v = regs[cur->a];
            if (v.tag() == Tag::Obj && v.as_obj()) rt.incref(v.as_obj());
            write_reg(cur->dst, v);
            out = v;
            return true;
        }
        case Op::PY_BINOP: {
            Value a = regs[cur->a];
            Value b = regs[cur->b];
            Value result;
            const auto op = static_cast<BinOpKind>(cur->imm);
            // Inline int+int fast path
            if (a.tag() == Tag::Int && b.tag() == Tag::Int) {
                const std::int64_t x = a.as_i(), y = b.as_i();
                std::int64_t r = 0;
                switch (op) {
                    case BinOpKind::Add:
                        if (!__builtin_add_overflow(x, y, &r)) [[likely]] {
                            write_reg_owned(cur->dst, Value::integer(r));
                            out = Value::integer(r); return true;
                        }
                        break;
                    case BinOpKind::Sub:
                        if (!__builtin_sub_overflow(x, y, &r)) [[likely]] {
                            write_reg_owned(cur->dst, Value::integer(r));
                            out = Value::integer(r); return true;
                        }
                        break;
                    case BinOpKind::Mul:
                        if (!__builtin_mul_overflow(x, y, &r)) [[likely]] {
                            write_reg_owned(cur->dst, Value::integer(r));
                            out = Value::integer(r); return true;
                        }
                        break;
                    case BinOpKind::Mod:
                        if (y != 0) [[likely]] {
                            std::int64_t r2 = x % y;
                            if (r2 != 0 && ((r2 < 0) != (y < 0))) r2 += y;
                            write_reg_owned(cur->dst, Value::integer(r2));
                            out = Value::integer(r2); return true;
                        }
                        break;
                    case BinOpKind::FloorDiv:
                        if (y != 0) [[likely]] {
                            std::int64_t q = x / y;
                            std::int64_t rem = x % y;
                            if ((rem != 0) && ((rem < 0) != (y < 0))) --q;
                            write_reg_owned(cur->dst, Value::integer(q));
                            out = Value::integer(q); return true;
                        }
                        break;
                    case BinOpKind::BitAnd: write_reg_owned(cur->dst, Value::integer(x & y)); out = Value::integer(x & y); return true;
                    case BinOpKind::BitOr:  write_reg_owned(cur->dst, Value::integer(x | y)); out = Value::integer(x | y); return true;
                    case BinOpKind::BitXor: write_reg_owned(cur->dst, Value::integer(x ^ y)); out = Value::integer(x ^ y); return true;
                    case BinOpKind::LShift: if (y >= 0 && y < 63) { write_reg_owned(cur->dst, Value::integer(x << y)); out = Value::integer(x << y); return true; } break;
                    case BinOpKind::RShift: if (y >= 0 && y < 63) { write_reg_owned(cur->dst, Value::integer(x >> y)); out = Value::integer(x >> y); return true; } break;
                    default: break;
                }
            }
            // Float+float fast path
            if (a.tag() == Tag::Float && b.tag() == Tag::Float) {
                const double x = a.as_f(), y = b.as_f();
                switch (op) {
                    case BinOpKind::Add: result = Value::real(x + y); break;
                    case BinOpKind::Sub: result = Value::real(x - y); break;
                    case BinOpKind::Mul: result = Value::real(x * y); break;
                    case BinOpKind::TrueDiv:
                        if (y != 0.0) { result = Value::real(x / y); break; }
                        raise_builtin(rt.type_value_error, "float division by zero"); return false;
                    default: result = Value::none(); break;
                }
                if (result.tag() != Tag::None) {
                    write_reg_owned(cur->dst, result);
                    out = result; return true;
                }
            }
            // Generic path
            bool ok = false;
            switch (op) {
                case BinOpKind::Add: ok = values_add(a, b, result); break;
                case BinOpKind::Sub: ok = values_sub(a, b, result); break;
                case BinOpKind::Mul: ok = values_mul(a, b, result); break;
                case BinOpKind::TrueDiv: ok = values_truediv(a, b, result); break;
                case BinOpKind::FloorDiv: ok = values_floordiv(a, b, result); break;
                case BinOpKind::Mod: ok = values_mod(a, b, result); break;
                case BinOpKind::Pow: ok = values_pow(a, b, result); break;
                case BinOpKind::BitAnd: ok = values_bitop(a, b, 0, result); break;
                case BinOpKind::BitOr: ok = values_bitop(a, b, 1, result); break;
                case BinOpKind::BitXor: ok = values_bitop(a, b, 2, result); break;
                case BinOpKind::LShift: ok = values_shift(a, b, true, result); break;
                case BinOpKind::RShift: ok = values_shift(a, b, false, result); break;
                default: break;
            }
            if (!ok) {
                if (!has_pending()) raise_builtin(rt.type_type_error, "unsupported operand type");
                return false;
            }
            write_reg_owned(cur->dst, result);
            out = result; return true;
        }
        case Op::PY_UNOP: {
            Value a = regs[cur->a];
            Value result;
            bool ok = false;
            switch (cur->imm) {
                case 1: ok = values_neg(a, result); break;  // -
                case 2: {  // ~
                    std::int64_t x = 0;
                    if (as_i64(a, x)) {
                        result = Value::integer(~x);
                        ok = true;
                    }
                    break;
                }
                case 3: {  // not
                    result = Value::boolean(!rt.truthy(a));
                    ok = true;
                    break;
                }
                case 4: {  // bool
                    result = Value::boolean(rt.truthy(a));
                    ok = true;
                    break;
                }
                case 5: {  // truth chain
                    result = a;
                    ok = true;
                    break;
                }
            }
            if (!ok) {
                if (!has_pending()) raise_builtin(rt.type_type_error, "bad operand for unary op");
                return false;
            }
            write_reg_owned(cur->dst, result);
            out = result; return true;
        }
        case Op::PY_CMP: {
            Value a = regs[cur->a];
            Value b = regs[cur->b];
            if (a.tag() == Tag::Int && b.tag() == Tag::Int) {
                const std::int64_t x = a.as_i(), y = b.as_i();
                bool result = false;
                switch (static_cast<CmpOpKind>(cur->imm)) {
                    case CmpOpKind::LT: result = x <  y; break;
                    case CmpOpKind::LE: result = x <= y; break;
                    case CmpOpKind::GT: result = x >  y; break;
                    case CmpOpKind::GE: result = x >= y; break;
                    case CmpOpKind::EQ: result = x == y; break;
                    case CmpOpKind::NE: result = x != y; break;
                    default: break;
                }
                write_reg_owned(cur->dst, Value::boolean(result));
                out = Value::boolean(result); return true;
            }
            if (a.tag() == Tag::Float && b.tag() == Tag::Float) {
                const double x = a.as_f(), y = b.as_f();
                bool result = false;
                switch (static_cast<CmpOpKind>(cur->imm)) {
                    case CmpOpKind::LT: result = x <  y; break;
                    case CmpOpKind::LE: result = x <= y; break;
                    case CmpOpKind::GT: result = x >  y; break;
                    case CmpOpKind::GE: result = x >= y; break;
                    case CmpOpKind::EQ: result = x == y; break;
                    case CmpOpKind::NE: result = x != y; break;
                    default: break;
                }
                write_reg_owned(cur->dst, Value::boolean(result));
                out = Value::boolean(result); return true;
            }
            bool result = false;
            if (!values_compare(a, b, cur->imm, result)) {
                if (!has_pending()) raise_builtin(rt.type_type_error, "'...' not supported between instances");
                return false;
            }
            write_reg_owned(cur->dst, Value::boolean(result));
            out = Value::boolean(result); return true;
        }
        case Op::LOAD_GLOBAL: {
            Value v;
            if (!get_global(cur->imm, v)) return false;
            write_reg_owned(cur->dst, v);
            out = v; return true;
        }
        case Op::STORE_GLOBAL: {
            Value v = regs[cur->a];
            PyStrObj* key = rt.new_str(global_symbols().text(cur->imm));
            if (!dict_set(program.globals, Value::object(reinterpret_cast<PyObj*>(key)), v)) {
                rt.decref(reinterpret_cast<PyObj*>(key));
                return false;
            }
            rt.decref(reinterpret_cast<PyObj*>(key));
            out = Value::none(); return true;
        }
        case Op::LOAD_ATTR: {
            Value v;
            if (!get_attr(regs[cur->a], cur->imm, v)) return false;
            write_reg_owned(cur->dst, v);
            out = v; return true;
        }
        case Op::STORE_ATTR: {
            if (!set_attr(regs[cur->a], cur->imm, regs[cur->b])) return false;
            out = Value::none(); return true;
        }
        case Op::LOAD_INDEX: {
            // Inline the common cases: list[int], tuple[int], str[int], dict[key].
            // No Frame fallback — the bridge must return the SINGLE instruction's
            // result, not run the whole function.
            Value obj = regs[cur->a];
            Value idx = regs[cur->b];
            Value result;
            bool ok = false;
            if (obj.tag() == Tag::Obj && obj.as_obj()) {
                std::int64_t i = 0;
                if (obj.as_obj()->tag == ObjTag::List && as_i64(idx, i)) {
                    auto* l = static_cast<PyListObj*>(obj.as_obj());
                    std::int64_t len = static_cast<std::int64_t>(l->length);
                    std::int64_t eff = i < 0 ? i + len : i;
                    if (eff >= 0 && eff < len) {
                        result = l->items[static_cast<std::uint32_t>(eff)];
                        ok = true;
                    } else {
                        raise_builtin(rt.type_index_error, "list index out of range");
                        return false;
                    }
                } else if (obj.as_obj()->tag == ObjTag::Tuple && as_i64(idx, i)) {
                    auto* t = static_cast<PyTupleObj*>(obj.as_obj());
                    std::int64_t len = static_cast<std::int64_t>(t->length);
                    std::int64_t eff = i < 0 ? i + len : i;
                    if (eff >= 0 && eff < len) {
                        result = t->items[static_cast<std::uint32_t>(eff)];
                        ok = true;
                    } else {
                        raise_builtin(rt.type_index_error, "tuple index out of range");
                        return false;
                    }
                } else if (obj.as_obj()->tag == ObjTag::Str && as_i64(idx, i)) {
                    auto* s = static_cast<PyStrObj*>(obj.as_obj());
                    std::int64_t len = static_cast<std::int64_t>(s->length);
                    std::int64_t eff = i < 0 ? i + len : i;
                    if (eff >= 0 && eff < len) {
                        result = Value::object(reinterpret_cast<PyObj*>(
                            rt.new_str(std::string_view(s->data() + eff, 1))));
                        ok = true;
                    } else {
                        raise_builtin(rt.type_index_error, "string index out of range");
                        return false;
                    }
                } else if (obj.as_obj()->tag == ObjTag::Dict) {
                    if (dict_get(static_cast<PyDictObj*>(obj.as_obj()), idx, result)) {
                        ok = true;
                    } else {
                        raise_builtin(rt.type_key_error, "key not found");
                        return false;
                    }
                }
            }
            if (!ok) {
                if (!has_pending()) raise_builtin(rt.type_type_error, "object is not subscriptable");
                return false;
            }
            write_reg_owned(cur->dst, result);
            out = result; return true;
        }
        case Op::STORE_INDEX: {
            Value obj = regs[cur->a];
            Value idx = regs[cur->b];
            Value val = regs[cur->c];
            bool ok = false;
            if (obj.tag() == Tag::Obj && obj.as_obj()) {
                std::int64_t i = 0;
                if (obj.as_obj()->tag == ObjTag::List && as_i64(idx, i)) {
                    auto* l = static_cast<PyListObj*>(obj.as_obj());
                    std::int64_t len = static_cast<std::int64_t>(l->length);
                    std::int64_t eff = i < 0 ? i + len : i;
                    if (eff >= 0 && eff < len) {
                        ok = list_set(l, static_cast<std::uint32_t>(eff), val);
                    } else {
                        raise_builtin(rt.type_index_error, "list assignment index out of range");
                        return false;
                    }
                } else if (obj.as_obj()->tag == ObjTag::Dict) {
                    ok = dict_set(static_cast<PyDictObj*>(obj.as_obj()), idx, val);
                }
            }
            if (!ok) {
                if (!has_pending()) raise_builtin(rt.type_type_error, "object does not support item assignment");
                return false;
            }
            out = Value::none(); return true;
        }
        case Op::NEW_LIST: {
            auto* l = rt.new_list(cur->b > 0 ? cur->b : 4);
            for (std::uint32_t i = 0; i < cur->b; ++i) {
                list_push(l, regs[cur->a + i]);
            }
            Value v = Value::object(reinterpret_cast<PyObj*>(l));
            write_reg_owned(cur->dst, v);
            out = v; return true;
        }
        case Op::NEW_TUPLE: {
            auto* t = rt.new_tuple(cur->b);
            for (std::uint32_t i = 0; i < cur->b; ++i) {
                t->items[i] = regs[cur->a + i];
                if (regs[cur->a + i].tag() == Tag::Obj) rt.incref(regs[cur->a + i].as_obj());
            }
            Value v = Value::object(reinterpret_cast<PyObj*>(t));
            write_reg_owned(cur->dst, v);
            out = v; return true;
        }
        case Op::NEW_DICT: {
            auto* d = rt.new_dict();
            Value v = Value::object(reinterpret_cast<PyObj*>(d));
            write_reg_owned(cur->dst, v);
            out = v; return true;
        }
        case Op::LIST_APPEND: {
            auto* l = static_cast<PyListObj*>(regs[cur->a].as_obj());
            if (!l) { raise_builtin(rt.type_type_error, "append on non-list"); return false; }
            if (!list_push(l, regs[cur->b])) {
                raise_builtin(rt.type_memory_error, "list allocation failed");
                return false;
            }
            out = Value::none(); return true;
        }
        case Op::CALL: {
            Value callee = regs[cur->a];
            Value* args = cur->c > 0 ? &regs[cur->b] : nullptr;
            bool prev = jit_disabled_in_bridge;
            jit_disabled_in_bridge = true;
            bool ok = call_value(callee, args, cur->c, out);
            jit_disabled_in_bridge = prev;
            if (!ok) return false;
            // call_value wrote to `out` (its out parameter). Write it to
            // the register file and incref (the JIT's register now owns
            // a ref, and the bridge caller also gets a ref in `out`).
            write_reg_owned(cur->dst, out);
            return true;
        }
        case Op::CALL_KW: {
            Value callee = regs[cur->a];
            Value* args = cur->c > 0 ? &regs[cur->b] : nullptr;
            std::uint32_t kwnode = cur->imm >> 16;
            PyTupleObj* kw_names = kwnode < n_regs && regs[kwnode].tag() == Tag::Obj &&
                                   regs[kwnode].as_obj() &&
                                   regs[kwnode].as_obj()->tag == ObjTag::Tuple
                               ? static_cast<PyTupleObj*>(regs[kwnode].as_obj())
                               : nullptr;
            std::uint32_t nkw = kw_names ? kw_names->length : 0;
            bool prev = jit_disabled_in_bridge;
            jit_disabled_in_bridge = true;
            bool ok = call_value_kw(callee, args, cur->c, kw_names, nkw, out);
            jit_disabled_in_bridge = prev;
            if (!ok) return false;
            write_reg_owned(cur->dst, out);
            return true;
        }
        case Op::NATIVE: {
            Value* args = cur->b > 0 ? &regs[cur->a] : nullptr;
            if (!native_helper(cur->imm, args, cur->b, out)) return false;
            write_reg_owned(cur->dst, out);
            return true;
        }
        case Op::ITER: {
            if (!get_iter(regs[cur->a], out)) return false;
            write_reg_owned(cur->dst, out);
            return true;
        }
        case Op::ITER_CHECK: {
            Value it = regs[cur->a];
            bool more = false;
            if (!iter_check(it, more)) return false;
            write_reg_owned(cur->dst, Value::boolean(more));
            out = Value::boolean(more);
            return true;
        }
        case Op::ITER_NEXT: {
            if (!iter_next(regs[cur->a], out)) return false;
            write_reg_owned(cur->dst, out);
            return true;
        }
        case Op::LOAD_FIELD: {
            Value v;
            // Delegate to the L_LOAD_FIELD logic via a temp Frame.
            // This is rare enough (only Pass 46 specialized dicts) that
            // the Frame overhead is acceptable.
            Frame f(unit);
            for (std::uint32_t i = 0; i < n_regs && i < f.n_regs; ++i) {
                f.regs[i] = regs[i];
                if (regs[i].tag() == Tag::Obj && regs[i].as_obj()) rt.incref(regs[i].as_obj());
            }
            f.pc = pc;
            bool prev = jit_disabled_in_bridge;
            jit_disabled_in_bridge = true;
            ExecStatus st = exec_frame(f);
            jit_disabled_in_bridge = prev;
            for (std::uint32_t i = 0; i < n_regs && i < f.n_regs; ++i) {
                regs[i] = f.regs[i];
                if (f.regs[i].tag() == Tag::Obj && f.regs[i].as_obj()) rt.incref(f.regs[i].as_obj());
            }
            if (st == ExecStatus::Returned) {
                out = frame_return_;
                frame_return_ = Value::none();
                return true;
            }
            return false;
        }
        case Op::STORE_FIELD: {
            // Same Frame fallback as LOAD_FIELD.
            Frame f(unit);
            for (std::uint32_t i = 0; i < n_regs && i < f.n_regs; ++i) {
                f.regs[i] = regs[i];
                if (regs[i].tag() == Tag::Obj && regs[i].as_obj()) rt.incref(regs[i].as_obj());
            }
            f.pc = pc;
            bool prev = jit_disabled_in_bridge;
            jit_disabled_in_bridge = true;
            ExecStatus st = exec_frame(f);
            jit_disabled_in_bridge = prev;
            for (std::uint32_t i = 0; i < n_regs && i < f.n_regs; ++i) {
                regs[i] = f.regs[i];
                if (f.regs[i].tag() == Tag::Obj && f.regs[i].as_obj()) rt.incref(f.regs[i].as_obj());
            }
            if (st == ExecStatus::Returned) {
                out = frame_return_;
                frame_return_ = Value::none();
                return true;
            }
            return false;
        }
        default:
            // Control-flow ops (JUMP, RETURN, RAISE, TRY_*, GET_EXC, YIELD)
            // are handled natively by the JIT — the bridge should never see
            // them. If we get here, it's a bug: either the JIT emitted a
            // CALLri for a control-flow op (code generation bug), or the
            // node_id_to_pc map pointed to the wrong instruction.
            //
            // Rule 58: No silent fallbacks. This path must not return false
            // silently (the CALL handler would treat it as "function returned
            // None"). Instead, raise a RuntimeError so the caller sees the
            // failure and can report it.
            raise_builtin(rt.type_runtime_error,
                          "step_one: unexpected control-flow op in bridge");
            return false;
    }
}

bool Vm::enter_at(CodeUnit* unit, Value* regs, std::uint32_t n_regs, std::uint32_t pc,
                  Value& out) noexcept {
    Frame f(unit);
    // Transfer owned refs into the frame's register file. INCREF all
    // copied object values — the Frame owns its own references, separate
    // from the caller's (the JIT's) register file. Without this, both
    // the Frame and the JIT hold the same pointers; when the Frame's
    // destructor decrefs them, the JIT's references become dangling
    // (double-free when the JIT's Frame is later destroyed).
    Runtime& rt = Runtime::instance();
    for (std::uint32_t i = 0; i < n_regs && i < f.n_regs; ++i) {
        f.regs[i] = regs[i];
        if (regs[i].tag() == Tag::Obj && regs[i].as_obj()) rt.incref(regs[i].as_obj());
    }
    
    ExecStatus st = exec_frame(f);
    if (st == ExecStatus::Returned) {
        out = frame_return_;
        frame_return_ = Value::none();
        return true;
    }
    return false;
}

// =============================================================================
// Rule 88: Safepoint polling infrastructure.
//
// Currently single-threaded: g_safepoint_requested is a plain bool, and
// request_safepoint() immediately calls vortex_safepoint_poll() which clears
// it. When multi-threading is added, this becomes atomic with relaxed loads
// in the JIT poll and a release store when requesting.
// =============================================================================

bool g_safepoint_requested{false};

void request_safepoint() noexcept {
    g_safepoint_requested = true;
    // Single-threaded: immediately process the safepoint.
    vortex_safepoint_poll();
}

void vortex_safepoint_poll() noexcept {
    // Single-threaded: just clear the flag. When multi-threaded, this
    // will block until the runtime releases the thread.
    g_safepoint_requested = false;
}

}  // namespace abi_v1
}  // namespace vortex::rt
