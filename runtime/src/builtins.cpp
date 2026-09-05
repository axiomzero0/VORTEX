// =============================================================================
// vortex/rt/builtins.cpp — builtin functions & native modules.
// =============================================================================

#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "vortex/ir/node.hpp"
#include "vortex/rt/interp.hpp"
#include "vortex/support/symbol_table.hpp"

namespace vortex::rt {
inline namespace abi_v1 {

namespace {

using NativeFn = Value (*)(void* user, Value* args, std::uint32_t argc);

Vm* g_vm = nullptr;   // set by install_builtins; single-VM programs (subset)

[[nodiscard]] Value ret_none() noexcept { return Value::none(); }
[[nodiscard]] Value error_msg(const char* msg) noexcept {
    if (g_vm) g_vm->raise_builtin(Runtime::instance().type_type_error, msg);
    return Value::object(nullptr);   // error sentinel
}

// --- print --------------------------------------------------------------------
Value bi_print(void*, Value* args, std::uint32_t argc) noexcept {
    Runtime& rt = Runtime::instance();
    for (std::uint32_t i = 0; i < argc; ++i) {
        if (i) std::fputc(' ', stdout);
        stdx::small_vector<char, 128> buf;
        rt.str_into(args[i], buf);
        std::fwrite(buf.data(), 1, buf.size(), stdout);
    }
    std::fputc('\n', stdout);
    return ret_none();
}

// --- len ----------------------------------------------------------------------
Value bi_len(void*, Value* args, std::uint32_t argc) noexcept {
    if (argc != 1) return error_msg("len() takes exactly one argument");
    const Value& v = args[0];
    if (v.tag == Tag::Obj && v.as.obj) {
        switch (v.as.obj->tag) {
            case ObjTag::Str: return Value::integer(static_cast<PyStrObj*>(v.as.obj)->length);
            case ObjTag::List: return Value::integer(static_cast<PyListObj*>(v.as.obj)->length);
            case ObjTag::Tuple: return Value::integer(static_cast<PyTupleObj*>(v.as.obj)->length);
            case ObjTag::Dict: return Value::integer(static_cast<PyDictObj*>(v.as.obj)->count);
            default: break;
        }
    }
    return error_msg("object has no len()");
}

// --- range --------------------------------------------------------------------
Value bi_range(void*, Value* args, std::uint32_t argc) noexcept {
    if (argc == 0 || argc > 3) return error_msg("range() takes 1-3 arguments");
    std::int64_t a = 0, b = 0, step = 1;
    if (argc == 1) {
        if (!as_i64(args[0], b)) return error_msg("range() arguments must be integers");
    } else {
        if (!as_i64(args[0], a) || !as_i64(args[1], b)) {
            return error_msg("range() arguments must be integers");
        }
        if (argc == 3 && !as_i64(args[2], step)) {
            return error_msg("range() arguments must be integers");
        }
        if (argc == 3 && step == 0) return error_msg("range() step must not be zero");
    }
    auto* it = Runtime::instance().new_range_iter(a, b, step);
    return Value::object(reinterpret_cast<PyObj*>(it));
}

// --- abs / min / max / sum -------------------------------------------------------
Value bi_abs(void*, Value* args, std::uint32_t argc) noexcept {
    if (argc != 1) return error_msg("abs() takes exactly one argument");
    Value out;
    if (args[0].tag == Tag::Int) {
        std::int64_t x = args[0].as.i;
        if (x == INT64_MIN) {
            if (values_neg(args[0], out)) return out;
        } else {
            return Value::integer(x < 0 ? -x : x);
        }
    }
    double d = 0;
    if (as_f64(args[0], d)) return Value::real(std::fabs(d));
    return error_msg("bad operand for abs()");
}

// Flatten a single iterable argument into candidate values (Python:
// max(iterable) and max(a, b, c) are both valid).
// OBJ-23 fix: the previous implementation truncated to 32 entries
// (out_cap was 32 in bi_min/bi_max), so max/min of a list with >32
// items silently returned the wrong element. We now grow the out
// buffer dynamically — flatten_candidates returns the true count.
[[nodiscard]] std::uint32_t flatten_candidates(Value* args, std::uint32_t argc,
                                               stdx::small_vector<Value, 32>& out) noexcept {
    if (argc != 1) {
        for (std::uint32_t i = 0; i < argc; ++i) out.push_back(args[i]);
        return argc;
    }
    if (args[0].tag == Tag::Obj && args[0].as.obj) {
        PyObj* o = args[0].as.obj;
        if (o->tag == ObjTag::List) {
            auto* l = static_cast<PyListObj*>(o);
            for (std::uint32_t i = 0; i < l->length; ++i) out.push_back(l->items[i]);
            return l->length;
        }
        if (o->tag == ObjTag::Tuple) {
            auto* t = static_cast<PyTupleObj*>(o);
            for (std::uint32_t i = 0; i < t->length; ++i) out.push_back(t->items[i]);
            return t->length;
        }
    }
    out.push_back(args[0]);
    return 1;
}

Value bi_min(void*, Value* args, std::uint32_t argc) noexcept {
    if (argc == 0) return error_msg("min() needs at least one argument");
    stdx::small_vector<Value, 32> flat;
    std::uint32_t n = flatten_candidates(args, argc, flat);
    if (n == 0) return error_msg("min() arg is an empty sequence");
    Value best = flat[0];
    for (std::uint32_t i = 1; i < n; ++i) {
        bool lt = false;
        if (!values_compare(flat[i], best, static_cast<std::uint16_t>(vortex::ir::CmpOpKind::LT), lt)) {
            return error_msg("min() arguments not comparable");
        }
        if (lt) best = flat[i];
    }
    // OBJ-15 fix: best is borrowed from flat[] (a stack copy of args).
    // Callers treat the return as owned; incref to avoid a UAF when the
    // caller's slot is released.
    Runtime& rt = Runtime::instance();
    if (best.tag == Tag::Obj && best.as.obj) rt.incref(best.as.obj);
    return best;
}

Value bi_max(void*, Value* args, std::uint32_t argc) noexcept {
    if (argc == 0) return error_msg("max() needs at least one argument");
    stdx::small_vector<Value, 32> flat;
    std::uint32_t n = flatten_candidates(args, argc, flat);
    if (n == 0) return error_msg("max() arg is an empty sequence");
    Value best = flat[0];
    for (std::uint32_t i = 1; i < n; ++i) {
        bool gt = false;
        if (!values_compare(flat[i], best, static_cast<std::uint16_t>(vortex::ir::CmpOpKind::GT), gt)) {
            return error_msg("max() arguments not comparable");
        }
        if (gt) best = flat[i];
    }
    // OBJ-15 fix: incref before returning (see bi_min).
    Runtime& rt = Runtime::instance();
    if (best.tag == Tag::Obj && best.as.obj) rt.incref(best.as.obj);
    return best;
}

Value bi_sum(void*, Value* args, std::uint32_t argc) noexcept {
    if (argc < 1 || argc > 2) return error_msg("sum() takes 1-2 arguments");
    Value acc = argc == 2 ? args[1] : Value::integer(0);
    if (args[0].tag == Tag::Obj && args[0].as.obj) {
        PyObj* o = args[0].as.obj;
        Value* items = nullptr;
        std::uint32_t n = 0;
        if (o->tag == ObjTag::List) {
            items = static_cast<PyListObj*>(o)->items;
            n = static_cast<PyListObj*>(o)->length;
        } else if (o->tag == ObjTag::Tuple) {
            items = static_cast<PyTupleObj*>(o)->items;
            n = static_cast<PyTupleObj*>(o)->length;
        }
        if (items) {
            for (std::uint32_t i = 0; i < n; ++i) {
                if (!values_add(acc, items[i], acc)) {
                    return error_msg("sum(): unsupported operand types");
                }
            }
            return acc;
        }
    }
    return error_msg("sum() expects an iterable");
}

// --- constructors -------------------------------------------------------------------
Value bi_str(void*, Value* args, std::uint32_t argc) noexcept {
    Runtime& rt = Runtime::instance();
    if (argc == 0) return Value::object(reinterpret_cast<PyObj*>(rt.new_str("")));
    stdx::small_vector<char, 128> buf;
    rt.str_into(args[0], buf);
    return Value::object(reinterpret_cast<PyObj*>(
        rt.new_str(std::string_view(buf.data(), buf.size()))));
}

Value bi_repr(void*, Value* args, std::uint32_t argc) noexcept {
    if (argc != 1) return error_msg("repr() takes exactly one argument");
    Runtime& rt = Runtime::instance();
    stdx::small_vector<char, 128> buf;
    rt.repr_into(args[0], buf);
    return Value::object(reinterpret_cast<PyObj*>(
        rt.new_str(std::string_view(buf.data(), buf.size()))));
}

Value bi_int(void*, Value* args, std::uint32_t argc) noexcept {
    if (argc == 0) return Value::integer(0);
    std::int64_t i = 0;
    if (as_i64(args[0], i)) return Value::integer(i);
    double d = 0;
    if (as_f64(args[0], d)) return Value::integer(static_cast<std::int64_t>(d));
    if (args[0].tag == Tag::Obj && args[0].as.obj &&
        args[0].as.obj->tag == ObjTag::Str) {
        auto* s = static_cast<PyStrObj*>(args[0].as.obj);
        // OBJ-19 fix: the previous code copied into a fixed 64-byte buffer
        // and used strtoll on the truncated copy. A 200-char string like
        // "999...9" would silently parse to the wrong int64 (truncated).
        // Use std::string to hold the whole input and pass the full text
        // to strtoll; report overflow as int max/min like CPython does.
        std::string text(s->data(), s->length);
        // strip leading/trailing whitespace (Python allows it)
        std::size_t a = text.find_first_not_of(" \t\n\r");
        if (a == std::string::npos) return error_msg("int() cannot convert value");
        std::size_t b = text.find_last_not_of(" \t\n\r");
        std::string trimmed = text.substr(a, b - a + 1);
        // try base 10 (Python int(s) defaults to base 10; int(s, 0) tries
        // 0x/0o/0b prefixes — we leave that to a separate builtin)
        errno = 0;
        char* end = nullptr;
        long long v = std::strtoll(trimmed.c_str(), &end, 10);
        if (end == trimmed.c_str()) return error_msg("int() cannot convert value");
        if (errno == ERANGE) {
            // Overflow: return int max or min like CPython's int(s) does
            // for huge strings — but CPython actually returns a bignum.
            // We return max/min as a documented subset limitation.
            return v > 0 ? Value::integer(INT64_MAX) : Value::integer(INT64_MIN);
        }
        // trailing chars after the number? Python rejects them
        while (*end) {
            if (!std::isspace(static_cast<unsigned char>(*end))) {
                return error_msg("int() invalid literal");
            }
            ++end;
        }
        return Value::integer(v);
    }
    if (args[0].tag == Tag::Obj && args[0].as.obj &&
        args[0].as.obj->tag == ObjTag::Bool) {
        return Value::integer(static_cast<PyBoolObj*>(args[0].as.obj)->value ? 1 : 0);
    }
    return error_msg("int() cannot convert value");
}

Value bi_float(void*, Value* args, std::uint32_t argc) noexcept {
    if (argc == 0) return Value::real(0.0);
    double d = 0;
    if (as_f64(args[0], d)) return Value::real(d);
    // OBJ-20 fix: float("3.14") was rejected because as_f64 didn't accept
    // a Str argument. Now we parse the string directly.
    if (args[0].tag == Tag::Obj && args[0].as.obj &&
        args[0].as.obj->tag == ObjTag::Str) {
        auto* s = static_cast<PyStrObj*>(args[0].as.obj);
        std::string text(s->data(), s->length);
        errno = 0;
        char* end = nullptr;
        double v = std::strtod(text.c_str(), &end);
        if (end != text.c_str() && *end == '\0') return Value::real(v);
        // allow trailing whitespace
        while (*end && std::isspace(static_cast<unsigned char>(*end))) ++end;
        if (end != text.c_str() && *end == '\0') return Value::real(v);
        return error_msg("float() cannot convert string");
    }
    return error_msg("float() cannot convert value");
}

Value bi_bool(void*, Value* args, std::uint32_t argc) noexcept {
    if (argc == 0) return Value::boolean(false);
    return Value::boolean(Runtime::instance().truthy(args[0]));
}

Value bi_list(void*, Value* args, std::uint32_t argc) noexcept {
    Runtime& rt = Runtime::instance();
    auto* l = rt.new_list();
    if (argc == 1 && args[0].tag == Tag::Obj && args[0].as.obj) {
        PyObj* o = args[0].as.obj;
        if (o->tag == ObjTag::List) {
            auto* src = static_cast<PyListObj*>(o);
            for (std::uint32_t i = 0; i < src->length; ++i) list_push(l, src->items[i]);
            return Value::object(reinterpret_cast<PyObj*>(l));
        }
        if (o->tag == ObjTag::Tuple) {
            auto* src = static_cast<PyTupleObj*>(o);
            for (std::uint32_t i = 0; i < src->length; ++i) list_push(l, src->items[i]);
            return Value::object(reinterpret_cast<PyObj*>(l));
        }
        if (o->tag == ObjTag::Str) {
            auto* s = static_cast<PyStrObj*>(o);
            for (std::uint32_t i = 0; i < s->length; ++i) {
                auto* ch = rt.new_str(std::string_view(s->data() + i, 1));
                list_push(l, Value::object(reinterpret_cast<PyObj*>(ch)));
            }
            return Value::object(reinterpret_cast<PyObj*>(l));
        }
        if (o->tag == ObjTag::Dict) {
            // OBJ-17 fix: bi_list(dict) used to include tombstone slots
            // (used=true, key=None). Now we skip them by checking the key
            // tag (None is never a valid dict key in the subset, so any
            // used slot with key.tag=None is a tombstone). With backward-
            // shift deletion in dict_del, there shouldn't be tombstones
            // in the live table at all; this guard is defensive.
            auto* d = static_cast<PyDictObj*>(o);
            for (std::uint32_t i = 0; i < d->capacity; ++i) {
                if (d->entries[i].used && d->entries[i].key.tag != Tag::None) {
                    list_push(l, d->entries[i].key);
                }
            }
            return Value::object(reinterpret_cast<PyObj*>(l));
        }
    }
    return Value::object(reinterpret_cast<PyObj*>(l));
}

Value bi_tuple(void*, Value* args, std::uint32_t argc) noexcept {
    Value lv = bi_list(nullptr, args, argc);
    if (lv.tag == Tag::Obj && lv.as.obj == nullptr) return lv;
    auto* l = static_cast<PyListObj*>(lv.as.obj);
    auto* t = Runtime::instance().new_tuple(l->length);
    for (std::uint32_t i = 0; i < l->length; ++i) {
        t->items[i] = l->items[i];
        if (l->items[i].tag == Tag::Obj) Runtime::instance().incref(l->items[i].as.obj);
    }
    Runtime::instance().decref(reinterpret_cast<PyObj*>(l));
    return Value::object(reinterpret_cast<PyObj*>(t));
}

Value bi_dict(void*, Value* args, std::uint32_t argc) noexcept {
    (void)args;
    (void)argc;
    return Value::object(reinterpret_cast<PyObj*>(Runtime::instance().new_dict()));
}

// --- enumerate / zip / map / filter / sorted ---------------------------------------
Value bi_enumerate(void*, Value* args, std::uint32_t argc) noexcept {
    Runtime& rt = Runtime::instance();
    if (argc == 0) return error_msg("enumerate() needs an iterable");
    auto* l = rt.new_list();
    // OBJ-24 fix: enumerate used to silently return an empty list for any
    // non-list iterable (tuples, strings, ranges). Now we walk the iterable
    // via a small flatten step. (Full iterator-protocol enumerate is left
    // for a follow-up; this covers the common cases.)
    stdx::small_vector<Value, 32> flat;
    if (args[0].tag == Tag::Obj && args[0].as.obj) {
        PyObj* src_obj = args[0].as.obj;
        if (src_obj->tag == ObjTag::List) {
            auto* src = static_cast<PyListObj*>(src_obj);
            for (std::uint32_t i = 0; i < src->length; ++i) flat.push_back(src->items[i]);
        } else if (src_obj->tag == ObjTag::Tuple) {
            auto* src = static_cast<PyTupleObj*>(src_obj);
            for (std::uint32_t i = 0; i < src->length; ++i) flat.push_back(src->items[i]);
        } else if (src_obj->tag == ObjTag::Str) {
            auto* src = static_cast<PyStrObj*>(src_obj);
            for (std::uint32_t i = 0; i < src->length; ++i) {
                auto* ch = rt.new_str(std::string_view(src->data() + i, 1));
                flat.push_back(Value::object(reinterpret_cast<PyObj*>(ch)));
            }
        }
    }
    std::int64_t start = 0;
    if (argc > 1) as_i64(args[1], start);
    for (std::uint32_t i = 0; i < flat.size(); ++i) {
        auto* pair = rt.new_tuple(2);
        pair->items[0] = Value::integer(start + i);
        pair->items[1] = flat[i];
        if (flat[i].tag == Tag::Obj && flat[i].as.obj) rt.incref(flat[i].as.obj);
        // OBJ-8 fix: list_push incref's the pair (adopts). The pair was
        // created with new_tuple(refcount=1, owned by us); after list_push
        // its refcount is 2 (list+us). Drop our reference so the list is
        // the sole owner. Previously we forgot to decref, leaking +1 per
        // enumerate entry.
        Value pair_val = Value::object(reinterpret_cast<PyObj*>(pair));
        list_push(l, pair_val);
        rt.decref(reinterpret_cast<PyObj*>(pair));
    }
    return Value::object(reinterpret_cast<PyObj*>(l));
}

Value bi_zip(void*, Value* args, std::uint32_t argc) noexcept {
    Runtime& rt = Runtime::instance();
    auto* out = rt.new_list();
    if (argc == 0) return Value::object(reinterpret_cast<PyObj*>(out));
    // OBJ-24 fix: support any combination of list/tuple/str iterables.
    // We materialize each into a flat small_vector, then zip by min_len.
    stdx::small_vector<stdx::small_vector<Value, 16>, 4> iterables;
    iterables.resize(argc);
    std::uint32_t min_len = 0xFFFFFFFFu;
    for (std::uint32_t a = 0; a < argc; ++a) {
        if (args[a].tag == Tag::Obj && args[a].as.obj) {
            PyObj* o = args[a].as.obj;
            if (o->tag == ObjTag::List) {
                auto* src = static_cast<PyListObj*>(o);
                for (std::uint32_t i = 0; i < src->length; ++i) iterables[a].push_back(src->items[i]);
            } else if (o->tag == ObjTag::Tuple) {
                auto* src = static_cast<PyTupleObj*>(o);
                for (std::uint32_t i = 0; i < src->length; ++i) iterables[a].push_back(src->items[i]);
            } else if (o->tag == ObjTag::Str) {
                auto* src = static_cast<PyStrObj*>(o);
                for (std::uint32_t i = 0; i < src->length; ++i) {
                    auto* ch = rt.new_str(std::string_view(src->data() + i, 1));
                    iterables[a].push_back(Value::object(reinterpret_cast<PyObj*>(ch)));
                }
            } else {
                min_len = 0;
            }
            if (iterables[a].size() < min_len) min_len = static_cast<std::uint32_t>(iterables[a].size());
        } else {
            min_len = 0;
        }
    }
    if (min_len == 0xFFFFFFFFu) min_len = 0;
    for (std::uint32_t i = 0; i < min_len; ++i) {
        auto* tup = rt.new_tuple(argc);
        for (std::uint32_t a = 0; a < argc; ++a) {
            Value v = iterables[a][i];
            tup->items[a] = v;
            if (v.tag == Tag::Obj) rt.incref(v.as.obj);
        }
        // OBJ-8 fix: list_push incref's tup; we own a separate ref from
        // new_tuple. Drop ours to avoid the per-zip-iteration leak.
        Value tup_val = Value::object(reinterpret_cast<PyObj*>(tup));
        list_push(out, tup_val);
        rt.decref(reinterpret_cast<PyObj*>(tup));
    }
    return Value::object(reinterpret_cast<PyObj*>(out));
}

// map/filter need call-backs into the VM — they live on Vm, not here.

Value bi_sorted(void*, Value* args, std::uint32_t argc) noexcept {
    // OBJ-24 fix: accept list/tuple/str (not just list).
    if (argc != 1 || args[0].tag != Tag::Obj || !args[0].as.obj) {
        return error_msg("sorted() expects an iterable");
    }
    PyObj* src_obj = args[0].as.obj;
    if (src_obj->tag != ObjTag::List && src_obj->tag != ObjTag::Tuple &&
        src_obj->tag != ObjTag::Str) {
        return error_msg("sorted() expects a list/tuple/str (subset)");
    }
    Value lv = bi_list(nullptr, args, 1);
    if (lv.tag == Tag::Obj && lv.as.obj == nullptr) return lv;
    auto* l = static_cast<PyListObj*>(lv.as.obj);
    // insertion sort (stable) — lists in scope are small
    const std::uint16_t lt = static_cast<std::uint16_t>(vortex::ir::CmpOpKind::LT);
    for (std::uint32_t i = 1; i < l->length; ++i) {
        Value key = l->items[i];
        std::int32_t j = static_cast<std::int32_t>(i) - 1;
        while (j >= 0) {
            bool less = false;
            if (!values_compare(key, l->items[j], lt, less)) {
                return error_msg("sorted(): elements not comparable");
            }
            if (!less) break;
            l->items[j + 1] = l->items[j];
            --j;
        }
        l->items[j + 1] = key;
    }
    return lv;
}

Value bi_next(void*, Value* args, std::uint32_t argc) noexcept {
    if (argc < 1) return error_msg("next() needs an iterator");
    if (g_vm) {
        Value out;
        // route through the VM's iterator protocol (raises StopIteration)
        bool more = false;
        if (!g_vm->iter_check(args[0], more)) return Value::object(nullptr);
        if (!more) {
            g_vm->raise_builtin(Runtime::instance().type_stop_iter, "");
            return Value::object(nullptr);
        }
        if (!g_vm->iter_next(args[0], out)) return Value::object(nullptr);
        return out;
    }
    return error_msg("next() without vm context");
}

Value bi_iter(void*, Value* args, std::uint32_t argc) noexcept {
    if (argc != 1) return error_msg("iter() takes exactly one argument");
    // iter(x) on a generator/x returns x itself in the subset.
    if (args[0].tag == Tag::Obj && args[0].as.obj) {
        switch (args[0].as.obj->tag) {
            case ObjTag::Generator: case ObjTag::RangeIter:
            case ObjTag::ListIter: case ObjTag::StrIter: case ObjTag::DictIter:
                // OBJ-15 fix: return an OWNED reference (incref) so the
                // caller's release balances. Previously we returned the
                // borrowed arg[0], and the caller's decref would drop
                // the iterator prematurely.
                Runtime::instance().incref(args[0].as.obj);
                return args[0];
            default: break;
        }
    }
    return error_msg("iter() argument is not iterable");
}

Value bi_isinstance(void*, Value* args, std::uint32_t argc) noexcept {
    if (argc != 2) return error_msg("isinstance() takes two arguments");
    Runtime& rt = Runtime::instance();
    if (args[1].tag == Tag::Obj && args[1].as.obj &&
        args[1].as.obj->tag == ObjTag::Type) {
        auto* t = static_cast<PyTypeObj*>(args[1].as.obj);
        if (args[0].tag == Tag::Obj && args[0].as.obj &&
            args[0].as.obj->tag == ObjTag::Instance) {
            for (PyTypeObj* tt = static_cast<PyInstanceObj*>(args[0].as.obj)->type; tt;
                 tt = tt->base) {
                if (tt == t) return Value::boolean(true);
            }
            return Value::boolean(false);
        }
        return Value::boolean(rt.type_of(args[0]) == t);
    }
    return error_msg("isinstance() second argument must be a class");
}

Value bi_type(void*, Value* args, std::uint32_t argc) noexcept {
    if (argc != 1) return error_msg("type() takes one argument");
    // OBJ-9 fix: bi_type used to return the repr string of the value
    // (e.g. type(42) returned "42" instead of <class 'int'>). Now we
    // return the actual type object, which the runtime already knows
    // how to build via Runtime::type_of. Callers can then access
    // type.__name__ or compare with isinstance.
    Runtime& rt = Runtime::instance();
    PyTypeObj* t = rt.type_of(args[0]);
    if (!t) return error_msg("type() cannot determine type");
    rt.incref(reinterpret_cast<PyObj*>(t));
    return Value::object(reinterpret_cast<PyObj*>(t));
}

// --- round --------------------------------------------------------------------
Value bi_round(void*, Value* args, std::uint32_t argc) noexcept {
    if (argc == 0) return error_msg("round() needs at least one argument");
    double x = 0;
    if (!as_f64(args[0], x)) return error_msg("round() needs a number");
    if (argc == 1) {
        // Python round() without ndigits returns an int for float input.
        // Uses banker's rounding (round half to even).
        double r = std::round(x);
        // Adjust for .5 cases: round() rounds to even, std::round rounds away
        if (std::fabs(x - std::floor(x) - 0.5) < 1e-15) {
            std::int64_t floor_val = static_cast<std::int64_t>(std::floor(x));
            if (floor_val % 2 != 0) r = floor_val;  // round to even
        }
        return Value::integer(static_cast<std::int64_t>(r));
    }
    std::int64_t ndigits = 0;
    if (!as_i64(args[1], ndigits)) return error_msg("round() ndigits must be int");
    double scale = std::pow(10.0, static_cast<double>(ndigits));
    double r = std::round(x * scale) / scale;
    return Value::real(r);
}

// --- divmod -------------------------------------------------------------------
Value bi_divmod(void*, Value* args, std::uint32_t argc) noexcept {
    if (argc != 2) return error_msg("divmod() takes two arguments");
    std::int64_t a = 0, b = 0;
    if (as_i64(args[0], a) && as_i64(args[1], b)) {
        if (b == 0) return error_msg("divmod() by zero");
        std::int64_t q = a / b;
        std::int64_t r = a % b;
        // Python floor division: adjust for negative
        if (r != 0 && ((r < 0) != (b < 0))) { --q; r += b; }
        Runtime& rt = Runtime::instance();
        PyTupleObj* t = rt.new_tuple(2);
        t->items[0] = Value::integer(q);
        t->items[1] = Value::integer(r);
        return Value::object(reinterpret_cast<PyObj*>(t));
    }
    double x = 0, y = 0;
    if (as_f64(args[0], x) && as_f64(args[1], y)) {
        if (y == 0.0) return error_msg("divmod() by zero");
        double q = std::floor(x / y);
        double r = x - q * y;
        Runtime& rt = Runtime::instance();
        PyTupleObj* t = rt.new_tuple(2);
        t->items[0] = Value::real(q);
        t->items[1] = Value::real(r);
        return Value::object(reinterpret_cast<PyObj*>(t));
    }
    return error_msg("divmod() needs numbers");
}

// --- pow ----------------------------------------------------------------------
Value bi_pow(void*, Value* args, std::uint32_t argc) noexcept {
    if (argc < 2 || argc > 3) return error_msg("pow() takes 2 or 3 arguments");
    std::int64_t a = 0, b = 0;
    if (as_i64(args[0], a) && as_i64(args[1], b)) {
        if (argc == 3) {
            std::int64_t m = 0;
            if (!as_i64(args[2], m) || m <= 0) return error_msg("pow() modulus must be positive int");
            // Modular exponentiation
            std::int64_t base = a % m;
            if (base < 0) base += m;
            std::int64_t result = 1;
            std::uint64_t exp = static_cast<std::uint64_t>(b);
            while (exp > 0) {
                if (exp & 1) result = (result * base) % m;
                base = (base * base) % m;
                exp >>= 1;
            }
            return Value::integer(result);
        }
        // Integer pow without modulus
        if (b < 0) {
            // Negative exponent → float
            return Value::real(std::pow(static_cast<double>(a), static_cast<double>(b)));
        }
        std::int64_t r = 1;
        for (std::int64_t i = 0; i < b; ++i) r *= a;
        return Value::integer(r);
    }
    double x = 0, y = 0;
    if (as_f64(args[0], x) && as_f64(args[1], y)) {
        return Value::real(std::pow(x, y));
    }
    return error_msg("pow() needs numbers");
}

// --- any / all ----------------------------------------------------------------
Value bi_any(void*, Value* args, std::uint32_t argc) noexcept {
    if (argc != 1) return error_msg("any() takes one argument");
    if (!g_vm) return error_msg("any() without vm context");
    Value it;
    if (!g_vm->get_iter(args[0], it)) return Value::object(nullptr);
    bool more = false;
    if (!g_vm->iter_check(it, more)) return Value::object(nullptr);
    while (more) {
        Value item;
        if (!g_vm->iter_next(it, item)) return Value::object(nullptr);
        if (Runtime::instance().truthy(item)) {
            return Value::boolean(true);
        }
        if (!g_vm->iter_check(it, more)) return Value::object(nullptr);
    }
    return Value::boolean(false);
}

Value bi_all(void*, Value* args, std::uint32_t argc) noexcept {
    if (argc != 1) return error_msg("all() takes one argument");
    if (!g_vm) return error_msg("all() without vm context");
    Value it;
    if (!g_vm->get_iter(args[0], it)) return Value::object(nullptr);
    bool more = false;
    if (!g_vm->iter_check(it, more)) return Value::object(nullptr);
    while (more) {
        Value item;
        if (!g_vm->iter_next(it, item)) return Value::object(nullptr);
        if (!Runtime::instance().truthy(item)) {
            return Value::boolean(false);
        }
        if (!g_vm->iter_check(it, more)) return Value::object(nullptr);
    }
    return Value::boolean(true);
}

// --- reversed -----------------------------------------------------------------
Value bi_reversed(void*, Value* args, std::uint32_t argc) noexcept {
    if (argc != 1) return error_msg("reversed() takes one argument");
    Runtime& rt = Runtime::instance();
    if (args[0].tag == Tag::Obj && args[0].as.obj) {
        if (args[0].as.obj->tag == ObjTag::List) {
            auto* l = static_cast<PyListObj*>(args[0].as.obj);
            auto* out = rt.new_list(l->length);
            for (std::int64_t i = static_cast<std::int64_t>(l->length) - 1; i >= 0; --i) {
                Value v = l->items[i];
                if (v.tag == Tag::Obj && v.as.obj) rt.incref(v.as.obj);
                list_push(out, v);
            }
            return Value::object(reinterpret_cast<PyObj*>(out));
        }
        if (args[0].as.obj->tag == ObjTag::Str) {
            auto* s = static_cast<PyStrObj*>(args[0].as.obj);
            stdx::small_vector<char, 128> rev;
            rev.resize(s->length);
            for (std::uint32_t i = 0; i < s->length; ++i) {
                rev[i] = s->data()[s->length - 1 - i];
            }
            auto* out = rt.new_str(std::string_view(rev.data(), s->length));
            return Value::object(reinterpret_cast<PyObj*>(out));
        }
    }
    return error_msg("reversed() argument is not reversible");
}

// --- map / filter (VM callback) ----------------------------------------------
// map(f, iter) → list of f(x) for x in iter
// We need the VM to call f for each element.
Value bi_map(void*, Value* args, std::uint32_t argc) noexcept {
    if (argc != 2) return error_msg("map() takes two arguments");
    if (!g_vm) return error_msg("map() without vm context");
    Value& func = args[0];
    Value& iterable = args[1];
    Runtime& rt = Runtime::instance();
    auto* out = rt.new_list(0);
    Value it;
    if (!g_vm->get_iter(iterable, it)) return Value::object(nullptr);
    bool more = false;
    if (!g_vm->iter_check(it, more)) return Value::object(nullptr);
    while (more) {
        Value item;
        if (!g_vm->iter_next(it, item)) return Value::object(nullptr);
        Value mapped;
        if (!g_vm->call_value(func, &item, 1, mapped)) return Value::object(nullptr);
        if (mapped.tag == Tag::Obj && mapped.as.obj) rt.incref(mapped.as.obj);
        list_push(out, mapped);
        if (!g_vm->iter_check(it, more)) return Value::object(nullptr);
    }
    return Value::object(reinterpret_cast<PyObj*>(out));
}

// filter(f, iter) → list of x for x in iter where truthy(f(x))
// filter(None, iter) → list of truthy items
Value bi_filter(void*, Value* args, std::uint32_t argc) noexcept {
    if (argc != 2) return error_msg("filter() takes two arguments");
    if (!g_vm) return error_msg("filter() without vm context");
    Value& func = args[0];
    Value& iterable = args[1];
    Runtime& rt = Runtime::instance();
    auto* out = rt.new_list(0);
    Value it;
    if (!g_vm->get_iter(iterable, it)) return Value::object(nullptr);
    bool more = false;
    if (!g_vm->iter_check(it, more)) return Value::object(nullptr);
    while (more) {
        Value item;
        if (!g_vm->iter_next(it, item)) return Value::object(nullptr);
        bool keep = false;
        if (func.tag == Tag::None) {
            keep = rt.truthy(item);
        } else {
            Value result;
            if (!g_vm->call_value(func, &item, 1, result)) return Value::object(nullptr);
            keep = rt.truthy(result);
        }
        if (keep) {
            if (item.tag == Tag::Obj && item.as.obj) rt.incref(item.as.obj);
            list_push(out, item);
        }
        if (!g_vm->iter_check(it, more)) return Value::object(nullptr);
    }
    return Value::object(reinterpret_cast<PyObj*>(out));
}

// --- chr / ord ----------------------------------------------------------------
Value bi_chr(void*, Value* args, std::uint32_t argc) noexcept {
    if (argc != 1) return error_msg("chr() takes one argument");
    std::int64_t cp = 0;
    if (!as_i64(args[0], cp)) return error_msg("chr() needs an int");
    if (cp < 0 || cp > 0x10FFFF) return error_msg("chr() arg out of range");
    Runtime& rt = Runtime::instance();
    // UTF-8 encode the code point
    char buf[4];
    std::uint32_t len = 0;
    if (cp < 0x80) {
        buf[0] = static_cast<char>(cp);
        len = 1;
    } else if (cp < 0x800) {
        buf[0] = static_cast<char>(0xC0 | (cp >> 6));
        buf[1] = static_cast<char>(0x80 | (cp & 0x3F));
        len = 2;
    } else if (cp < 0x10000) {
        buf[0] = static_cast<char>(0xE0 | (cp >> 12));
        buf[1] = static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        buf[2] = static_cast<char>(0x80 | (cp & 0x3F));
        len = 3;
    } else {
        buf[0] = static_cast<char>(0xF0 | (cp >> 18));
        buf[1] = static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        buf[2] = static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        buf[3] = static_cast<char>(0x80 | (cp & 0x3F));
        len = 4;
    }
    auto* s = rt.new_str(std::string_view(buf, len));
    return Value::object(reinterpret_cast<PyObj*>(s));
}

Value bi_ord(void*, Value* args, std::uint32_t argc) noexcept {
    if (argc != 1) return error_msg("ord() takes one argument");
    if (args[0].tag != Tag::Obj || !args[0].as.obj ||
        args[0].as.obj->tag != ObjTag::Str) {
        return error_msg("ord() needs a string");
    }
    auto* s = static_cast<PyStrObj*>(args[0].as.obj);
    if (s->length == 0) return error_msg("ord() needs a non-empty string");
    // Decode first UTF-8 code point
    unsigned char c0 = static_cast<unsigned char>(s->data()[0]);
    std::int64_t cp = 0;
    if (c0 < 0x80) {
        cp = c0;
    } else if ((c0 & 0xE0) == 0xC0 && s->length >= 2) {
        cp = (c0 & 0x1F) << 6;
        cp |= (static_cast<unsigned char>(s->data()[1]) & 0x3F);
    } else if ((c0 & 0xF0) == 0xE0 && s->length >= 3) {
        cp = (c0 & 0x0F) << 12;
        cp |= (static_cast<unsigned char>(s->data()[1]) & 0x3F) << 6;
        cp |= (static_cast<unsigned char>(s->data()[2]) & 0x3F);
    } else if ((c0 & 0xF8) == 0xF0 && s->length >= 4) {
        cp = (c0 & 0x07) << 18;
        cp |= (static_cast<unsigned char>(s->data()[1]) & 0x3F) << 12;
        cp |= (static_cast<unsigned char>(s->data()[2]) & 0x3F) << 6;
        cp |= (static_cast<unsigned char>(s->data()[3]) & 0x3F);
    } else {
        return error_msg("ord() invalid UTF-8");
    }
    return Value::integer(cp);
}

// --- hex / oct / bin ----------------------------------------------------------
Value bi_hex(void*, Value* args, std::uint32_t argc) noexcept {
    if (argc != 1) return error_msg("hex() takes one argument");
    std::int64_t v = 0;
    if (!as_i64(args[0], v)) return error_msg("hex() needs an int");
    char buf[20];
    int n;
    if (v < 0) {
        n = std::snprintf(buf, sizeof(buf), "-0x%llx", static_cast<unsigned long long>(-v));
    } else {
        n = std::snprintf(buf, sizeof(buf), "0x%llx", static_cast<unsigned long long>(v));
    }
    Runtime& rt = Runtime::instance();
    auto* s = rt.new_str(std::string_view(buf, static_cast<std::size_t>(n)));
    return Value::object(reinterpret_cast<PyObj*>(s));
}

Value bi_oct(void*, Value* args, std::uint32_t argc) noexcept {
    if (argc != 1) return error_msg("oct() takes one argument");
    std::int64_t v = 0;
    if (!as_i64(args[0], v)) return error_msg("oct() needs an int");
    char buf[24];
    int n;
    if (v < 0) {
        n = std::snprintf(buf, sizeof(buf), "-0o%llo", static_cast<unsigned long long>(-v));
    } else {
        n = std::snprintf(buf, sizeof(buf), "0o%llo", static_cast<unsigned long long>(v));
    }
    Runtime& rt = Runtime::instance();
    auto* s = rt.new_str(std::string_view(buf, static_cast<std::size_t>(n)));
    return Value::object(reinterpret_cast<PyObj*>(s));
}

Value bi_bin(void*, Value* args, std::uint32_t argc) noexcept {
    if (argc != 1) return error_msg("bin() takes one argument");
    std::int64_t v = 0;
    if (!as_i64(args[0], v)) return error_msg("bin() needs an int");
    // C's printf has no %b format specifier — manually build the binary string.
    bool negative = v < 0;
    std::uint64_t u = negative ? static_cast<std::uint64_t>(-v) : static_cast<std::uint64_t>(v);
    char tmp[66];
    int pos = 0;
    if (u == 0) {
        tmp[pos++] = '0';
    } else {
        while (u > 0) {
            tmp[pos++] = '0' + (u & 1);
            u >>= 1;
        }
    }
    // Reverse into the output buffer: sign + "0b" + digits
    char buf[70];
    int out_pos = 0;
    if (negative) buf[out_pos++] = '-';
    buf[out_pos++] = '0';
    buf[out_pos++] = 'b';
    for (int i = pos - 1; i >= 0; --i) {
        buf[out_pos++] = tmp[i];
    }
    Runtime& rt = Runtime::instance();
    auto* s = rt.new_str(std::string_view(buf, static_cast<std::size_t>(out_pos)));
    return Value::object(reinterpret_cast<PyObj*>(s));
}

// --- hash / id ----------------------------------------------------------------
Value bi_hash(void*, Value* args, std::uint32_t argc) noexcept {
    if (argc != 1) return error_msg("hash() takes one argument");
    // Hash by tag: int→itself, float→bit pattern, str→FNV-1a, bool→0/1
    switch (args[0].tag) {
        case Tag::Int: return Value::integer(args[0].as.i);
        case Tag::Bool: return Value::integer(args[0].as.i);
        case Tag::None: return Value::integer(0);
        case Tag::Float: {
            std::int64_t bits;
            std::memcpy(&bits, &args[0].as.f, sizeof(double));
            return Value::integer(bits);
        }
        case Tag::Obj: {
            if (args[0].as.obj && args[0].as.obj->tag == ObjTag::Str) {
                auto* s = static_cast<PyStrObj*>(args[0].as.obj);
                std::uint64_t h = 14695981039346656037ULL;
                for (std::uint32_t i = 0; i < s->length; ++i) {
                    h ^= static_cast<unsigned char>(s->data()[i]);
                    h *= 1099511628211ULL;
                }
                return Value::integer(static_cast<std::int64_t>(h));
            }
            return Value::integer(static_cast<std::int64_t>(
                reinterpret_cast<std::uintptr_t>(args[0].as.obj)));
        }
    }
    return error_msg("hash() unhashable type");
}

Value bi_id(void*, Value* args, std::uint32_t argc) noexcept {
    if (argc != 1) return error_msg("id() takes one argument");
    // For objects, return the pointer address; for ints/floats, return the value
    if (args[0].tag == Tag::Obj) {
        return Value::integer(static_cast<std::int64_t>(
            reinterpret_cast<std::uintptr_t>(args[0].as.obj)));
    }
    return Value::integer(args[0].as.i);
}

// --- callable -----------------------------------------------------------------
Value bi_callable(void*, Value* args, std::uint32_t argc) noexcept {
    if (argc != 1) return error_msg("callable() takes one argument");
    if (args[0].tag != Tag::Obj || !args[0].as.obj) return Value::boolean(false);
    switch (args[0].as.obj->tag) {
        case ObjTag::Function: case ObjTag::NativeFn:
        case ObjTag::Type: case ObjTag::BoundMethod:
        case ObjTag::Generator:
            return Value::boolean(true);
        default:
            return Value::boolean(false);
    }
}

// --- math module -------------------------------------------------------------------
Value math_call(void* user, Value* args, std::uint32_t argc) noexcept {
    auto* fn = reinterpret_cast<double (*)(double)>(user);
    if (argc != 1) return error_msg("math function takes one argument");
    double x = 0;
    if (!as_f64(args[0], x)) return error_msg("math function needs a number");
    return Value::real(fn(x));
}

Value math_pow(void*, Value* args, std::uint32_t argc) noexcept {
    if (argc != 2) return error_msg("math.pow takes two arguments");
    double x = 0, y = 0;
    if (!as_f64(args[0], x) || !as_f64(args[1], y)) {
        return error_msg("math.pow needs numbers");
    }
    return Value::real(std::pow(x, y));
}

Value math_sqrt(void*, Value* args, std::uint32_t argc) noexcept {
    if (argc != 1) return error_msg("math.sqrt takes one argument");
    double x = 0;
    if (!as_f64(args[0], x)) return error_msg("math.sqrt needs a number");
    if (x < 0) {
        if (g_vm) g_vm->raise_builtin(Runtime::instance().type_value_error,
                                      "math domain error");
        return Value::object(nullptr);
    }
    return Value::real(std::sqrt(x));
}

Value math_floor(void*, Value* args, std::uint32_t argc) noexcept {
    if (argc != 1) return error_msg("math.floor takes one argument");
    double x = 0;
    if (!as_f64(args[0], x)) return error_msg("math.floor needs a number");
    return Value::integer(static_cast<std::int64_t>(std::floor(x)));
}

Value math_ceil(void*, Value* args, std::uint32_t argc) noexcept {
    if (argc != 1) return error_msg("math.ceil takes one argument");
    double x = 0;
    if (!as_f64(args[0], x)) return error_msg("math.ceil needs a number");
    return Value::integer(static_cast<std::int64_t>(std::ceil(x)));
}

// --- time module ---------------------------------------------------------------------
Value time_time(void*, Value*, std::uint32_t) noexcept {
    return Value::real(static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(
                           std::chrono::steady_clock::now().time_since_epoch())
                           .count()) /
                       1e6);
}

/// time.perf_counter() — fractional seconds, highest-available resolution.
/// Uses steady_clock (monotonic, nanosecond precision on Linux). Returns
/// a float matching CPython's perf_counter semantics.
Value time_perf_counter(void*, Value*, std::uint32_t) noexcept {
    return Value::real(std::chrono::duration<double>(
                           std::chrono::steady_clock::now().time_since_epoch())
                           .count());
}

/// time.perf_counter_ns() — integer nanoseconds, highest-available
/// resolution. This is the workhorse for in-process benchmarking: it
/// avoids the float-conversion overhead of perf_counter() and gives
/// sub-microsecond timing precision on modern x86-64.
Value time_perf_counter_ns(void*, Value*, std::uint32_t) noexcept {
    return Value::integer(static_cast<std::int64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count()));
}

// --- random module (xorshift64 — deterministic under seed()) --------------------------
std::uint64_t rng_state = 0x9E3779B97F4A7C15ull;

Value random_seed(void*, Value* args, std::uint32_t argc) noexcept {
    std::int64_t s = 0;
    if (argc >= 1 && as_i64(args[0], s)) {
        rng_state = static_cast<std::uint64_t>(s) ^ 0x9E3779B97F4A7C15ull;
    } else {
        rng_state = static_cast<std::uint64_t>(std::chrono::steady_clock::now()
                                                  .time_since_epoch()
                                                  .count());
    }
    if (rng_state == 0) rng_state = 1;
    return Value::none();
}

Value random_random(void*, Value*, std::uint32_t) noexcept {
    std::uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    rng_state = x;
    return Value::real(static_cast<double>(x >> 11) / 9007199254740992.0);
}

Value random_randint(void*, Value* args, std::uint32_t argc) noexcept {
    if (argc != 2) return error_msg("randint takes two arguments");
    std::int64_t lo = 0, hi = 0;
    if (!as_i64(args[0], lo) || !as_i64(args[1], hi) || lo > hi) {
        return error_msg("randint needs lo <= hi");
    }
    std::uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    rng_state = x;
    return Value::integer(lo + static_cast<std::int64_t>(x % static_cast<std::uint64_t>(hi - lo + 1)));
}

Value random_choice(void*, Value* args, std::uint32_t argc) noexcept {
    if (argc != 1 || args[0].tag != Tag::Obj || !args[0].as.obj ||
        args[0].as.obj->tag != ObjTag::List || static_cast<PyListObj*>(args[0].as.obj)->length == 0) {
        return error_msg("choice needs a non-empty list");
    }
    auto* l = static_cast<PyListObj*>(args[0].as.obj);
    std::uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    rng_state = x;
    return l->items[x % l->length];
}

}  // namespace

// =============================================================================
// Installation
// =============================================================================
void install_builtins(Program& program) noexcept {
    Runtime& rt = Runtime::instance();
    (void)g_vm;   // set by the embedder (Vm owns the program)

    struct Entry {
        const char* name;
        NativeFn fn;
    };
    static const Entry entries[] = {
        {"print", bi_print}, {"len", bi_len}, {"range", bi_range},
        {"abs", bi_abs}, {"min", bi_min}, {"max", bi_max}, {"sum", bi_sum},
        {"str", bi_str}, {"repr", bi_repr}, {"int", bi_int}, {"float", bi_float},
        {"bool", bi_bool}, {"list", bi_list}, {"tuple", bi_tuple}, {"dict", bi_dict},
        {"enumerate", bi_enumerate}, {"zip", bi_zip}, {"sorted", bi_sorted},
        {"next", bi_next}, {"isinstance", bi_isinstance}, {"type", bi_type},
        // Python 3.16 features (batch): missing builtins.
        {"round", bi_round}, {"divmod", bi_divmod}, {"pow", bi_pow},
        {"any", bi_any}, {"all", bi_all}, {"reversed", bi_reversed},
        {"map", bi_map}, {"filter", bi_filter},
        {"chr", bi_chr}, {"ord", bi_ord},
        {"hex", bi_hex}, {"oct", bi_oct}, {"bin", bi_bin},
        {"hash", bi_hash}, {"id", bi_id}, {"callable", bi_callable},
    };
    for (const Entry& e : entries) {
        SymbolId sym = global_symbols().intern(e.name);
        auto* fn = rt.new_native(sym, e.fn, nullptr);
        dict_set(program.globals, Value::integer(sym),
                 Value::object(reinterpret_cast<PyObj*>(fn)));
        if (program.globals->count > 100) {
        }
    }
    // Exception classes & True/False/None names.
    struct TypeEntry {
        const char* name;
        PyTypeObj* type;
    };
    static const TypeEntry types[] = {
        {"Exception", nullptr}, {"ValueError", nullptr}, {"TypeError", nullptr},
        {"ZeroDivisionError", nullptr}, {"IndexError", nullptr}, {"KeyError", nullptr},
        {"StopIteration", nullptr}, {"RuntimeError", nullptr},
        {"AssertionError", nullptr}, {"AttributeError", nullptr},
        {"NameError", nullptr}, {"MemoryError", nullptr},
        {"NotImplementedError", nullptr},
    };
    // (types filled from Runtime singletons on each install)
    PyTypeObj* type_map[] = {
        rt.type_exc_base, rt.type_value_error, rt.type_type_error, rt.type_zero_div,
        rt.type_index_error, rt.type_key_error, rt.type_stop_iter, rt.type_runtime_error,
        rt.type_assertion_error, rt.type_attribute_error, rt.type_name_error,
        rt.type_memory_error, rt.type_not_implemented_error,
    };
    for (std::size_t i = 0; i < sizeof(types) / sizeof(types[0]); ++i) {
        SymbolId sym = global_symbols().intern(types[i].name);
        dict_set(program.globals, Value::integer(sym),
                 Value::object(reinterpret_cast<PyObj*>(type_map[i])));
    }
}
void set_vm_for_builtins(Vm* vm) noexcept { g_vm = vm; }
Vm* active_vm() noexcept { return g_vm; }
PyModuleObj* load_native_module(std::uint32_t name_symbol) noexcept {
    Runtime& rt = Runtime::instance();
    std::string_view name = global_symbols().text(name_symbol);
    PyModuleObj* mod = rt.new_module(name_symbol);
    bool known = false;
    if (name == "math") {
        known = true;
        struct MF { const char* n; double (*f)(double); };
        static const MF fns[] = {
            {"sin", std::sin}, {"cos", std::cos}, {"tan", std::tan},
            {"exp", std::exp}, {"log", std::log}, {"log2", std::log2},
            {"log10", std::log10}, {"asin", std::asin}, {"acos", std::acos},
            {"atan", std::atan}, {"sinh", std::sinh}, {"cosh", std::cosh},
            {"tanh", std::tanh},
        };
        for (const MF& m : fns) {
            SymbolId sym = global_symbols().intern(m.n);
            auto* fn = rt.new_native(sym, math_call, reinterpret_cast<void*>(m.f));
            dict_set(mod->ns, Value::integer(sym),
                     Value::object(reinterpret_cast<PyObj*>(fn)));
        }
        dict_set(mod->ns, Value::integer(global_symbols().intern("sqrt")),
                 Value::object(reinterpret_cast<PyObj*>(
                     rt.new_native(global_symbols().intern("sqrt"), math_sqrt, nullptr))));
        dict_set(mod->ns, Value::integer(global_symbols().intern("floor")),
                 Value::object(reinterpret_cast<PyObj*>(
                     rt.new_native(global_symbols().intern("floor"), math_floor, nullptr))));
        dict_set(mod->ns, Value::integer(global_symbols().intern("ceil")),
                 Value::object(reinterpret_cast<PyObj*>(
                     rt.new_native(global_symbols().intern("ceil"), math_ceil, nullptr))));
        dict_set(mod->ns, Value::integer(global_symbols().intern("pow")),
                 Value::object(reinterpret_cast<PyObj*>(
                     rt.new_native(global_symbols().intern("pow"), math_pow, nullptr))));
        dict_set(mod->ns, Value::integer(global_symbols().intern("pi")),
                 Value::real(3.14159265358979323846));
        dict_set(mod->ns, Value::integer(global_symbols().intern("e")),
                 Value::real(2.71828182845904523536));
        dict_set(mod->ns, Value::integer(global_symbols().intern("inf")),
                 Value::real(HUGE_VAL));
    } else if (name == "time") {
        known = true;
        dict_set(mod->ns, Value::integer(global_symbols().intern("time")),
                 Value::object(reinterpret_cast<PyObj*>(
                     rt.new_native(global_symbols().intern("time"), time_time, nullptr))));
        dict_set(mod->ns, Value::integer(global_symbols().intern("perf_counter")),
                 Value::object(reinterpret_cast<PyObj*>(
                     rt.new_native(global_symbols().intern("perf_counter"),
                                   time_perf_counter, nullptr))));
        dict_set(mod->ns, Value::integer(global_symbols().intern("perf_counter_ns")),
                 Value::object(reinterpret_cast<PyObj*>(
                     rt.new_native(global_symbols().intern("perf_counter_ns"),
                                   time_perf_counter_ns, nullptr))));
    } else if (name == "random") {
        known = true;
        auto add = [&](const char* n, NativeFn fn) {
            SymbolId sym = global_symbols().intern(n);
            dict_set(mod->ns, Value::integer(sym),
                     Value::object(reinterpret_cast<PyObj*>(rt.new_native(sym, fn, nullptr))));
        };
        add("seed", random_seed);
        add("random", random_random);
        add("randint", random_randint);
        add("choice", random_choice);
    }
    if (!known) {
        rt.decref(reinterpret_cast<PyObj*>(mod));
        return nullptr;
    }
    return mod;
}
}  // namespace abi_v1
}  // namespace vortex::rt
