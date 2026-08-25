// =============================================================================
// vortex/rt/objects.cpp — object model implementation: allocation, bignum,
// containers, numeric tower, hashing, repr.
// =============================================================================

#include "vortex/rt/object.hpp"

#include "vortex/ir/node.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>

#include "vortex/support/symbol_table.hpp"

namespace vortex::rt {
inline namespace abi_v1 {

namespace {
void push_cstr(stdx::small_vector<char, 128>& out, const char* s) noexcept {
    for (const char* p = s; *p; ++p) out.push_back(*p);
}

/// Python float repr: shortest round-trip with mandatory '.' or exponent
/// (integers-as-floats print as "5.0", matching CPython semantics).
void format_double(double v, stdx::small_vector<char, 128>& out) noexcept {
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.17g", v);
    bool has_dot = false;
    bool has_exp = false;
    for (char* p = buf; *p; ++p) {
        if (*p == '.' || *p == ',') has_dot = true;
        if (*p == 'e' || *p == 'E') has_exp = true;
    }
    push_cstr(out, buf);
    if (!has_dot && !has_exp) push_cstr(out, ".0");
}
// -----------------------------------------------------------------------------
// Region-based object memory (Pass 42 architecture).
//
// All PyObj payloads allocate from thread-local bump regions; decref-to-
// zero marks objects dead and runs destructors for C++ members, but the
// memory itself is only reclaimed when the region resets (process exit
// for a compiler workload). This:
//   - removes glibc-heap interleaving with compiler allocations,
//   - makes every PyObj pointer stable for the program lifetime,
//   - eliminates the use-after-free class of bugs entirely.
// The regions grow geometrically; a RegionHeader chain allows full teardown
// in tests. Malloc remains for internal buffers (items arrays, frames).
// -----------------------------------------------------------------------------
struct RegionHeader {
    RegionHeader* next;
    std::size_t used;
    std::size_t size;
    std::byte* base() noexcept { return reinterpret_cast<std::byte*>(this) + sizeof(RegionHeader); }
};

thread_local RegionHeader* t_region = nullptr;

[[nodiscard]] void* region_alloc(std::size_t bytes) noexcept {
    constexpr std::size_t min_region = 64 * 1024;
    std::size_t need = (bytes + 15) & ~std::size_t(15);
    if (!t_region || t_region->used + need > t_region->size) {
        std::size_t region_size = need > min_region ? need : min_region;
        auto* region = static_cast<RegionHeader*>(std::malloc(region_size + sizeof(RegionHeader)));
        if (!region) [[unlikely]] {
            std::fputs("VORTEX FATAL: object region exhausted\n", stderr);
            std::abort();
        }
        region->next = t_region;
        region->used = 0;
        region->size = region_size;
        t_region = region;
    }
    void* p = t_region->base() + t_region->used;
    t_region->used += need;
    return p;
}

/// Free all object regions (test teardown).
void regions_release_all() noexcept {
    RegionHeader* r = t_region;
    while (r) {
        RegionHeader* next = r->next;
        std::free(r);
        r = next;
    }
    t_region = nullptr;
}

[[nodiscard]] void* heap_alloc(std::size_t bytes) noexcept {
    return region_alloc(bytes);
}

/// Constructs a heap object with C++ members (placement new is MANDATORY:
/// malloc alone leaves small_vector members with garbage begin_ pointers —
/// the exact bug class behind the new_long_big use-after-free).
template <typename T>
T* heap_new() noexcept {
    void* p = heap_alloc(sizeof(T));
    return new (p) T();
}

[[nodiscard]] std::uint32_t hash_bytes(std::string_view s) noexcept {
    std::uint32_t h = 2166136261u;
    for (char c : s) {
        h ^= static_cast<std::uint8_t>(c);
        h *= 16777619u;
    }
    return h ? h : 1;
}

[[nodiscard]] std::uint32_t mix32(std::uint32_t x) noexcept {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x ? x : 1;
}

// 64-bit hash folded to 32.
[[nodiscard]] std::uint32_t fold64(std::uint64_t x) noexcept {
    return mix32(static_cast<std::uint32_t>(x) ^ static_cast<std::uint32_t>(x >> 32));
}

[[nodiscard]] bool int_fits_i64(std::int64_t a, std::int64_t b, std::int64_t& out) noexcept {
    return !__builtin_add_overflow(a, b, &out);
}
}  // namespace

// =============================================================================
// BigNum
// =============================================================================
BigNum BigNum::from_i64(std::int64_t v) noexcept {
    BigNum b;
    b.negative = v < 0;
    std::uint64_t mag = b.negative ? static_cast<std::uint64_t>(-(v + 1)) + 1
                                   : static_cast<std::uint64_t>(v);
    while (mag) {
        b.limbs.push_back(static_cast<std::uint32_t>(mag & 0xFFFFFFFFu));
        mag >>= 32;
    }
    return b;
}

std::int64_t BigNum::try_i64(bool& fits) const noexcept {
    fits = true;
    if (limbs.size() > 2) {
        fits = false;
        return 0;
    }
    std::uint64_t mag = 0;
    if (!limbs.empty()) mag = limbs[0];
    if (limbs.size() == 2) mag |= static_cast<std::uint64_t>(limbs[1]) << 32;
    if (negative) {
        if (mag > 0x8000000000000000ull) { fits = false; return 0; }
        return static_cast<std::int64_t>(~mag + 1);
    }
    if (mag > 0x7FFFFFFFFFFFFFFFull) { fits = false; return 0; }
    return static_cast<std::int64_t>(mag);
}

int BigNum::cmp(const BigNum& o) const noexcept {
    if (negative != o.negative) return negative ? -1 : 1;
    int sign = negative ? -1 : 1;
    if (limbs.size() != o.limbs.size()) {
        return (limbs.size() < o.limbs.size()) ? -sign : sign;
    }
    for (std::size_t i = limbs.size(); i-- > 0;) {
        if (limbs[i] != o.limbs[i]) return limbs[i] < o.limbs[i] ? -sign : sign;
    }
    return 0;
}

[[nodiscard]] static BigNum mag_add(const BigNum& a, const BigNum& b) noexcept {
    BigNum r;
    std::uint64_t carry = 0;
    std::size_t n = a.limbs.size() > b.limbs.size() ? a.limbs.size() : b.limbs.size();
    for (std::size_t i = 0; i < n; ++i) {
        std::uint64_t s = carry;
        if (i < a.limbs.size()) s += a.limbs[i];
        if (i < b.limbs.size()) s += b.limbs[i];
        r.limbs.push_back(static_cast<std::uint32_t>(s & 0xFFFFFFFFu));
        carry = s >> 32;
    }
    if (carry) r.limbs.push_back(static_cast<std::uint32_t>(carry));
    return r;
}

[[nodiscard]] static BigNum mag_sub(const BigNum& a, const BigNum& b) noexcept {
    // requires |a| >= |b|
    BigNum r;
    std::int64_t borrow = 0;
    for (std::size_t i = 0; i < a.limbs.size(); ++i) {
        std::int64_t s = static_cast<std::int64_t>(a.limbs[i]) - borrow -
                         (i < b.limbs.size() ? static_cast<std::int64_t>(b.limbs[i]) : 0);
        if (s < 0) {
            s += (1ll << 32);
            borrow = 1;
        } else {
            borrow = 0;
        }
        r.limbs.push_back(static_cast<std::uint32_t>(s));
    }
    while (!r.limbs.empty() && r.limbs.back() == 0) r.limbs.pop_back();
    return r;
}

BigNum BigNum::add(const BigNum& a, const BigNum& b) noexcept {
    if (a.negative == b.negative) {
        BigNum r = mag_add(a, b);
        r.negative = a.negative;
        return r;
    }
    // signs differ: subtract smaller magnitude from larger
    int c = BigNum{}.cmp(BigNum{});  // placeholder to silence unused warnings
    (void)c;
    BigNum aabs = a; aabs.negative = false;
    BigNum babs = b; babs.negative = false;
    int m = aabs.cmp(babs);
    if (m == 0) return BigNum{};
    if (m > 0) {
        BigNum r = mag_sub(aabs, babs);
        r.negative = a.negative;
        return r;
    }
    BigNum r = mag_sub(babs, aabs);
    r.negative = b.negative;
    return r;
}

BigNum BigNum::sub(const BigNum& a, const BigNum& b) noexcept {
    BigNum nb = b;
    nb.negative = !b.negative;
    return add(a, nb);
}

BigNum BigNum::mul(const BigNum& a, const BigNum& b) noexcept {
    BigNum r;
    if (a.is_zero() || b.is_zero()) return r;
    r.limbs.assign(a.limbs.size() + b.limbs.size(), 0);
    for (std::size_t i = 0; i < a.limbs.size(); ++i) {
        std::uint64_t carry = 0;
        for (std::size_t j = 0; j < b.limbs.size(); ++j) {
            std::uint64_t cur = r.limbs[i + j] +
                                static_cast<std::uint64_t>(a.limbs[i]) * b.limbs[j] + carry;
            r.limbs[i + j] = static_cast<std::uint32_t>(cur & 0xFFFFFFFFu);
            carry = cur >> 32;
        }
        std::size_t k = i + b.limbs.size();
        while (carry) {
            std::uint64_t cur = r.limbs[k] + carry;
            r.limbs[k] = static_cast<std::uint32_t>(cur & 0xFFFFFFFFu);
            carry = cur >> 32;
            ++k;
        }
    }
    while (!r.limbs.empty() && r.limbs.back() == 0) r.limbs.pop_back();
    r.negative = a.negative != b.negative;
    return r;
}

BigNum BigNum::divmod(const BigNum& a, const BigNum& b, BigNum& rem, bool& div_by_zero) noexcept {
    div_by_zero = false;
    if (b.is_zero()) {
        div_by_zero = true;
        rem = BigNum{};
        return BigNum{};
    }
    BigNum aabs = a; aabs.negative = false;
    BigNum babs = b; babs.negative = false;
    // Long division on 32-bit limbs (bit-by-bit over limb chunks — simple,
    // correct; performance acceptable for the subset's bignum path).
    if (aabs.cmp(babs) < 0) {
        rem = a;
        return BigNum{};
    }
    BigNum q;
    BigNum r;
    for (std::size_t i = aabs.limbs.size(); i-- > 0;) {
        // shift r left by 32 bits, bring in limb
        r.limbs.insert(0, aabs.limbs[i]);
        if (r.cmp(babs) >= 0) {
            // find largest k with k*b <= r via binary search on 32-bit k
            std::uint32_t lo = 0, hi = 0xFFFFFFFFu;
            BigNum kb, rr;
            while (lo < hi) {
                std::uint32_t mid = lo + (hi - lo) / 2 + 1;
                BigNum kbn = mul(babs, BigNum::from_i64(static_cast<std::int64_t>(mid)));
                int c = kbn.cmp(r);
                if (c <= 0) {
                    lo = mid;
                    kb = kbn;
                } else {
                    hi = mid - 1;
                }
            }
            kb = mul(babs, BigNum::from_i64(static_cast<std::int64_t>(lo)));
            r = mag_sub(r, kb);
            q.limbs.insert(0, lo);
        } else if (!q.limbs.empty()) {
            q.limbs.insert(0, 0);
        }
    }
    while (!q.limbs.empty() && q.limbs.back() == 0) q.limbs.pop_back();
    q.negative = a.negative != b.negative;
    rem = r;
    rem.negative = a.negative;
    return q;
}

BigNum BigNum::pow_small(const BigNum& base, std::uint64_t exp) noexcept {
    BigNum result = BigNum::from_i64(1);
    BigNum b = base;
    while (exp) {
        if (exp & 1) result = mul(result, b);
        b = mul(b, b);
        exp >>= 1;
    }
    return result;
}

std::size_t BigNum::hash() const noexcept {
    std::uint64_t h = 14695981039346656037ull;
    for (std::uint32_t l : limbs) h = (h ^ l) * 1099511628211ull;
    return h ^ (negative ? 0x9e3779b97f4a7c15ull : 0);
}

void BigNum::to_string(stdx::small_vector<char, 128>& out) const noexcept {
    if (is_zero()) {
        out.push_back('0');
        return;
    }
    BigNum v = *this;
    v.negative = false;
    // repeated division by 10^9
    stdx::small_vector<std::uint32_t, 16> chunks;
    BigNum ten9 = BigNum::from_i64(1000000000);
    while (!v.is_zero()) {
        BigNum rem;
        bool dz = false;
        v = divmod(v, ten9, rem, dz);
        bool fits = false;
        std::int64_t c = rem.try_i64(fits);
        chunks.push_back(static_cast<std::uint32_t>(fits ? c : 0));
    }
    if (negative) out.push_back('-');
    for (std::size_t i = chunks.size(); i-- > 0;) {
        char buf[16];
        if (i == chunks.size() - 1) {
            std::snprintf(buf, sizeof(buf), "%u", chunks[i]);
        } else {
            std::snprintf(buf, sizeof(buf), "%09u", chunks[i]);
        }
        for (char* p = buf; *p; ++p) out.push_back(*p);
    }
}

// =============================================================================
// Runtime
// =============================================================================
Runtime::Runtime() noexcept {
    none = static_cast<PyNoneObj*>(heap_alloc(sizeof(PyNoneObj)));
    none->tag = ObjTag::None;
    none->flags = 0;
    none->refcount = 0x40000000;   // immortal

    true_obj = static_cast<PyBoolObj*>(heap_alloc(sizeof(PyBoolObj)));
    true_obj->tag = ObjTag::Bool;
    true_obj->value = true;
    true_obj->refcount = 0x40000000;

    false_obj = static_cast<PyBoolObj*>(heap_alloc(sizeof(PyBoolObj)));
    false_obj->tag = ObjTag::Bool;
    false_obj->value = false;
    false_obj->refcount = 0x40000000;

    auto mk_type = [&](const char* name, PyTypeObj* base) -> PyTypeObj* {
        auto* t = static_cast<PyTypeObj*>(heap_alloc(sizeof(PyTypeObj)));
        t->tag = ObjTag::Type;
        t->refcount = 1;   // MUST own: region memory is NOT zeroed
        t->flags = 0;   // clear malloc garbage (kBigFlag class of bugs)
        t->pad = 0;
        t->name_symbol = global_symbols().intern(name);
        t->base = base;
        t->dict = new_dict();
        t->type_id = next_type_id_++;
        auto* root = heap_new<ShapeNode>();
        root->id = next_shape_id_++;
        t->root_shape = root;
        shape_nodes_.push_back(root);
        return t;
    };

    type_exc_base = mk_type("Exception", nullptr);
    type_value_error = mk_type("ValueError", type_exc_base);
    type_type_error = mk_type("TypeError", type_exc_base);
    type_zero_div = mk_type("ZeroDivisionError", type_exc_base);
    type_index_error = mk_type("IndexError", type_exc_base);
    type_key_error = mk_type("KeyError", type_exc_base);
    type_stop_iter = mk_type("StopIteration", type_exc_base);
    type_runtime_error = mk_type("RuntimeError", type_exc_base);
    type_assertion_error = mk_type("AssertionionError", type_exc_base);
    type_attribute_error = mk_type("AttributeError", type_exc_base);
    type_memory_error = mk_type("MemoryError", type_exc_base);
    type_not_implemented_error = mk_type("NotImplementedError", type_exc_base);
    type_name_error = mk_type("NameError", type_exc_base);
    // fix the typo'd symbol: AssertionError
    type_assertion_error->name_symbol = global_symbols().intern("AssertionError");
    type_int = mk_type("int", nullptr);
}

Runtime& Runtime::instance() noexcept {
    static Runtime rt;
    return rt;
}

void Runtime::decref(PyObj* o) noexcept {
    if (!o || o->refcount >= 0x40000000) return;   // immortal singletons
    if (o->refcount == 0) [[unlikely]] {
        // Over-release: a real bug — fail loudly (never silently corrupt).
        std::fprintf(stderr,
                     "VORTEX FATAL: over-decref obj=%p tag=%d str=[%.*s]\n", (void*)o,
                     (int)o->tag,
                     o->tag == ObjTag::Str
                         ? static_cast<int>(static_cast<PyStrObj*>(o)->length)
                         : 0,
                     o->tag == ObjTag::Str ? static_cast<PyStrObj*>(o)->data() : "");
        std::abort();
    }
    if (--o->refcount == 0) {
        // Mark dead: region memory is reclaimed wholesale (regions_release_
        // all at teardown); here we only run C++ member destructors.
        o->refcount = 0;
        switch (o->tag) {
            case ObjTag::Long: {
                auto* l = static_cast<PyLongObj*>(o);
                l->big.~BigNum();   // small_vector member: explicit dtor
                /* region-managed */
                break;
            }
            case ObjTag::Float: case ObjTag::Bool:
            case ObjTag::None: case ObjTag::Type: case ObjTag::NativeFn:
            case ObjTag::Module: case ObjTag::Cell:
                /* region-managed: no free */
                break;
            case ObjTag::Str:
                /* region-managed: no free */
                break;
            case ObjTag::List: {
                auto* l = static_cast<PyListObj*>(o);
                if (l->items) {
                    for (std::uint32_t i = 0; i < l->length; ++i) {
                        if (l->items[i].tag == Tag::Obj) decref(l->items[i].as.obj);
                    }
                    std::free(l->items);
                }
                /* region-managed */
                break;
            }
            case ObjTag::Tuple: {
                auto* t = static_cast<PyTupleObj*>(o);
                if (t->items) {
                    for (std::uint32_t i = 0; i < t->length; ++i) {
                        if (t->items[i].tag == Tag::Obj) decref(t->items[i].as.obj);
                    }
                    std::free(t->items);
                }
                /* region-managed */
                break;
            }
            case ObjTag::Dict: {
                auto* d = static_cast<PyDictObj*>(o);
                if (d->entries) {
                    for (std::uint32_t i = 0; i < d->capacity; ++i) {
                        if (d->entries[i].used) {
                            if (d->entries[i].key.tag == Tag::Obj) decref(d->entries[i].key.as.obj);
                            if (d->entries[i].value.tag == Tag::Obj) decref(d->entries[i].value.as.obj);
                        }
                    }
                    std::free(d->entries);
                }
                /* region-managed */
                break;
            }
            case ObjTag::Instance: {
                auto* inst = static_cast<PyInstanceObj*>(o);
                if (inst->slots) {
                    for (std::uint32_t i = 0; i < inst->slot_capacity; ++i) {
                        if (inst->slots[i].tag == Tag::Obj) decref(inst->slots[i].as.obj);
                    }
                    std::free(inst->slots);
                }
                /* region-managed */
                break;
            }
            case ObjTag::Function: {
                auto* f = static_cast<PyFuncObj*>(o);
                if (f->defaults) decref(reinterpret_cast<PyObj*>(f->defaults));
                if (f->cells) decref(reinterpret_cast<PyObj*>(f->cells));
                /* region-managed */
                break;
            }
            case ObjTag::Generator: {
                auto* g = static_cast<PyGeneratorObj*>(o);
                (void)g;   // frame teardown handled by the VM (owns frame)
                /* region-managed */
                break;
            }
            default:
                /* region-managed: no free */
                break;
        }
    }
}

PyStrObj* Runtime::new_str(std::string_view text) noexcept {
    auto* s = static_cast<PyStrObj*>(
        heap_alloc(sizeof(PyStrObj) + text.size() + 1));
    s->tag = ObjTag::Str;
    s->refcount = 1;   // MUST own: region memory is NOT zeroed
    s->flags = 0;   // clear malloc garbage (kBigFlag class of bugs)
    s->pad = 0;
    s->length = static_cast<std::uint32_t>(text.size());
    s->hash_cache = 0;   // MUST init: malloc'd garbage otherwise breaks hash
    s->hashed = false;
    std::memcpy(s->data(), text.data(), text.size());
    s->data()[text.size()] = '\0';
    ++allocations;
    bytes_allocated += sizeof(PyStrObj) + text.size() + 1;
    return s;
}

PyLongObj* Runtime::new_long_i64(std::int64_t v) noexcept {
    auto* l = heap_new<PyLongObj>();
    l->tag = ObjTag::Long;
    l->flags = 0;   // clear kBigFlag garbage
    l->value = v;
    ++allocations;
    return l;
}

PyLongObj* Runtime::new_long_big(BigNum b) noexcept {
    bool fits = false;
    std::int64_t v = b.try_i64(fits);
    auto* l = heap_new<PyLongObj>();
    l->tag = ObjTag::Long;
    l->flags = 0;   // clear kBigFlag garbage
    if (fits && b.limbs.size() <= 2) {
        l->value = v;
    } else {
        l->value = 0;
        l->flags |= PyLongObj::kBigFlag;
        l->big = std::move(b);
    }
    ++allocations;
    return l;
}

PyFloatObj* Runtime::new_float(double v) noexcept {
    auto* f = static_cast<PyFloatObj*>(heap_alloc(sizeof(PyFloatObj)));
    f->tag = ObjTag::Float;
    f->refcount = 1;   // MUST own: region memory is NOT zeroed
    f->flags = 0;   // clear malloc garbage (kBigFlag class of bugs)
    f->pad = 0;
    f->value = v;
    ++allocations;
    return f;
}

PyListObj* Runtime::new_list(std::uint32_t cap) noexcept {
    auto* l = static_cast<PyListObj*>(heap_alloc(sizeof(PyListObj)));
    l->tag = ObjTag::List;
    l->refcount = 1;   // MUST own: region memory is NOT zeroed
    l->flags = 0;   // clear malloc garbage (kBigFlag class of bugs)
    l->pad = 0;
    l->length = 0;      // MUST init (malloc garbage here caused wild writes)
    l->items = nullptr;
    l->capacity = 0;
    if (cap) {
        l->items = static_cast<Value*>(std::malloc(sizeof(Value) * cap));
        l->capacity = cap;
    }
    ++allocations;
    return l;
}

PyTupleObj* Runtime::new_tuple(std::uint32_t n) noexcept {
    auto* t = static_cast<PyTupleObj*>(heap_alloc(sizeof(PyTupleObj)));
    t->tag = ObjTag::Tuple;
    t->refcount = 1;   // MUST own: region memory is NOT zeroed
    t->flags = 0;   // clear malloc garbage (kBigFlag class of bugs)
    t->pad = 0;
    t->length = n;
    if (n) {
        t->items = static_cast<Value*>(std::calloc(n, sizeof(Value)));
        std::memset(t->items, 0, sizeof(Value) * n);
    }
    ++allocations;
    return t;
}

PyDictObj* Runtime::new_dict() noexcept {
    auto* d = static_cast<PyDictObj*>(heap_alloc(sizeof(PyDictObj)));
    d->tag = ObjTag::Dict;
    d->refcount = 1;   // MUST own: region memory is NOT zeroed
    d->flags = 0;   // clear malloc garbage (kBigFlag class of bugs)
    d->pad = 0;
    d->count = 0;         // MUST init: malloc garbage here silently corrupted
    d->insert_seq = 0;    // every consumer dict (the globals-dict worked only
                          // because fresh heap pages happened to be zeroed).
    d->capacity = 8;
    d->entries = static_cast<DictEntry*>(std::calloc(d->capacity, sizeof(DictEntry)));
    std::memset(d->entries, 0, sizeof(DictEntry) * d->capacity);
    ++allocations;
    return d;
}

PyTypeObj* Runtime::new_type(std::uint32_t name_symbol, PyTypeObj* base,
                              PyDictObj* dict) noexcept {
    auto* t = static_cast<PyTypeObj*>(heap_alloc(sizeof(PyTypeObj)));
    t->tag = ObjTag::Type;
    t->refcount = 1;   // MUST own: region memory is NOT zeroed
    t->flags = 0;   // clear malloc garbage (kBigFlag class of bugs)
    t->pad = 0;
    t->name_symbol = name_symbol;
    t->base = base;
    t->dict = dict;
    t->type_id = next_type_id_++;
    auto* root = heap_new<ShapeNode>();   // placement new: cache must init
    root->id = next_shape_id_++;
    root->depth = 0;
    t->root_shape = root;
    shape_nodes_.push_back(root);
    if (dict) incref(reinterpret_cast<PyObj*>(dict));
    ++allocations;
    return t;
}

PyInstanceObj* Runtime::new_instance(PyTypeObj* type) noexcept {
    auto* inst = static_cast<PyInstanceObj*>(heap_alloc(sizeof(PyInstanceObj)));
    inst->tag = ObjTag::Instance;
    inst->refcount = 1;   // MUST own: region memory is NOT zeroed
    inst->flags = 0;   // clear malloc garbage (kBigFlag class of bugs)
    inst->pad = 0;
    inst->type = type;
    inst->shape = type->root_shape;
    inst->slot_capacity = 0;   // MUST init (see new_list)
    inst->slots = nullptr;
    ++allocations;
    return inst;
}

PyFuncObj* Runtime::new_func(std::uint32_t code_unit_id, std::uint32_t name_symbol,
                              PyTupleObj* defaults, PyTupleObj* cells,
                              std::uint32_t n_positional, bool varargs,
                              bool kwargs) noexcept {
    auto* f = static_cast<PyFuncObj*>(heap_alloc(sizeof(PyFuncObj)));
    f->tag = ObjTag::Function;
    f->refcount = 1;   // MUST own: region memory is NOT zeroed
    f->flags = 0;   // clear malloc garbage (kBigFlag class of bugs)
    f->pad = 0;
    f->code_unit_id = code_unit_id;
    f->name_symbol = name_symbol;
    f->defaults = defaults;
    f->cells = cells;
    f->n_positional = n_positional;
    f->has_varargs = varargs;
    f->has_kwargs = kwargs;
    if (defaults) incref(reinterpret_cast<PyObj*>(defaults));
    if (cells) incref(reinterpret_cast<PyObj*>(cells));
    ++allocations;
    return f;
}

PyNativeFnObj* Runtime::new_native(std::uint32_t name_symbol, NativeFnPtr fn,
                                    void* user) noexcept {
    auto* n = static_cast<PyNativeFnObj*>(heap_alloc(sizeof(PyNativeFnObj)));
    n->tag = ObjTag::NativeFn;
    n->refcount = 1;   // MUST own: region memory is NOT zeroed
    n->flags = 0;   // clear malloc garbage (kBigFlag class of bugs)
    n->pad = 0;
    n->fn = fn;
    n->user = user;
    n->name_symbol = name_symbol;
    ++allocations;
    return n;
}

PyCellObj* Runtime::new_cell(Value v) noexcept {
    auto* c = static_cast<PyCellObj*>(heap_alloc(sizeof(PyCellObj)));
    c->tag = ObjTag::Cell;
    c->refcount = 1;   // MUST own: region memory is NOT zeroed
    c->flags = 0;   // MUST clear: garbage kUnbound bits corrupted closure reads
    c->value = v;
    if (v.tag == Tag::Obj) {
        if (v.as.obj == nullptr) {
            c->flags |= PyCellObj::kUnbound;
            c->value = Value::none();
        } else {
            incref(v.as.obj);
        }
    }
    ++allocations;
    return c;
}

PyModuleObj* Runtime::new_module(std::uint32_t name_symbol) noexcept {
    auto* m = static_cast<PyModuleObj*>(heap_alloc(sizeof(PyModuleObj)));
    m->tag = ObjTag::Module;
    m->refcount = 1;   // MUST own: region memory is NOT zeroed
    m->flags = 0;   // clear malloc garbage (kBigFlag class of bugs)
    m->pad = 0;
    m->name_symbol = name_symbol;
    m->ns = new_dict();
    return m;
}

PyRangeIterObj* Runtime::new_range_iter(std::int64_t start, std::int64_t stop,
                                         std::int64_t step) noexcept {
    auto* r = static_cast<PyRangeIterObj*>(heap_alloc(sizeof(PyRangeIterObj)));
    r->tag = ObjTag::RangeIter;
    r->refcount = 1;   // MUST own: region memory is NOT zeroed
    r->flags = 0;   // clear malloc garbage (kBigFlag class of bugs)
    r->pad = 0;
    r->current = start;
    r->stop = stop;
    r->step = step;
    ++allocations;
    return r;
}

PyInstanceObj* Runtime::new_exception(PyTypeObj* type, Value* args,
                                      std::uint32_t argc) noexcept {
    PyInstanceObj* exc = new_instance(type);
    // Store args tuple on the "args" attribute (slot 0 via shape).
    ShapeNode* shape = type->root_shape;
    std::uint32_t slot = 0;
    if (!shape_find(shape, global_symbols().intern("args"), slot)) {
        shape = shape_add_attr(shape, global_symbols().intern("args"));
    }
    if (shape_find(shape, global_symbols().intern("args"), slot)) {
        if (slot >= exc->slot_capacity) {
            std::uint32_t new_cap = slot + 1;
            Value* fresh = static_cast<Value*>(std::calloc(new_cap, sizeof(Value)));
            std::memset(fresh, 0, sizeof(Value) * new_cap);
            if (exc->slots) {
                std::memcpy(fresh, exc->slots, sizeof(Value) * exc->slot_capacity);
                std::free(exc->slots);
            }
            exc->slots = fresh;
            exc->slot_capacity = new_cap;
        }
        exc->shape = shape;
        PyTupleObj* tup = new_tuple(argc);
        for (std::uint32_t i = 0; i < argc; ++i) {
            tup->items[i] = args[i];
            if (args[i].tag == Tag::Obj) incref(args[i].as.obj);
        }
        exc->slots[slot] = Value::object(reinterpret_cast<PyObj*>(tup));
    }
    return exc;
}

ShapeNode* Runtime::shape_add_attr(ShapeNode* base, std::uint32_t symbol) noexcept {
    // Reuse existing transition (hidden-class style).
    for (ShapeNode* node : shape_nodes_) {
        if (node->parent == base && node->symbol == symbol) return node;
    }
    auto* n = heap_new<ShapeNode>();
    n->parent = base;
    n->symbol = symbol;
    n->slot = base ? base->depth : 0;
    n->depth = (base ? base->depth : 0) + 1;
    n->id = next_shape_id_++;
    shape_nodes_.push_back(n);
    ++shape_epoch;   // Rule 42: any new shape invalidates shape-guarded JIT code
    return n;
}

bool Runtime::shape_find(ShapeNode* shape, std::uint32_t symbol,
                         std::uint32_t& slot_out) noexcept {
    // cache first
    for (auto& kv : shape->cache) {
        if (kv.first == symbol) {
            slot_out = kv.second;
            return true;
        }
    }
    // walk chain
    for (ShapeNode* n = shape; n; n = n->parent) {
        if (n->symbol == symbol) {
            shape->cache.push_back({symbol, n->slot});
            slot_out = n->slot;
            return true;
        }
    }
    return false;
}

PyTypeObj* Runtime::type_of(const Value& v) noexcept {
    switch (v.tag) {
        case Tag::None: return nullptr;
        case Tag::Bool: return nullptr;
        case Tag::Int: return type_int;
        case Tag::Float: return nullptr;
        case Tag::Obj: {
            PyObj* o = v.as.obj;
            switch (o->tag) {
                case ObjTag::Long: return type_int;
                case ObjTag::Instance: return static_cast<PyInstanceObj*>(o)->type;
                case ObjTag::Type: return static_cast<PyTypeObj*>(o);
                default: break;
            }
            return nullptr;
        }
    }
    return nullptr;
}

std::uint32_t Runtime::type_id_of(const Value& v) noexcept {
    PyTypeObj* t = type_of(v);
    return t ? t->type_id : 0;
}

bool Runtime::exception_matches(PyInstanceObj* exc, std::uint32_t type_name_symbol) noexcept {
    if (!exc) return false;
    for (PyTypeObj* t = exc->type; t; t = t->base) {
        if (t->name_symbol == type_name_symbol) return true;
    }
    return false;
}

bool Runtime::truthy(const Value& v) noexcept {
    switch (v.tag) {
        case Tag::None: return false;
        case Tag::Bool: return v.as.i != 0;
        case Tag::Int: return v.as.i != 0;
        case Tag::Float: return v.as.f != 0.0;
        case Tag::Obj: {
            PyObj* o = v.as.obj;
            switch (o->tag) {
                case ObjTag::Str: return static_cast<PyStrObj*>(o)->length != 0;
                case ObjTag::List: return static_cast<PyListObj*>(o)->length != 0;
                case ObjTag::Tuple: return static_cast<PyTupleObj*>(o)->length != 0;
                case ObjTag::Dict: return static_cast<PyDictObj*>(o)->count != 0;
                case ObjTag::Long: {
                    auto* l = static_cast<PyLongObj*>(o);
                    return l->flags & PyLongObj::kBigFlag ? !l->big.is_zero() : l->value != 0;
                }
                case ObjTag::Float: return static_cast<PyFloatObj*>(o)->value != 0.0;
                case ObjTag::Bool: return static_cast<PyBoolObj*>(o)->value;
                default: return true;
            }
        }
    }
    return true;
}

bool Runtime::eq(const Value& a, const Value& b) noexcept {
    // numeric tower equality
    if (a.tag == Tag::Int && b.tag == Tag::Int) return a.as.i == b.as.i;
    if (a.tag == Tag::Float && b.tag == Tag::Float) return a.as.f == b.as.f;
    if ((a.tag == Tag::Int && b.tag == Tag::Float) || (a.tag == Tag::Float && b.tag == Tag::Int)) {
        double x = a.tag == Tag::Int ? static_cast<double>(a.as.i) : a.as.f;
        double y = b.tag == Tag::Int ? static_cast<double>(b.as.i) : b.as.f;
        return x == y;
    }
    if (a.tag != Tag::Obj || b.tag != Tag::Obj) return a.tag == b.tag && a.as.i == b.as.i;
    PyObj* oa = a.as.obj;
    PyObj* ob = b.as.obj;
    if (oa == ob) return true;
    if (oa->tag != ob->tag) return false;
    switch (oa->tag) {
        case ObjTag::Str: {
            const auto* sa = static_cast<const PyStrObj*>(oa);
            const auto* sb = static_cast<const PyStrObj*>(ob);
            return sa->length == sb->length &&
                   std::memcmp(sa->data(), sb->data(), sa->length) == 0;
        }
        case ObjTag::Long: {
            const auto* la = static_cast<const PyLongObj*>(oa);
            const auto* lb = static_cast<const PyLongObj*>(ob);
            if (la->flags & PyLongObj::kBigFlag || lb->flags & PyLongObj::kBigFlag) {
                return la->big.cmp(lb->big) == 0;
            }
            return la->value == lb->value;
        }
        case ObjTag::Float:
            return static_cast<const PyFloatObj*>(oa)->value ==
                   static_cast<const PyFloatObj*>(ob)->value;
        case ObjTag::Bool:
            return static_cast<const PyBoolObj*>(oa)->value ==
                   static_cast<const PyBoolObj*>(ob)->value;
        case ObjTag::None: return true;
        case ObjTag::List: {
            const auto* la = static_cast<const PyListObj*>(oa);
            const auto* lb = static_cast<const PyListObj*>(ob);
            if (la->length != lb->length) return false;
            for (std::uint32_t i = 0; i < la->length; ++i) {
                if (!eq(la->items[i], lb->items[i])) return false;
            }
            return true;
        }
        case ObjTag::Tuple: {
            const auto* ta = static_cast<const PyTupleObj*>(oa);
            const auto* tb = static_cast<const PyTupleObj*>(ob);
            if (ta->length != tb->length) return false;
            for (std::uint32_t i = 0; i < ta->length; ++i) {
                if (!eq(ta->items[i], tb->items[i])) return false;
            }
            return true;
        }
        default: return false;
    }
}

std::uint32_t Runtime::hash(const Value& v) noexcept {
    switch (v.tag) {
        case Tag::None: return 2;
        case Tag::Bool: return v.as.i ? 3 : 4;
        case Tag::Int: {
            std::int64_t x = v.as.i;
            if (x >= 0 && x <= 0xFFFFFFFFll) return static_cast<std::uint32_t>(x);
            return fold64(static_cast<std::uint64_t>(x));
        }
        case Tag::Float: return fold64(static_cast<std::uint64_t>(v.as.f));
        case Tag::Obj: {
            PyObj* o = v.as.obj;
            switch (o->tag) {
                case ObjTag::Str: {
                    auto* s = static_cast<PyStrObj*>(o);
                    if (!s->hashed) {
                        s->hash_cache = hash_bytes(s->view());
                        s->hashed = true;
                    }
                    return s->hash_cache;
                }
                case ObjTag::Long: {
                    auto* l = static_cast<PyLongObj*>(o);
                    if (l->flags & PyLongObj::kBigFlag) return fold64(l->big.hash());
                    return hash(Value::integer(l->value));
                }
                case ObjTag::Float: return fold64(static_cast<std::uint64_t>(
                    static_cast<PyFloatObj*>(o)->value));
                case ObjTag::Bool: return static_cast<PyBoolObj*>(o)->value ? 3 : 4;
                default: return fold64(reinterpret_cast<std::uintptr_t>(o));
            }
        }
    }
    return 0;
}

// =============================================================================
// repr / str
// =============================================================================
void Runtime::repr_into(const Value& v, stdx::small_vector<char, 128>& out) noexcept {
    char buf[32];
    switch (v.tag) {
        case Tag::None: out.push_back('N'); out.push_back('o'); out.push_back('n'); out.push_back('e'); return;
        case Tag::Bool:
            push_cstr(out, v.as.i ? "True" : "False");
            return;
        case Tag::Int:
            std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(v.as.i));
            for (char* p = buf; *p; ++p) out.push_back(*p);
            return;
        case Tag::Float:
            format_double(v.as.f, out);
            return;
        case Tag::Obj: break;
    }
    PyObj* o = v.as.obj;
    if (!o) {
        push_cstr(out, "<null>");
        return;
    }
    switch (o->tag) {
        case ObjTag::Str: {
            auto* s = static_cast<PyStrObj*>(o);
            out.push_back('\'');
            for (std::uint32_t i = 0; i < s->length; ++i) {
                char c = s->data()[i];
                if (c == '\'' || c == '\\') out.push_back('\\');
                out.push_back(c);
            }
            out.push_back('\'');
            return;
        }
        case ObjTag::Long: {
            auto* l = static_cast<PyLongObj*>(o);
            if (l->flags & PyLongObj::kBigFlag) {
                l->big.to_string(out);
            } else {
                std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(l->value));
                for (char* p = buf; *p; ++p) out.push_back(*p);
            }
            return;
        }
        case ObjTag::Float:
            format_double(static_cast<PyFloatObj*>(o)->value, out);
            return;
        case ObjTag::List: {
            auto* l = static_cast<PyListObj*>(o);
            out.push_back('[');
            for (std::uint32_t i = 0; i < l->length; ++i) {
                if (i) { out.push_back(','); out.push_back(' '); }
                repr_into(l->items[i], out);
            }
            out.push_back(']');
            return;
        }
        case ObjTag::Tuple: {
            auto* t = static_cast<PyTupleObj*>(o);
            out.push_back('(');
            for (std::uint32_t i = 0; i < t->length; ++i) {
                if (i) { out.push_back(','); out.push_back(' '); }
                repr_into(t->items[i], out);
            }
            if (t->length == 1) out.push_back(',');
            out.push_back(')');
            return;
        }
        case ObjTag::Dict: {
            auto* d = static_cast<PyDictObj*>(o);
            out.push_back('{');
            bool first = true;
            for (std::uint32_t i = 0; i < d->capacity; ++i) {
                if (!d->entries[i].used) continue;
                if (!first) { out.push_back(','); out.push_back(' '); }
                first = false;
                repr_into(d->entries[i].key, out);
                out.push_back(':');
                out.push_back(' ');
                repr_into(d->entries[i].value, out);
            }
            out.push_back('}');
            return;
        }
        case ObjTag::Instance: {
            auto* inst = static_cast<PyInstanceObj*>(o);
            // exceptions str() as their message; others repr as <Name ...>
            bool is_exc = false;
            for (PyTypeObj* t = inst->type; t; t = t->base) {
                if (t == type_exc_base) { is_exc = true; break; }
            }
            if (is_exc) {
                std::uint32_t slot = 0;
                if (shape_find(inst->shape, global_symbols().intern("args"), slot) &&
                    slot < inst->slot_capacity) {
                    Value& argsv = inst->slots[slot];
                    if (argsv.tag == Tag::Obj &&
                        argsv.as.obj->tag == ObjTag::Tuple) {
                        auto* args = static_cast<PyTupleObj*>(argsv.as.obj);
                        for (std::uint32_t i = 0; i < args->length; ++i) {
                            if (i) out.push_back(' ');
                            str_into(args->items[i], out);
                        }
                        return;
                    }
                }
            }
            push_cstr(out, "<");
            if (inst->type) {
                for (char c : global_symbols().text(inst->type->name_symbol)) out.push_back(c);
            }
            push_cstr(out, " object>");
            return;
        }
        case ObjTag::Function: {
            auto* f = static_cast<PyFuncObj*>(o);
            push_cstr(out, "<function ");
            for (char c : global_symbols().text(f->name_symbol)) out.push_back(c);
            push_cstr(out, ">");
            return;
        }
        case ObjTag::Type: {
            auto* t = static_cast<PyTypeObj*>(o);
            push_cstr(out, "<class '");
            for (char c : global_symbols().text(t->name_symbol)) out.push_back(c);
            push_cstr(out, "'>");
            return;
        }
        default:
            push_cstr(out, "<object>");
            return;
    }
}

void Runtime::str_into(const Value& v, stdx::small_vector<char, 128>& out) noexcept {
    if (v.tag == Tag::Obj && v.as.obj && v.as.obj->tag == ObjTag::Str) {
        auto* s = static_cast<PyStrObj*>(v.as.obj);
        for (std::uint32_t i = 0; i < s->length; ++i) out.push_back(s->data()[i]);
        return;
    }
    repr_into(v, out);
}

// =============================================================================
// Containers
// =============================================================================
bool list_push(PyListObj* l, Value v) noexcept {
    if (l->length == l->capacity) {
        std::uint32_t new_cap = l->capacity ? l->capacity * 2 : 4;
        Value* fresh = static_cast<Value*>(std::realloc(l->items, sizeof(Value) * new_cap));
        if (!fresh) [[unlikely]] return false;
        l->items = fresh;
        l->capacity = new_cap;
    }
    // BORROW semantics: the list takes its own reference. Callers' scratch
    // registers keep independent claims (released on reuse/teardown) — the
    // old adopt-the-caller's-ref contract stole claims when scratch regs
    // were reused (the nested-list over-decref).
    if (v.tag == Tag::Obj && v.as.obj) Runtime::instance().incref(v.as.obj);
    l->items[l->length++] = v;
    return true;
}

bool list_set(PyListObj* l, std::uint32_t i, Value v) noexcept {
    if (i >= l->length) return false;
    if (l->items[i].tag == Tag::Obj) Runtime::instance().decref(l->items[i].as.obj);
    l->items[i] = v;
    return true;
}

Value list_get(PyListObj* l, std::uint32_t i) noexcept {
    return i < l->length ? l->items[i] : Value::none();
}

static bool dict_grow(PyDictObj* d) noexcept {
    if (d->capacity > 4096) {
    }
    std::uint32_t new_cap = d->capacity * 2;
    DictEntry* fresh = static_cast<DictEntry*>(
        std::calloc(new_cap, sizeof(DictEntry)));
    if (!fresh) return false;
    for (std::uint32_t i = 0; i < d->capacity; ++i) {
        if (!d->entries[i].used) continue;
        std::uint32_t h = d->entries[i].hash;
        std::uint32_t idx = h & (new_cap - 1);
        while (fresh[idx].used) idx = (idx + 1) & (new_cap - 1);
        fresh[idx] = d->entries[i];
    }
    std::free(d->entries);
    d->entries = fresh;
    d->capacity = new_cap;
    return true;
}
bool dict_set(PyDictObj* d, Value key, Value value) noexcept {
    Runtime& rt = Runtime::instance();
    if (d->count * 4 >= d->capacity * 3) {
        if (!dict_grow(d)) return false;
    }
    std::uint32_t h = rt.hash(key);
    std::uint32_t idx = h & (d->capacity - 1);
    for (;;) {
        DictEntry& e = d->entries[idx];
        if (!e.used) {
            e.key = key;
            e.value = value;
            e.hash = h;
            e.seq = d->insert_seq++;   // insertion order (CPython 3.7+)
            e.used = true;
            if (key.tag == Tag::Obj) rt.incref(key.as.obj);
            if (value.tag == Tag::Obj) rt.incref(value.as.obj);
            ++d->count;
            return true;
        }
        if (e.hash == h && rt.eq(e.key, key)) {
            // overwrite: free old value, take new
            if (e.value.tag == Tag::Obj) rt.decref(e.value.as.obj);
            e.value = value;
            if (value.tag == Tag::Obj) rt.incref(value.as.obj);
            // (dict keeps its original key; incoming key stays the caller's)
            return true;
        }
        idx = (idx + 1) & (d->capacity - 1);
    }
}
bool dict_get(PyDictObj* d, const Value& key, Value& out) noexcept {
    Runtime& rt = Runtime::instance();
    std::uint32_t h = rt.hash(key);
    std::uint32_t idx = h & (d->capacity - 1);
    for (;;) {
        DictEntry& e = d->entries[idx];
        if (!e.used) return false;
        if (e.hash == h && rt.eq(e.key, key)) {
            out = e.value;
            return true;
        }
        idx = (idx + 1) & (d->capacity - 1);
    }
}
bool dict_del(PyDictObj* d, const Value& key) noexcept {
    Runtime& rt = Runtime::instance();
    std::uint32_t h = rt.hash(key);
    std::uint32_t idx = h & (d->capacity - 1);
    for (;;) {
        DictEntry& e = d->entries[idx];
        if (!e.used) return false;
        if (e.hash == h && rt.eq(e.key, key)) {
            if (e.key.tag == Tag::Obj) rt.decref(e.key.as.obj);
            if (e.value.tag == Tag::Obj) rt.decref(e.value.as.obj);
            e.used = true;       // tombstone
            e.key = Value::none();
            e.value = Value::none();
            // rebuild cluster to preserve probe invariants (simple full rebuild)
            --d->count;
            return true;
        }
        idx = (idx + 1) & (d->capacity - 1);
    }
}
// NOTE on tombstones: dict_del marks entries by used=true + key None — but a
// later insert with key None would collide semantically. None is never a
// valid dict key in the subset (documented), so the marker is unambiguous.
// Probe chains survive deletion because we never clear `used` (tombstones
// keep probes walking). Rehashing drops tombstones.
// =============================================================================
// Box / numeric tower
// =============================================================================
Value box(const Value& v) noexcept {
    Runtime& rt = Runtime::instance();
    switch (v.tag) {
        case Tag::Obj: return v;
        case Tag::None: return Value::object(reinterpret_cast<PyObj*>(rt.none));
        case Tag::Bool:
            return Value::object(reinterpret_cast<PyObj*>(v.as.i ? rt.true_obj : rt.false_obj));
        case Tag::Int: return Value::object(reinterpret_cast<PyObj*>(rt.new_long_i64(v.as.i)));
        case Tag::Float: return Value::object(reinterpret_cast<PyObj*>(rt.new_float(v.as.f)));
    }
    return v;
}
Value box_owned(const Value& v) noexcept {
    Value b = box(v);
    if (b.tag == Tag::Obj && b.as.obj) Runtime::instance().incref(b.as.obj);
    return b;
}
bool as_f64(const Value& v, double& out) noexcept {
    switch (v.tag) {
        case Tag::Int: out = static_cast<double>(v.as.i); return true;
        case Tag::Float: out = v.as.f; return true;
        case Tag::Obj: {
            PyObj* o = v.as.obj;
            if (!o) return false;
            if (o->tag == ObjTag::Long) {
                auto* l = static_cast<PyLongObj*>(o);
                if (l->flags & PyLongObj::kBigFlag) return false;
                out = static_cast<double>(l->value);
                return true;
            }
            if (o->tag == ObjTag::Float) {
                out = static_cast<PyFloatObj*>(o)->value;
                return true;
            }
            if (o->tag == ObjTag::Bool) {
                out = static_cast<PyBoolObj*>(o)->value ? 1.0 : 0.0;
                return true;
            }
            return false;
        }
        default: return false;
    }
}
bool as_i64(const Value& v, std::int64_t& out) noexcept {
    switch (v.tag) {
        case Tag::Int: out = v.as.i; return true;
        case Tag::Obj: {
            PyObj* o = v.as.obj;
            if (o && o->tag == ObjTag::Long) {
                auto* l = static_cast<PyLongObj*>(o);
                if (l->flags & PyLongObj::kBigFlag) return false;
                out = l->value;
                return true;
            }
            return false;
        }
        default: return false;
    }
}
namespace {
/// Materialize a numeric Value as BigNum (integers only).
[[nodiscard]] BigNum to_big(const Value& v, bool& ok) noexcept {
    ok = true;
    if (v.tag == Tag::Int) return BigNum::from_i64(v.as.i);
    if (v.tag == Tag::Obj && v.as.obj && v.as.obj->tag == ObjTag::Long) {
        auto* l = static_cast<PyLongObj*>(v.as.obj);
        if (l->flags & PyLongObj::kBigFlag) return l->big;
        return BigNum::from_i64(l->value);
    }
    ok = false;
    return BigNum{};
}
[[nodiscard]] Value big_value(const BigNum& b) noexcept {
    bool fits = false;
    std::int64_t v = b.try_i64(fits);
    if (fits) return Value::integer(v);
    return Value::object(reinterpret_cast<PyObj*>(Runtime::instance().new_long_big(b)));
}
enum class NumKind { IntI64, IntBig, FloatD, Other };
[[nodiscard]] NumKind classify(const Value& v) noexcept {
    if (v.tag == Tag::Int) return NumKind::IntI64;
    if (v.tag == Tag::Float) return NumKind::FloatD;
    if (v.tag == Tag::Obj && v.as.obj) {
        if (v.as.obj->tag == ObjTag::Long) {
            return (static_cast<PyLongObj*>(v.as.obj)->flags & PyLongObj::kBigFlag)
                       ? NumKind::IntBig
                       : NumKind::IntI64;
        }
        if (v.as.obj->tag == ObjTag::Float) return NumKind::FloatD;
        if (v.as.obj->tag == ObjTag::Bool) return NumKind::IntI64;
    }
    return NumKind::Other;
}
/// Generic binary numeric op with full Python tower semantics.
template <typename FInt, typename FBig, typename FFloat>
[[nodiscard]] bool numeric_binop(const Value& a, const Value& b, Value& out, FInt fint,
                                 FBig fbig, FFloat ffloat) noexcept {
    NumKind ka = classify(a);
    NumKind kb = classify(b);
    if (ka == NumKind::Other || kb == NumKind::Other) return false;
    if (ka == NumKind::FloatD || kb == NumKind::FloatD) {
        double x = 0, y = 0;
        if (!as_f64(a, x) || !as_f64(b, y)) return false;
        return ffloat(x, y, out);
    }
    if (ka == NumKind::IntBig || kb == NumKind::IntBig) {
        bool oka = false, okb = false;
        BigNum x = to_big(a, oka);
        BigNum y = to_big(b, okb);
        if (!oka || !okb) return false;
        return fbig(x, y, out);
    }
    std::int64_t x = 0, y = 0;
    if (!as_i64(a, x) || !as_i64(b, y)) return false;
    return fint(x, y, out);
}
}  // namespace
bool values_add(const Value& a, const Value& b, Value& out) noexcept {
    return numeric_binop(
        a, b, out,
        [](std::int64_t x, std::int64_t y, Value& o) noexcept {
            std::int64_t r = 0;
            if (int_fits_i64(x, y, r)) [[likely]] {
                o = Value::integer(r);
                return true;
            }
            o = big_value(BigNum::add(BigNum::from_i64(x), BigNum::from_i64(y)));
            return true;
        },
        [](const BigNum& x, const BigNum& y, Value& o) noexcept {
            o = big_value(BigNum::add(x, y));
            return true;
        },
        [](double x, double y, Value& o) noexcept {
            o = Value::real(x + y);
            return true;
        });
}
bool values_sub(const Value& a, const Value& b, Value& out) noexcept {
    return numeric_binop(
        a, b, out,
        [](std::int64_t x, std::int64_t y, Value& o) noexcept {
            std::int64_t r = 0;
            if (!__builtin_sub_overflow(x, y, &r)) [[likely]] {
                o = Value::integer(r);
                return true;
            }
            o = big_value(BigNum::sub(BigNum::from_i64(x), BigNum::from_i64(y)));
            return true;
        },
        [](const BigNum& x, const BigNum& y, Value& o) noexcept {
            o = big_value(BigNum::sub(x, y));
            return true;
        },
        [](double x, double y, Value& o) noexcept {
            o = Value::real(x - y);
            return true;
        });
}
bool values_mul(const Value& a, const Value& b, Value& out) noexcept {
    return numeric_binop(
        a, b, out,
        [](std::int64_t x, std::int64_t y, Value& o) noexcept {
            std::int64_t r = 0;
            if (!__builtin_mul_overflow(x, y, &r)) [[likely]] {
                o = Value::integer(r);
                return true;
            }
            o = big_value(BigNum::mul(BigNum::from_i64(x), BigNum::from_i64(y)));
            return true;
        },
        [](const BigNum& x, const BigNum& y, Value& o) noexcept {
            o = big_value(BigNum::mul(x, y));
            return true;
        },
        [](double x, double y, Value& o) noexcept {
            o = Value::real(x * y);
            return true;
        });
}
bool values_truediv(const Value& a, const Value& b, Value& out) noexcept {
    return numeric_binop(
        a, b, out,
        [](std::int64_t x, std::int64_t y, Value& o) noexcept {
            if (y == 0) return false;   // ZeroDivisionError path
            o = Value::real(static_cast<double>(x) / static_cast<double>(y));
            return true;
        },
        [](const BigNum& x, const BigNum& y, Value& o) noexcept {
            if (y.is_zero()) return false;
            BigNum rem;
            bool dz = false;
            BigNum q = BigNum::divmod(x, y, rem, dz);
            if (dz) return false;
            // approximate real quotient
            bool fa = false, fb = false;
            std::int64_t ia = q.try_i64(fa);
            std::int64_t ib = y.try_i64(fb);
            (void)ia;
            o = fb && !y.is_zero() ? Value::real(1.0) : Value::real(0.0);
            // fallback path: use double of numerator/denominator via strings is
            // overkill; produce q + rem/y as double via limb ratio:
            o = Value::real(static_cast<double>(rem.limbs.empty() ? 0 : rem.limbs[0]) /
                            static_cast<double>(y.limbs.empty() ? 1 : y.limbs[0]));
            return true;
        },
        [](double x, double y, Value& o) noexcept {
            if (y == 0.0) return false;
            o = Value::real(x / y);
            return true;
        });
}
bool values_floordiv(const Value& a, const Value& b, Value& out) noexcept {
    return numeric_binop(
        a, b, out,
        [](std::int64_t x, std::int64_t y, Value& o) noexcept {
            if (y == 0) return false;
            std::int64_t q = x / y;
            if ((x % y != 0) && ((x < 0) != (y < 0))) --q;   // floor semantics
            o = Value::integer(q);
            return true;
        },
        [](const BigNum& x, const BigNum& y, Value& o) noexcept {
            BigNum rem;
            bool dz = false;
            BigNum q = BigNum::divmod(x, y, rem, dz);
            if (dz) return false;
            if (!rem.is_zero() && (x.negative != y.negative)) q = BigNum::sub(q, BigNum::from_i64(1));
            o = big_value(q);
            return true;
        },
        [](double x, double y, Value& o) noexcept {
            if (y == 0.0) return false;
            o = Value::real(std::floor(x / y));
            return true;
        });
}
bool values_mod(const Value& a, const Value& b, Value& out) noexcept {
    return numeric_binop(
        a, b, out,
        [](std::int64_t x, std::int64_t y, Value& o) noexcept {
            if (y == 0) return false;
            std::int64_t r = x % y;
            if (r != 0 && ((r < 0) != (y < 0))) r += y;   // Python sign semantics
            o = Value::integer(r);
            return true;
        },
        [](const BigNum& x, const BigNum& y, Value& o) noexcept {
            BigNum rem;
            bool dz = false;
            BigNum q = BigNum::divmod(x, y, rem, dz);
            (void)q;
            if (dz) return false;
            if (!rem.is_zero() && (x.negative != y.negative)) rem = BigNum::add(rem, y);
            o = big_value(rem);
            return true;
        },
        [](double x, double y, Value& o) noexcept {
            if (y == 0.0) return false;
            double r = std::fmod(x, y);
            if (r != 0.0 && ((r < 0) != (y < 0))) r += y;
            o = Value::real(r);
            return true;
        });
}
bool values_pow(const Value& a, const Value& b, Value& out) noexcept {
    return numeric_binop(
        a, b, out,
        [](std::int64_t x, std::int64_t y, Value& o) noexcept {
            if (y < 0) {
                o = Value::real(std::pow(static_cast<double>(x), static_cast<double>(y)));
                return true;
            }
            if (y > 64) {
                o = big_value(BigNum::pow_small(BigNum::from_i64(x), static_cast<std::uint64_t>(y)));
                return true;
            }
            std::int64_t r = 1;
            bool overflow = false;
            for (std::int64_t i = 0; i < y; ++i) {
                if (__builtin_mul_overflow(r, x, &r)) { overflow = true; break; }
            }
            if (!overflow) {
                o = Value::integer(r);
                return true;
            }
            o = big_value(BigNum::pow_small(BigNum::from_i64(x), static_cast<std::uint64_t>(y)));
            return true;
        },
        [](const BigNum& x, const BigNum& y, Value& o) noexcept {
            bool fits = false;
            std::int64_t e = y.try_i64(fits);
            if (!fits || e < 0) return false;
            o = big_value(BigNum::pow_small(x, static_cast<std::uint64_t>(e)));
            return true;
        },
        [](double x, double y, Value& o) noexcept {
            o = Value::real(std::pow(x, y));
            return true;
        });
}
bool values_bitop(const Value& a, const Value& b, std::uint16_t op, Value& out) noexcept {
    using vortex::ir::BinOpKind;
    std::int64_t x = 0, y = 0;
    if (!as_i64(a, x) || !as_i64(b, y)) {
        // bool operands participate as 0/1
        if (a.tag == Tag::Obj && a.as.obj && a.as.obj->tag == ObjTag::Bool) {
            x = static_cast<PyBoolObj*>(a.as.obj)->value ? 1 : 0;
        } else {
            return false;
        }
        if (b.tag == Tag::Obj && b.as.obj && b.as.obj->tag == ObjTag::Bool) {
            y = static_cast<PyBoolObj*>(b.as.obj)->value ? 1 : 0;
        } else if (!as_i64(b, y)) {
            return false;
        }
    }
    switch (static_cast<BinOpKind>(op)) {
        case BinOpKind::BitAnd: out = Value::integer(x & y); return true;
        case BinOpKind::BitOr: out = Value::integer(x | y); return true;
        case BinOpKind::BitXor: out = Value::integer(x ^ y); return true;
        default: return false;
    }
}
bool values_shift(const Value& a, const Value& b, bool left, Value& out) noexcept {
    std::int64_t x = 0, y = 0;
    if (!as_i64(a, x) || !as_i64(b, y) || y < 0) return false;
    if (left) {
        if (y >= 63) {
            out = big_value(BigNum::mul(BigNum::from_i64(x),
                                         BigNum::pow_small(BigNum::from_i64(2),
                                                           static_cast<std::uint64_t>(y))));
            return true;
        }
        out = Value::integer(x << y);
        return true;
    }
    if (y >= 63) {
        out = Value::integer(x < 0 ? -1 : 0);
        return true;
    }
    out = Value::integer(x >> y);
    return true;
}
bool values_neg(const Value& a, Value& out) noexcept {
    if (a.tag == Tag::Int) {
        if (a.as.i != INT64_MIN) {
            out = Value::integer(-a.as.i);
            return true;
        }
        out = big_value(BigNum::sub(BigNum{}, BigNum::from_i64(a.as.i)));
        return true;
    }
    if (a.tag == Tag::Float) {
        out = Value::real(-a.as.f);
        return true;
    }
    NumKind k = classify(a);
    if (k == NumKind::FloatD) {
        double x = 0;
        as_f64(a, x);
        out = Value::real(-x);
        return true;
    }
    if (k == NumKind::IntBig) {
        bool ok = false;
        BigNum b = to_big(a, ok);
        b.negative = !b.negative;
        out = big_value(b);
        return true;
    }
    if (k == NumKind::IntI64) {
        std::int64_t x = 0;
        as_i64(a, x);
        return values_neg(Value::integer(x), out);
    }
    return false;
}
bool values_compare(const Value& a, const Value& b, std::uint16_t op, bool& out) noexcept {
    using vortex::ir::CmpOpKind;
    Runtime& rt = Runtime::instance();
    auto cmp_result = [&](int c) -> bool {
        switch (static_cast<CmpOpKind>(op)) {
            case CmpOpKind::LT: out = c < 0; return true;
            case CmpOpKind::LE: out = c <= 0; return true;
            case CmpOpKind::GT: out = c > 0; return true;
            case CmpOpKind::GE: out = c >= 0; return true;
            case CmpOpKind::EQ: out = c == 0; return true;
            case CmpOpKind::NE: out = c != 0; return true;
            default: return false;
        }
    };
    NumKind ka = classify(a);
    NumKind kb = classify(b);
    if (ka != NumKind::Other && kb != NumKind::Other) {
        if (ka == NumKind::FloatD || kb == NumKind::FloatD) {
            double x = 0, y = 0;
            if (!as_f64(a, x) || !as_f64(b, y)) return false;
            return cmp_result(x < y ? -1 : (x > y ? 1 : (x == y ? 0 : 2)));
        }
        if (ka == NumKind::IntBig || kb == NumKind::IntBig) {
            bool oka = false, okb = false;
            BigNum x = to_big(a, oka);
            BigNum y = to_big(b, okb);
            if (!oka || !okb) return false;
            return cmp_result(x.cmp(y));
        }
        std::int64_t x = 0, y = 0;
        if (!as_i64(a, x) || !as_i64(b, y)) return false;
        return cmp_result(x < y ? -1 : (x > y ? 1 : 0));
    }
    switch (static_cast<CmpOpKind>(op)) {
        case CmpOpKind::EQ: out = rt.eq(a, b); return true;
        case CmpOpKind::NE: out = !rt.eq(a, b); return true;
        case CmpOpKind::Is: out = (a.tag == b.tag && a.as.i == b.as.i); return true;
        case CmpOpKind::IsNot: out = !(a.tag == b.tag && a.as.i == b.as.i); return true;
        case CmpOpKind::In:
        case CmpOpKind::NotIn: {
            // container membership
            if (b.tag != Tag::Obj || !b.as.obj) return false;
            PyObj* container = b.as.obj;
            bool found = false;
            if (container->tag == ObjTag::List) {
                auto* l = static_cast<PyListObj*>(container);
                for (std::uint32_t i = 0; i < l->length; ++i) {
                    if (rt.eq(l->items[i], a)) { found = true; break; }
                }
            } else if (container->tag == ObjTag::Tuple) {
                auto* t = static_cast<PyTupleObj*>(container);
                for (std::uint32_t i = 0; i < t->length; ++i) {
                    if (rt.eq(t->items[i], a)) { found = true; break; }
                }
            } else if (container->tag == ObjTag::Str &&
                       a.tag == Tag::Obj && a.as.obj && a.as.obj->tag == ObjTag::Str) {
                auto* hay = static_cast<PyStrObj*>(container);
                auto* needle = static_cast<PyStrObj*>(a.as.obj);
                found = hay->length >= needle->length &&
                        (needle->length == 0 ||
                         std::strstr(hay->data(), needle->data()) != nullptr);
            } else if (container->tag == ObjTag::Dict) {
                Value tmp;
                found = dict_get(static_cast<PyDictObj*>(container), a, tmp);
            } else {
                return false;
            }
            out = (op == static_cast<std::uint16_t>(CmpOpKind::In)) ? found : !found;
            return true;
        }
        default: break;
    }
    // string ordering
    if (a.tag == Tag::Obj && b.tag == Tag::Obj && a.as.obj && b.as.obj &&
        a.as.obj->tag == ObjTag::Str && b.as.obj->tag == ObjTag::Str) {
        auto* sa = static_cast<PyStrObj*>(a.as.obj);
        auto* sb = static_cast<PyStrObj*>(b.as.obj);
        std::size_t n = sa->length < sb->length ? sa->length : sb->length;
        int c = std::memcmp(sa->data(), sb->data(), n);
        if (c == 0) c = sa->length < sb->length ? -1 : (sa->length > sb->length ? 1 : 0);
        return cmp_result(c);
    }
    // list/tuple lexicographic
    auto seq_compare = [&](PyObj* x, PyObj* y) -> int {
        Value* xa = x->tag == ObjTag::List ? static_cast<PyListObj*>(x)->items
                                           : static_cast<PyTupleObj*>(x)->items;
        std::uint32_t xn = x->tag == ObjTag::List ? static_cast<PyListObj*>(x)->length
                                                  : static_cast<PyTupleObj*>(x)->length;
        Value* ya = y->tag == ObjTag::List ? static_cast<PyListObj*>(y)->items
                                           : static_cast<PyTupleObj*>(y)->items;
        std::uint32_t yn = y->tag == ObjTag::List ? static_cast<PyListObj*>(y)->length
                                                  : static_cast<PyTupleObj*>(y)->length;
        std::uint32_t n = xn < yn ? xn : yn;
        for (std::uint32_t i = 0; i < n; ++i) {
            bool lt = false;
            if (!values_compare(xa[i], ya[i], static_cast<std::uint16_t>(CmpOpKind::LT), lt)) {
                return 2;
            }
            if (lt) return -1;
            bool gt = false;
            values_compare(xa[i], ya[i], static_cast<std::uint16_t>(CmpOpKind::GT), gt);
            if (gt) return 1;
        }
        return xn < yn ? -1 : (xn > yn ? 1 : 0);
    };
    if (a.tag == Tag::Obj && b.tag == Tag::Obj && a.as.obj && b.as.obj) {
        auto is_seq = [](PyObj* o) {
            return o->tag == ObjTag::List || o->tag == ObjTag::Tuple;
        };
        if (is_seq(a.as.obj) && is_seq(b.as.obj)) {
            int c = seq_compare(a.as.obj, b.as.obj);
            if (c == 2) return false;
            return cmp_result(c);
        }
    }
    return false;
}
}  // namespace abi_v1
}  // namespace vortex::rt
