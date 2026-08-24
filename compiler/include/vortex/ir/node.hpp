// =============================================================================
// vortex/ir/node.hpp — IR node structure (Rules 15, 19, 32, 33)
//
// Purpose:
//   The Sea-of-Nodes node. All edges are NodeId (uint32_t) indices — never
//   raw pointers (Rule 15). Inputs use small_vector SBO (Rule 19). Orthogonal
//   boolean state is a Flags<NodeFlag> bitmask (Rule 32).
//
// Layout discipline (Rule 20):
//   Node stays under 128 bytes; bulk passes that only need {kind, flags} or
//   {kind, uses_count} extract them into SoA arrays via Graph::gather_kinds().
//
// Input conventions per class (checked by the Verifier, Rule 40):
//   Control nodes: ins[0] is control input (Region/Loop: 2..N + LoopProj).
//   Memory nodes:  ins[0]=control, ins[1]=effect-in, ins[2..]=data.
//   Pure data:     ins are data only.
//   Guard:         ins[0]=condition, ins[1]=frame-state node.
// =============================================================================

#pragma once

#include <cstdint>

#include "vortex/common/value.hpp"
#include "vortex/ir/node_kind.hpp"
#include "vortex/stdx/small_vector.hpp"
#include "vortex/support/flags.hpp"

namespace vortex::ir {

inline namespace abi_v1 {

using NodeId = std::uint32_t;
inline constexpr NodeId invalid_node = 0xFFFF'FFFF;

enum class NodeFlag : std::uint32_t {
    None = 0,
    Pure = 1u << 0,           // no effects, depends only on inputs
    MayThrow = 1u << 1,       // can raise a Python exception
    MayCall = 1u << 2,        // can execute arbitrary Python (call/__add__)
    Speculative = 1u << 3,    // requires FrameState (Rule 5)
    Pinned = 1u << 4,         // cannot float across its control input
    OnEffectChain = 1u << 5,  // serializes with memory ops
    Dead = 1u << 6,           // removed by DCE / unreachable
    Unboxed = 1u << 7,        // value is a native scalar (post-unboxing)
    Vectorizable = 1u << 8,   // candidate for SLP packing (Pass 31a marks)
    ShapeGuarded = 1u << 9,   // guarded by a shape check
    TypeGuarded = 1u << 10,   // guarded by a type check
    NoGIL = 1u << 11,         // pure native region (Pass 38)
    Cold = 1u << 12,          // placed in cold partition (Pass 55)
    Hot = 1u << 13,           // PGO says hot
    Escapes = 1u << 14,       // escape analysis result
    RegionAlloc = 1u << 15,   // allocated in bump region (Pass 42)
};

using NodeFlags = Flags<NodeFlag>;

/// Sub-op codes stored in Node::subop for parameterized kinds.
enum class BinOpKind : std::uint8_t {
    Add, Sub, Mul, TrueDiv, FloorDiv, Mod, Pow,
    LShift, RShift, BitAnd, BitOr, BitXor, MatMul,
};
enum class CmpOpKind : std::uint8_t {
    LT, LE, GT, GE, EQ, NE, Is, IsNot, In, NotIn,
};
enum class ConvKind : std::uint8_t {
    ToInt64, ToFloat64, ToBool, ToObject,
};
enum class GuardKind : std::uint8_t {
    TypeIs,          // value's runtime type id equals expected
    ShapeIs,         // object's shape id equals expected
    IntFits,         // unboxed int still within int64 (no bignum promotion)
    NotNone,         // value is not None (Pass 37)
    Bounds,          // 0 <= idx < bound (Pass 36)
    NoOverflow,      // arithmetic did not overflow
    AliasDisjoint,   // two base pointers do not alias (Pass 49 / 31c)
    ModuleVersion,   // module version counter unchanged
    MonomorphicCall, // call site still monomorphic (Pass 20)
};

enum class UnboxType : std::uint8_t {
    Int64, Float64, Bool,
};

struct Node {
    NodeKind kind{NodeKind::Unreachable};
    std::uint16_t subop{0};          // BinOpKind/CmpOpKind/ConvKind/GuardKind/...
    NodeFlags flags{};
    NodeId id{invalid_node};

    stdx::small_vector<NodeId, 4> ins{};   // inputs (see file-header conventions)

    // --- payload (per-kind interpretation; Verifier validates consistency) ---
    Value const_value{};             // ConstInt/ConstFloat/ConstPy payload
    std::uint32_t symbol{0xFFFF'FFFF};  // SymbolId: names, attribute keys
    std::uint32_t shape_id{0};       // shape/type expectation (guards, fields)
    std::uint32_t aux0{0};           // arity / offset / lane width / param index
    std::uint32_t aux1{0};           // frame-state index / secondary payload
    std::uint64_t pgo_count{0};      // PGO execution count (hot passes read)

    // --- analysis outputs (overwritten per pass run; not part of identity) ---
    std::uint32_t type_tag{0};       // Ty index — see ty.hpp
    std::uint32_t dom_depth{0};      // dominator depth (analysis cache)
    std::int32_t loop_depth{-1};     // -1 = not in a loop
    std::uint32_t use_count{0};      // maintained by Graph

    [[nodiscard]] bool has(NodeFlag f) const noexcept { return flags.has(f); }
    void set_flag(NodeFlag f) noexcept { flags.set(f); }
    void clear_flag(NodeFlag f) noexcept { flags.clear(f); }

    /// Value-numbering key inputs for GVN: kind+subop+payload+inputs.
    [[nodiscard]] bool structurally_equal(const Node& o) const noexcept {
        return kind == o.kind && subop == o.subop && const_value.tag == o.const_value.tag &&
               const_value.as.i == o.const_value.as.i && symbol == o.symbol &&
               shape_id == o.shape_id && aux0 == o.aux0 && ins.size() == o.ins.size() &&
               std::memcmp(ins.data(), o.ins.data(), ins.size() * sizeof(NodeId)) == 0;
    }
};

static_assert(sizeof(Node) <= 160, "Node must stay compact (Rule 20 cache discipline)");

}  // namespace abi_v1
}  // namespace vortex::ir
