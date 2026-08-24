// =============================================================================
// vortex/ir/node_kind.cpp — canonical names for golden-format round-trips.
// =============================================================================

#include "vortex/ir/node_kind.hpp"

#include <cstring>

namespace vortex::ir {
inline namespace abi_v1 {

namespace {
struct KindName {
    NodeKind kind;
    const char* name;
};
constexpr KindName kNames[] = {
    {NodeKind::Start, "start"},
    {NodeKind::Region, "region"},
    {NodeKind::If, "if"},
    {NodeKind::IfTrue, "if_true"},
    {NodeKind::IfFalse, "if_false"},
    {NodeKind::Jump, "jump"},
    {NodeKind::Loop, "loop"},
    {NodeKind::LoopExit, "loop_exit"},
    {NodeKind::Return, "return"},
    {NodeKind::Throw, "throw"},
    {NodeKind::Unreachable, "unreachable"},
    {NodeKind::Catch, "catch"},
    {NodeKind::Parameter, "param"},
    {NodeKind::ConstInt, "const_int"},
    {NodeKind::ConstFloat, "const_float"},
    {NodeKind::ConstPy, "const_py"},
    {NodeKind::Phi, "phi"},
    {NodeKind::EffectPhi, "effect_phi"},
    {NodeKind::Add, "iadd"},
    {NodeKind::Sub, "isub"},
    {NodeKind::Mul, "imul"},
    {NodeKind::Div, "idiv"},
    {NodeKind::Mod, "imod"},
    {NodeKind::Pow, "ipow"},
    {NodeKind::Neg, "ineg"},
    {NodeKind::BitAnd, "iand"},
    {NodeKind::BitOr, "ior"},
    {NodeKind::BitXor, "ixor"},
    {NodeKind::Shl, "ishl"},
    {NodeKind::Shr, "ishr"},
    {NodeKind::CmpLT, "ilt"},
    {NodeKind::CmpLE, "ile"},
    {NodeKind::CmpGT, "igt"},
    {NodeKind::CmpGE, "ige"},
    {NodeKind::CmpEQ, "ieq"},
    {NodeKind::CmpNE, "ine"},
    {NodeKind::Not, "inot"},
    {NodeKind::SExt, "sext"},
    {NodeKind::ZExt, "zext"},
    {NodeKind::Trunc, "trunc"},
    {NodeKind::BitCast, "bitcast"},
    {NodeKind::I2F, "i2f"},
    {NodeKind::F2I, "f2i"},
    {NodeKind::Unbox, "unbox"},
    {NodeKind::Box, "box"},
    {NodeKind::PyBinary, "pybin"},
    {NodeKind::PyUnary, "pyun"},
    {NodeKind::PyCompare, "pycmp"},
    {NodeKind::CallPy, "call"},
    {NodeKind::CallDirect, "call_direct"},
    {NodeKind::GuardedDirectCall, "call_guarded"},
    {NodeKind::CallNative, "call_native"},
    {NodeKind::DispatchCache, "ic"},
    {NodeKind::LoadAttr, "load_attr"},
    {NodeKind::StoreAttr, "store_attr"},
    {NodeKind::LoadIndex, "load_index"},
    {NodeKind::StoreIndex, "store_index"},
    {NodeKind::LoadGlobal, "load_global"},
    {NodeKind::StoreGlobal, "store_global"},
    {NodeKind::Len, "len"},
    {NodeKind::Iter, "iter"},
    {NodeKind::IterNext, "iter_next"},
    {NodeKind::NewList, "new_list"},
    {NodeKind::NewDict, "new_dict"},
    {NodeKind::NewTuple, "new_tuple"},
    {NodeKind::NewObject, "new_object"},
    {NodeKind::ListAppend, "list_append"},
    {NodeKind::Yield, "yield"},
    {NodeKind::GetIterCheck, "getiter_check"},
    {NodeKind::Load, "load"},
    {NodeKind::Store, "store"},
    {NodeKind::LoadField, "load_field"},
    {NodeKind::StoreField, "store_field"},
    {NodeKind::Allocated, "allocated"},
    {NodeKind::RegionFree, "region_free"},
    {NodeKind::Altered, "altered"},
    {NodeKind::Guard, "guard"},
    {NodeKind::DeoptBarrier, "deopt_barrier"},
    {NodeKind::FrameStateNode, "frame_state"},
    {NodeKind::VecPack, "vec_pack"},
    {NodeKind::VecExtract, "vec_extract"},
    {NodeKind::VecLoad, "vec_load"},
    {NodeKind::VecStore, "vec_store"},
    {NodeKind::VecOp, "vec_op"},
    {NodeKind::Gather, "gather"},
    {NodeKind::Scatter, "scatter"},
    {NodeKind::VecShuffle, "vec_shuffle"},
    {NodeKind::VecReduce, "vec_reduce"},
    {NodeKind::TupleProj, "tuple_proj"},
    {NodeKind::CallResult, "call_result"},
    {NodeKind::DebugBreak, "debug_break"},
};
}  // namespace

const char* node_kind_name(NodeKind k) noexcept {
    for (const KindName& kn : kNames) {
        if (kn.kind == k) return kn.name;
    }
    return "<bad-kind>";
}

bool node_kind_from_name(const char* name, std::size_t len, NodeKind& out) noexcept {
    for (const KindName& kn : kNames) {
        if (std::strlen(kn.name) == len && std::memcmp(kn.name, name, len) == 0) {
            out = kn.kind;
            return true;
        }
    }
    return false;
}

}  // namespace abi_v1
}  // namespace vortex::ir
