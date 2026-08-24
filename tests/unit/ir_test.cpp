// =============================================================================
// tests/unit/ir_test.cpp — IR core tests: construction, printing round-trip,
// verifier, graph mutations (Rules 35, 40, 52).
// =============================================================================

#include "harness.hpp"
#include "vortex/ir/parser.hpp"
#include "vortex/ir/printer.hpp"
#include "vortex/ir/verifier.hpp"

#include <cstring>

using namespace vortex;
using namespace vortex::ir;

namespace {
Graph build_sample_graph() {
    Graph g;
    NodeId start = g.create(NodeKind::Start);
    g.set_start(start);

    NodeId p0 = g.create(NodeKind::Parameter, {start});
    g.node(p0).aux0 = 0;
    NodeId c1 = g.create(NodeKind::ConstInt);
    g.node(c1).const_value = Value::integer(1);
    NodeId add = g.create(NodeKind::Add, {p0, c1});
    NodeId ret = g.create(NodeKind::Return, {start, add});

    g.node(p0).set_flag(NodeFlag::Pure);
    g.node(c1).set_flag(NodeFlag::Pure);
    g.node(add).set_flag(NodeFlag::Pure);

    g.set_end(ret);
    g.n_parameters = 1;
    g.function_name = global_symbols().intern("sample");
    return g;
}
}  // namespace

TEST(ir_construct_and_access) {
    Graph g = build_sample_graph();
    CHECK_EQ(g.node_count(), 6u);   // reserved 0 + 5 nodes
    CHECK_EQ(g.live_node_count(), 5u);
    CHECK_EQ(g.node(g.start()).kind, NodeKind::Start);
    NodeId add = g.node(g.end()).ins[1];
    CHECK_EQ(g.node(add).kind, NodeKind::Add);
    CHECK_EQ(g.node(add).ins.size(), 2u);
    CHECK_EQ(g.node(add).use_count, 1u);
}

TEST(ir_print_parse_roundtrip) {
    Graph g = build_sample_graph();
    stdx::small_vector<char, 4096> text;
    CHECK(graph_to_string(g, text));
    std::string_view sv(text.data(), text.size());
    CHECK(sv.find("fun sample params=1") != std::string_view::npos);
    CHECK(sv.find("const_int 1") != std::string_view::npos);
    CHECK(sv.find("iadd") != std::string_view::npos);

    // Parse back and compare canonical text.
    Graph h;
    Result<void> parsed = parse_graph(sv, h);
    CHECK(parsed.has_value());
    stdx::small_vector<char, 4096> text2;
    CHECK(graph_to_string(h, text2));
    CHECK(text.size() == text2.size());
    CHECK(std::memcmp(text.data(), text2.data(), text.size()) == 0);
}

TEST(ir_verifier_catches_use_count_drift) {
    Graph g = build_sample_graph();
    CHECK(verify_or_report(g, "baseline"));

    // Corrupt bookkeeping directly (simulating a buggy pass).
    Node& ret = g.node(g.end());
    ret.use_count = 5;
    auto problems = verify_graph(g);
    CHECK(!problems.empty());
}

TEST(ir_verifier_catches_dangling_input) {
    Graph g = build_sample_graph();
    Node& ret = g.node(g.end());
    ret.ins[1] = 99999;   // dangling
    auto problems = verify_graph(g);
    CHECK(!problems.empty());
}

TEST(ir_verifier_requires_framestate_on_speculative_guard) {
    Graph g = build_sample_graph();
    NodeId cond = g.create(NodeKind::ConstInt);
    g.node(cond).const_value = Value::integer(1);
    NodeId guard = g.create(NodeKind::Guard, {cond});
    Node& gn = g.node(guard);
    gn.subop = static_cast<std::uint16_t>(GuardKind::IntFits);
    gn.set_flag(NodeFlag::Speculative);   // Rule 5 violated: no FrameState
    auto problems = verify_graph(g);
    bool caught = false;
    for (auto& d : problems) {
        if (d.code == diag_code::graph_verify_framestate) caught = true;
    }
    CHECK(caught);

    // Attach a FrameState -> valid.
    FrameState fs;
    fs.values.push_back(cond);
    fs.kinds.push_back(1);
    gn.aux1 = g.add_frame_state(fs);
    CHECK(verify_or_report(g, "guard-with-fs"));
}

TEST(ir_replace_all_uses) {
    Graph g = build_sample_graph();
    NodeId p0 = g.node(g.end()).ins[1];   // add
    NodeId old = g.node(p0).ins[0];       // param
    NodeId replacement = g.create(NodeKind::ConstInt);
    g.node(replacement).const_value = Value::integer(7);
    g.replace_all_uses(old, replacement);
    CHECK_EQ(g.node(p0).ins[0], replacement);
    CHECK_EQ(g.node(old).use_count, 0u);
    CHECK_EQ(g.node(replacement).use_count, 1u);
    CHECK(verify_or_report(g, "rauw"));
}

TEST(ir_pure_construction_dedupes) {
    Graph g;
    NodeId start = g.create(NodeKind::Start);
    g.set_start(start);
    NodeId a = g.create(NodeKind::ConstInt);
    g.node(a).const_value = Value::integer(5);
    NodeId b = g.create(NodeKind::ConstInt);
    g.node(b).const_value = Value::integer(5);
    NodeId x = g.create_pure_consed(NodeKind::Add, {a, b});
    NodeId y = g.create_pure_consed(NodeKind::Add, {a, b});
    CHECK(x == y);   // hash-consed
    NodeId c = g.create(NodeKind::ConstInt);
    g.node(c).const_value = Value::integer(6);
    NodeId z = g.create_pure_consed(NodeKind::Add, {a, c});
    CHECK(z != x);
}

TEST(ir_frame_state_storage) {
    Graph g;
    FrameState fs;
    fs.bytecode_offset = 128;
    fs.code_unit_id = 3;
    fs.values.push_back(7);
    fs.values.push_back(9);
    fs.kinds.push_back(0);
    fs.kinds.push_back(2);
    std::uint32_t idx = g.add_frame_state(fs);
    CHECK_EQ(g.frame_state_count(), 1u);
    CHECK_EQ(g.frame_state(idx).bytecode_offset, 128u);
    CHECK_EQ(g.frame_state(idx).values.size(), 2u);
}

TEST(ir_ty_lattice_joins) {
    TyTable t;
    TyIndex i = t.int_ty();
    TyIndex f = t.float_ty();
    TyIndex ior = t.join(i, f);
    CHECK(t[ior].kind == TyKind::FloatTy);
    TyIndex big = t.intern(Ty{.kind = TyKind::BignumTy});
    TyIndex j2 = t.join(i, big);
    CHECK(t[j2].kind == TyKind::BignumTy);
    TyIndex s = t.str_ty();
    CHECK(t.join(i, s) == ty_top);
    TyIndex l1 = t.list_ty(i);
    TyIndex l2 = t.list_ty(big);
    TyIndex lj = t.join(l1, l2);
    CHECK(t[lj].kind == TyKind::ListTy);
    CHECK(t[t[lj].elem].kind == TyKind::BignumTy);
}
