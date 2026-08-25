// =============================================================================
// vortex/passes/pipeline.cpp — pipeline entry points.
// =============================================================================

#include "vortex/passes/pass_pipeline.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

// Global hooks (Rule 40 verifier, Rule 26 telemetry) — defined once here;
// every pass links against these.
bool (*g_verify_after_each_pass)(const ir::Graph&, const char*) = nullptr;
Telemetry* g_pass_telemetry = nullptr;

Result<void> optimize(ir::Graph& g, const PassContext& ctx) noexcept {
    OptPipeline pipeline;
    return run_pipeline(g, ctx, pipeline);
}

}  // namespace abi_v1
}  // namespace vortex::passes
