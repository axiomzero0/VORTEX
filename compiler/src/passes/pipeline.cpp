// =============================================================================
// vortex/passes/pipeline.cpp — pipeline entry points.
// =============================================================================

#include "vortex/passes/pass_pipeline.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

Result<void> optimize(ir::Graph& g, const PassContext& ctx) noexcept {
    OptPipeline pipeline;
    return run_pipeline(g, ctx, pipeline);
}

}  // namespace abi_v1
}  // namespace vortex::passes
