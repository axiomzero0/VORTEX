// =============================================================================
// vortex/rt/deopt.cpp — Deoptimization engine (Rule 4).
//
// Deoptimization reconstructs Tier-0 frames from JIT machine state using the
// FrameState attachments emitted at guard sites. Wired to the backend in
// phase 8; the interpreter entry (Vm::enter_at) is the landing pad.
// =============================================================================
