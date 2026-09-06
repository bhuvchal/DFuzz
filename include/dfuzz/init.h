#pragma once

namespace dfuzz {

// ggml's SIMD kernels read runtime-initialized lookup tables (UE4M3 scales,
// E8M0 half-scales, fp16->fp32). ggml_cpu_init() fills them; without it they
// are all zero and affected kernels silently return 0 rather than crashing.
//
// This is a harness trap, not an upstream bug, and it is a nasty one: the
// _generic paths compute their scales directly and are unaffected, so the
// failure looks exactly like "the SIMD kernel is broken".
//
// Guard is an RAII sentinel: constructing one guarantees initialization has
// happened. Every harness entry point constructs one before touching a kernel,
// and require_initialized() aborts loudly if some path forgets.
class Guard {
public:
    Guard();
    ~Guard();
    Guard(const Guard&)            = delete;
    Guard& operator=(const Guard&) = delete;
    Guard(Guard&&)                 = delete;
    Guard& operator=(Guard&&)      = delete;
};

// Aborts unless a Guard is live. Called by the kernel-invoking wrappers so a
// missing Guard fails immediately instead of producing a phantom divergence.
void require_initialized(const char* where);

// Self-check: verifies the tables ggml_cpu_init() should have filled are
// actually non-zero. Returns false if initialization did not take effect.
bool tables_look_initialized();

} // namespace dfuzz
