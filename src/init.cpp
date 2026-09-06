#include "dfuzz/init.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>

extern "C" {
void ggml_cpu_init(void);
extern float ggml_table_f32_ue4m3[1 << 8];
extern float ggml_table_f32_e8m0_half[1 << 8];
}

namespace dfuzz {
namespace {
std::once_flag g_once;
std::atomic<int> g_live{0};
} // namespace

Guard::Guard() {
    std::call_once(g_once, [] {
        ggml_cpu_init();
        if (!tables_look_initialized()) {
            std::fprintf(stderr,
                "dfuzz: ggml_cpu_init() ran but lookup tables are still zero.\n"
                "       Kernel results would be silently wrong. Aborting.\n");
            std::abort();
        }
    });
    g_live.fetch_add(1, std::memory_order_relaxed);
}

Guard::~Guard() { g_live.fetch_sub(1, std::memory_order_relaxed); }

void require_initialized(const char* where) {
    if (g_live.load(std::memory_order_relaxed) <= 0) {
        std::fprintf(stderr,
            "dfuzz: %s invoked a ggml kernel with no dfuzz::Guard in scope.\n"
            "       SIMD kernels read uninitialized tables and return 0.\n", where);
        std::abort();
    }
}

bool tables_look_initialized() {
    // ue4m3(0x23) and e8m0_half(127) are both nonzero once filled.
    return ggml_table_f32_ue4m3[0x23] != 0.0f &&
           ggml_table_f32_e8m0_half[127] != 0.0f;
}

} // namespace dfuzz
