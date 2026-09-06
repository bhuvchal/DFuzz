// Does buffer reuse in Oracle actually matter? Measures it rather than
// asserting it: same workload, reused buffers vs a fresh Oracle per call.
#include "dfuzz/oracle.h"
#include "dfuzz/init.h"
#include <chrono>
#include <cstdio>
#include <random>

using namespace dfuzz;
using clk = std::chrono::steady_clock;

int main() {
    dfuzz::Guard guard;
    std::mt19937 rng(4);
    std::normal_distribution<float> d(0.f, 1.f);

    Case c; c.fmt = Format::nvfp4;
    const int n = 1024;
    c.x.resize(n); c.y.resize(n);
    for (int i = 0; i < n; ++i) { c.x[i] = d(rng); c.y[i] = d(rng); }

    const int iters = 20000;

    Oracle reused;
    reused.evaluate(c); // warm
    auto t0 = clk::now();
    for (int i = 0; i < iters; ++i) reused.evaluate(c);
    auto t1 = clk::now();

    auto t2 = clk::now();
    long long fresh_allocs = 0;
    for (int i = 0; i < iters; ++i) { Oracle o; o.evaluate(c); fresh_allocs += o.allocations(); }
    auto t3 = clk::now();

    const double ms_reused = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double ms_fresh  = std::chrono::duration<double, std::milli>(t3 - t2).count();

    std::printf("# oracle buffer reuse, nvfp4 n=%d, %d evaluations\n", n, iters);
    std::printf("  reused buffers : %8.2f ms  (%.3f us/eval)  buffer growths=%lld\n",
                ms_reused, ms_reused * 1000.0 / iters, reused.allocations());
    std::printf("  fresh per call : %8.2f ms  (%.3f us/eval)  buffer growths=%lld\n",
                ms_fresh, ms_fresh * 1000.0 / iters, fresh_allocs);
    std::printf("  speedup        : %.2fx\n", ms_fresh / ms_reused);
    return 0;
}
