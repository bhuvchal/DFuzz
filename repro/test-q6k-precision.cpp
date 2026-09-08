// Regression test for q6_K vec_dot precision.
//
// Shaped to drop into llama.cpp's tests/ directory. Fails on x86 AVX2 before
// a fix, passes after, and passes today on arm64 and on the generic path.
//
// The case is a single q6_K super-block in which exactly one 16-element group
// contributes. The kernel's task therefore reduces to one product of four
// exactly-representable fp32 values:
//
//     d * scales[8] * d8 * sumi  =  2790 * 9 * 57.346172332763672 * 20
//
// with exact value 28799247.745513916. Because there is no accumulation, the
// admissible result set is small and can be stated exactly: enumerating all
// evaluation orders in fp32 yields a worst case of 1.127 ulp, and the
// IEEE-754 bound over any order is ((1+u)^3 - 1)*result = 2.575 ulp.
//
// The tolerance below is 4 ulp -- comfortably above the 2.575 ulp bound, so
// no correctly-rounded implementation can fail it, and far below the 2183.9
// ulp currently observed on x86 AVX2.

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

extern "C" {
void quantize_row_q6_K_ref(const float *, void *, int64_t);
void quantize_row_q8_K_ref(const float *, void *, int64_t);
void dequantize_row_q6_K(const void *, float *, int64_t);
void ggml_vec_dot_q6_K_q8_K(int, float *, size_t, const void *, size_t, const void *, size_t, int);
}

int main() {
    ggml_cpu_init();

    constexpr int N = 256;

    std::vector<float> x(N, 0.0f);
    std::vector<float> y(N, 0.0f);

    x[87]  =  11429230.0f;
    x[140] = -774724.25f;
    x[143] =  510633.031f;
    y[86]  = -7282.96387f;
    y[143] =  65.4250183f;

    std::vector<uint8_t> xq(ggml_row_size(GGML_TYPE_Q6_K, N));
    std::vector<uint8_t> yq(ggml_row_size(GGML_TYPE_Q8_K, N));
    quantize_row_q6_K_ref(x.data(), xq.data(), N);
    quantize_row_q8_K_ref(y.data(), yq.data(), N);

    float res = 0.0f;
    ggml_vec_dot_q6_K_q8_K(N, &res, 0, xq.data(), 0, yq.data(), 0, 1);

    // Exact value: dequantize with ggml's own dequantizer, sum in double.
    // Both operands are consumed exactly as the kernel sees them.
    std::vector<float> xd(N);
    dequantize_row_q6_K(xq.data(), xd.data(), N);

    float d8;
    std::memcpy(&d8, yq.data(), sizeof(float));
    const int8_t * q8 = reinterpret_cast<const int8_t *>(yq.data() + sizeof(float));

    double exact = 0.0;
    for (int i = 0; i < N; ++i) {
        exact += (double) xd[i] * ((double) d8 * (double) q8[i]);
    }

    const double ulp = (double) std::nextafterf((float) exact, INFINITY) - (double) (float) exact;
    const double err = std::fabs((double) res - exact) / ulp;

    printf("q6_K vec_dot: got %.9g, exact %.17g, error %.3f ulp\n", (double) res, exact, err);

    if (err > 4.0) {
        printf("FAIL: %.1f ulp exceeds the 4 ulp tolerance "
               "(IEEE-754 bound for this case is 2.575 ulp)\n", err);
        return 1;
    }

    printf("OK\n");
    return 0;
}
