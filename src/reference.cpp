#include "dfuzz/reference.h"
#include <cstring>

namespace dfuzz {
namespace {

// Block layouts, mirrored from ggml-common.h. We redeclare rather than include
// so the harness is not coupled to ggml's internal headers.
constexpr int K_SCALE_SIZE = 12;

struct block_q4_K { uint16_t d, dmin; uint8_t scales[K_SCALE_SIZE]; uint8_t qs[QK_K/2]; };
struct block_q6_K { uint8_t ql[QK_K/2]; uint8_t qh[QK_K/4]; int8_t scales[QK_K/16]; uint16_t d; };
struct block_q8_K { float d; int8_t qs[QK_K]; int16_t bsums[QK_K/16]; };

static_assert(sizeof(block_q4_K) == 2*sizeof(uint16_t) + K_SCALE_SIZE + QK_K/2, "q4_K layout");
static_assert(sizeof(block_q6_K) == sizeof(uint16_t) + QK_K/16 + 3*QK_K/4,      "q6_K layout");
static_assert(sizeof(block_q8_K) == sizeof(float) + QK_K + QK_K/16*sizeof(int16_t), "q8_K layout");

// IEEE-754 binary16 -> double, exact. Handles subnormals, Inf and NaN.
// ggml's own converter goes via fp32; fp16->fp32 is exact, so routing through
// double here loses nothing and keeps the reference free of fp32 rounding.
double fp16_to_double(uint16_t h) {
    const uint32_t sign = (uint32_t)(h >> 15) & 1u;
    const uint32_t exp  = (uint32_t)(h >> 10) & 0x1Fu;
    const uint32_t mant = (uint32_t)h & 0x3FFu;
    double mag;
    if (exp == 0) {
        mag = (double)mant * 5.9604644775390625e-08; // 2^-24
    } else if (exp == 0x1F) {
        mag = mant ? __builtin_nan("") : __builtin_inf();
    } else {
        // (1 + mant/1024) * 2^(exp-15), built exactly.
        mag = (1.0 + (double)mant / 1024.0) * __builtin_ldexp(1.0, (int)exp - 15);
    }
    return sign ? -mag : mag;
}

} // namespace

void dequantize_q4_K(const void* vx, int nblocks, std::vector<double>& out) {
    const auto* x = static_cast<const block_q4_K*>(vx);
    out.resize((size_t)nblocks * QK_K);
    double* y = out.data();
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* q = x[i].qs;
        const double d    = fp16_to_double(x[i].d);
        const double dmin = fp16_to_double(x[i].dmin);
        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const double d1 = d * sc, m1 = dmin * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const double d2 = d * sc, m2 = dmin * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >>  4) - m2;
            q += 32; is += 2;
        }
    }
}

void dequantize_q6_K(const void* vx, int nblocks, std::vector<double>& out) {
    const auto* x = static_cast<const block_q6_K*>(vx);
    out.resize((size_t)nblocks * QK_K);
    double* y = out.data();
    for (int i = 0; i < nblocks; ++i) {
        const double d = fp16_to_double(x[i].d);
        const uint8_t* ql = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const int8_t*  sc = x[i].scales;
        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int q1 = (int)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = (int)((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = (int)((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y += 128; ql += 64; qh += 32; sc += 8;
        }
    }
}

void dequantize_q8_K(const void* vy, int nblocks, std::vector<double>& out) {
    const auto* y = static_cast<const block_q8_K*>(vy);
    out.resize((size_t)nblocks * QK_K);
    double* o = out.data();
    for (int i = 0; i < nblocks; ++i) {
        const double d = (double)y[i].d;
        for (int l = 0; l < QK_K; ++l) *o++ = d * (double)y[i].qs[l];
    }
}

RefResult reference_dot(const std::vector<double>& x, const std::vector<double>& y) {
    RefResult r{0.0, 0.0};
    const size_t n = x.size() < y.size() ? x.size() : y.size();
    for (size_t i = 0; i < n; ++i) {
        const double p = x[i] * y[i];
        r.dot     += p;
        r.abs_sum += p < 0 ? -p : p;
    }
    return r;
}

} // namespace dfuzz
