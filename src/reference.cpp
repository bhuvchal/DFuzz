#include "dfuzz/reference.h"
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <vector>

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

// Sum over sub-blocks of |dmin * min * d8 * sum(q8)|: the magnitude of the
// offset correction the q4_K kernel forms and rounds in fp32.
double q4k_offset_magnitude(const void* vx, const void* vy, int nblocks) {
    struct blk { uint16_t d, dmin; uint8_t scales[12]; uint8_t qs[128]; };
    struct b8  { float d; int8_t qs[256]; int16_t bsums[16]; };
    const auto* x = static_cast<const blk*>(vx);
    const auto* y = static_cast<const b8*>(vy);

    auto h2d = [](uint16_t h) {
        const uint32_t s = (h >> 15) & 1u, e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
        double v;
        if (e == 0)        v = (double)m * 5.9604644775390625e-08;
        else if (e == 0x1F) v = m ? __builtin_nan("") : __builtin_inf();
        else               v = (1.0 + (double)m / 1024.0) * __builtin_ldexp(1.0, (int)e - 15);
        return s ? -v : v;
    };

    double total = 0.0;
    for (int i = 0; i < nblocks; ++i) {
        const double dmin = h2d(x[i].dmin);
        const double d8   = (double)y[i].d;
        for (int sb = 0; sb < 8; ++sb) {
            uint8_t mn;
            if (sb < 4) mn = x[i].scales[sb + 4] & 63;
            else        mn = (uint8_t)((x[i].scales[sb + 4] >> 4) | ((x[i].scales[sb] >> 6) << 4));
            int sum8 = 0;
            for (int k = sb * 32; k < (sb + 1) * 32; ++k) sum8 += y[i].qs[k];
            const double t = dmin * (double)mn * d8 * (double)sum8;
            total += t < 0 ? -t : t;
        }
    }
    return total;
}


RefResult reference_dot(const std::vector<double>& x, const std::vector<double>& y) {
    RefResult r{0.0, 0.0, 0.0};
    const size_t n = x.size() < y.size() ? x.size() : y.size();
    for (size_t i = 0; i < n; ++i) {
        const double p = x[i] * y[i];
        r.dot     += p;
        r.abs_sum += p < 0 ? -p : p;
    }
    r.intermediate_mag = r.abs_sum;
    return r;
}

RefResult reference_dot_blocked(const std::vector<double>& x,
                                const std::vector<double>& y,
                                int sub_block) {
    RefResult r{0.0, 0.0, 0.0};
    const size_t n = x.size() < y.size() ? x.size() : y.size();
    if (sub_block <= 0) return reference_dot(x, y);

    // Track the peak magnitude of each sub-block's partial sum alongside the
    // total. When sub-blocks are large and cancel against each other, the
    // intermediates the kernel rounds are far bigger than either the result or
    // sum|x_i y_i|, and the bound must scale with them.
    for (size_t base = 0; base < n; base += (size_t)sub_block) {
        const size_t hi = std::min(base + (size_t)sub_block, n);
        double part = 0.0;
        for (size_t i = base; i < hi; ++i) {
            const double p = x[i] * y[i];
            part      += p;
            r.abs_sum += p < 0 ? -p : p;
        }
        r.dot += part;
        const double a = part < 0 ? -part : part;
        if (a > r.intermediate_mag) r.intermediate_mag = a;
    }
    // The bound must cover both the per-term products and the sub-block
    // partials, whichever dominates.
    if (r.abs_sum > r.intermediate_mag) r.intermediate_mag = r.abs_sum;
    return r;
}

} // namespace dfuzz
