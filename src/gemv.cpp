#include "dfuzz/gemv.h"
#include "dfuzz/formats.h"
#include "dfuzz/reference.h"
#include "dfuzz/init.h"
#include <algorithm>
#include <cstring>

extern "C" {
void quantize_row_mxfp4_ref(const float*, void*, int64_t);
void quantize_row_q8_0_ref (const float*, void*, int64_t);
void quantize_row_q4_K_ref (const float*, void*, int64_t);
void quantize_row_q8_K_ref (const float*, void*, int64_t);
void ggml_vec_dot_q4_K_q8_K(int,float*,size_t,const void*,size_t,const void*,size_t,int);
void ggml_gemv_q4_K_8x8_q8_K        (int,float*,size_t,const void*,const void*,int,int);
void ggml_gemv_q4_K_8x8_q8_K_generic(int,float*,size_t,const void*,const void*,int,int);
void ggml_gemm_mxfp4_4x4_q8_0        (int,float*,size_t,const void*,const void*,int,int);
void ggml_gemm_mxfp4_4x4_q8_0_generic(int,float*,size_t,const void*,const void*,int,int);
void ggml_quantize_mat_q8_0_4x4(const float*, void*, int64_t);
void ggml_quantize_mat_q8_K_4x8(const float*, void*, int64_t);
void ggml_gemm_q4_K_8x8_q8_K        (int,float*,size_t,const void*,const void*,int,int);
void ggml_gemm_q4_K_8x8_q8_K_generic(int,float*,size_t,const void*,const void*,int,int);
void ggml_vec_dot_mxfp4_q8_0(int,float*,size_t,const void*,size_t,const void*,size_t,int);
void ggml_gemv_mxfp4_4x4_q8_0        (int,float*,size_t,const void*,const void*,int,int);
void ggml_gemv_mxfp4_4x4_q8_0_generic(int,float*,size_t,const void*,const void*,int,int);
}

namespace dfuzz {

namespace {
struct block_mxfp4   { uint8_t e;    uint8_t qs[QK_MXFP4/2]; };
struct block_mxfp4x4 { uint8_t e[4]; uint8_t qs[QK_MXFP4*2]; };
static_assert(sizeof(block_mxfp4x4) == 4 + QK_MXFP4*2, "mxfp4x4 layout");
struct block_q4_K   { uint16_t d, dmin; uint8_t scales[12]; uint8_t qs[128]; };
struct block_q4_Kx8 { uint16_t d[8]; uint16_t dmin[8]; uint8_t scales[96]; uint8_t qs[1024]; };
static_assert(sizeof(block_q4_Kx8) == 16*sizeof(uint16_t) + 96 + 1024, "q4_Kx8 layout");
} // namespace

void repack_mxfp4_4x4(const void* src_rows, int nrows, int nblocks,
                      std::vector<uint8_t>& out) {
    // src_rows is nrows contiguous rows of nblocks block_mxfp4 each.
    // Output is (nrows/4) groups, each of nblocks block_mxfp4x4.
    const auto* src = static_cast<const block_mxfp4*>(src_rows);
    out.assign((size_t)(nrows / 4) * nblocks * sizeof(block_mxfp4x4), 0);
    auto* dst = reinterpret_cast<block_mxfp4x4*>(out.data());

    constexpr int interleave = 4;
    for (int b = 0; b < nrows; b += 4) {
        for (int x = 0; x < nblocks; ++x) {
            block_mxfp4 in[4];
            for (int i = 0; i < 4; ++i) in[i] = src[x + (size_t)(b + i) * nblocks];

            block_mxfp4x4 o{};
            for (int i = 0; i < 4; ++i) o.e[i] = in[i].e;
            // 4-byte round-robin across the four rows. Mirrors
            // make_block_mxfp4x4(.., blck_size_interleave = 4).
            const int end = QK_MXFP4 * 2 / interleave;
            for (int i = 0; i < end; ++i) {
                const int src_id     = i % 4;
                const int src_offset = (i / 4) * interleave;
                const int dst_offset = i * interleave;
                std::memcpy(&o.qs[dst_offset], &in[src_id].qs[src_offset], sizeof(uint32_t));
            }
            *dst++ = o;
        }
    }
}

void repack_q4_K_8x8(const void* src_rows, int nrows, int nblocks,
                     std::vector<uint8_t>& out) {
    const auto* src = static_cast<const block_q4_K*>(src_rows);
    out.assign((size_t)(nrows / 8) * nblocks * sizeof(block_q4_Kx8), 0);
    auto* dst = reinterpret_cast<block_q4_Kx8*>(out.data());

    constexpr int interleave = 8;
    for (int b = 0; b < nrows; b += 8) {
        for (int x = 0; x < nblocks; ++x) {
            block_q4_K in[8];
            for (int i = 0; i < 8; ++i) in[i] = src[x + (size_t)(b + i) * nblocks];

            block_q4_Kx8 o{};
            for (int i = 0; i < 8; ++i) { o.d[i] = in[i].d; o.dmin[i] = in[i].dmin; }

            // 8-byte round-robin over the 4-bit quants.
            const int end = QK_K * 4 / interleave;
            for (int i = 0; i < end; ++i) {
                const int src_id     = i % 8;
                const int src_offset = (i / 8) * interleave;
                const int dst_offset = i * interleave;
                uint64_t elems;
                std::memcpy(&elems, &in[src_id].qs[src_offset], interleave);
                std::memcpy(&o.qs[dst_offset], &elems, interleave);
            }

            // Scale/min re-pack: low 6 bits of sub-blocks 0-3 for all 8 rows,
            // with each value's high 2 bits folded into a companion byte.
            uint8_t s[8], m[8];
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 8; ++j) {
                    s[j] = in[j].scales[i]     & 63;
                    m[j] = in[j].scales[i + 4] & 63;
                }
                for (int j = 0; j < 4; ++j) {
                    o.scales[i*12 + j]     = (uint8_t)((s[j] & 63) + ((s[j+4] & 48) << 2));
                    o.scales[i*12 + 4 + j] = (uint8_t)((m[j] & 63) + ((m[j+4] & 48) << 2));
                    o.scales[i*12 + 8 + j] = (uint8_t)((s[j+4] & 15) + ((m[j+4] & 15) << 4));
                }
            }
            // Sub-blocks 4-7, whose 6 bits are split across scales[i] and scales[i+8].
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 8; ++j) {
                    s[j] = (uint8_t)(((in[j].scales[i]     & 192) >> 2) |  (in[j].scales[i+8] & 15));
                    m[j] = (uint8_t)(((in[j].scales[i + 4] & 192) >> 2) | ((in[j].scales[i+8] & 240) >> 4));
                }
                for (int j = 0; j < 4; ++j) {
                    o.scales[i*12 + 48 + j]     = (uint8_t)((s[j] & 63) + ((s[j+4] & 48) << 2));
                    o.scales[i*12 + 52 + j]     = (uint8_t)((m[j] & 63) + ((m[j+4] & 48) << 2));
                    o.scales[i*12 + 56 + j]     = (uint8_t)((s[j+4] & 15) + ((m[j+4] & 15) << 4));
                }
            }
            *dst++ = o;
        }
    }
}

GemvResult run_q4_K_gemv(const std::vector<std::vector<float>>& rows,
                         const std::vector<float>& act) {
    require_initialized("run_q4_K_gemv");
    GemvResult r;
    const int nc = (int)rows.size();
    const int n  = (int)act.size();
    const int nb = n / 256;
    if (nc == 0 || nc % 8 != 0 || nb == 0) return r;

    std::vector<uint8_t> rowq((size_t)nc * nb * sizeof(block_q4_K));
    for (int j = 0; j < nc; ++j)
        quantize_row_q4_K_ref(rows[j].data(), rowq.data() + (size_t)j*nb*sizeof(block_q4_K), n);

    std::vector<uint8_t> packed;
    repack_q4_K_8x8(rowq.data(), nc, nb, packed);

    std::vector<uint8_t> yq((size_t)nb * 292);
    quantize_row_q8_K_ref(act.data(), yq.data(), n);

    r.simd.assign(nc, 0.0f);
    r.generic.assign(nc, 0.0f);
    ggml_gemv_q4_K_8x8_q8_K        (n, r.simd.data(),    0, packed.data(), yq.data(), 1, nc);
    ggml_gemv_q4_K_8x8_q8_K_generic(n, r.generic.data(), 0, packed.data(), yq.data(), 1, nc);

    r.vecdot.assign(nc, 0.0f);
    r.reference.assign(nc, 0.0);
    r.intermediate_mag.assign(nc, 0.0);
    std::vector<double> yd;
    dequantize_q8_K(yq.data(), nb, yd);
    for (int j = 0; j < nc; ++j) {
        const void* rp = rowq.data() + (size_t)j*nb*sizeof(block_q4_K);
        ggml_vec_dot_q4_K_q8_K(n, &r.vecdot[j], 0, rp, 0, yq.data(), 0, 1);
        std::vector<double> xd;
        dequantize_q4_K(rp, nb, xd);
        const RefResult rr = reference_dot_blocked(xd, yd, 32);
        r.reference[j] = rr.dot;
        // q4_K carries an offset term whose magnitude dwarfs sum|x_i y_i|;
        // the bound is wrong without it. Shared with the dot-product path
        // rather than reimplemented, which is why it lives in reference.h.
        r.intermediate_mag[j] = std::max(rr.intermediate_mag,
                                         q4k_offset_magnitude(rp, yq.data(), nb));
    }
    return r;
}

GemvResult run_mxfp4_gemv(const std::vector<std::vector<float>>& rows,
                          const std::vector<float>& act) {
    require_initialized("run_mxfp4_gemv");
    GemvResult r;
    const int nc = (int)rows.size();
    const int n  = (int)act.size();
    const int nb = n / QK_MXFP4;
    if (nc == 0 || nc % 4 != 0 || nb == 0) return r;

    // Quantize each weight row independently, then interleave.
    std::vector<uint8_t> rowq((size_t)nc * nb * sizeof(block_mxfp4));
    for (int j = 0; j < nc; ++j)
        quantize_row_mxfp4_ref(rows[j].data(), rowq.data() + (size_t)j*nb*sizeof(block_mxfp4), n);

    std::vector<uint8_t> packed;
    repack_mxfp4_4x4(rowq.data(), nc, nb, packed);

    std::vector<uint8_t> yq((size_t)nb * 34);
    quantize_row_q8_0_ref(act.data(), yq.data(), n);

    r.simd.assign(nc, 0.0f);
    r.generic.assign(nc, 0.0f);
    ggml_gemv_mxfp4_4x4_q8_0        (n, r.simd.data(),    0, packed.data(), yq.data(), 1, nc);
    ggml_gemv_mxfp4_4x4_q8_0_generic(n, r.generic.data(), 0, packed.data(), yq.data(), 1, nc);

    // Per-row plain dot product: the strongest check on this surface, because
    // it makes the interleaving falsifiable with no reference of ours involved.
    r.vecdot.assign(nc, 0.0f);
    r.reference.assign(nc, 0.0);
    r.intermediate_mag.assign(nc, 0.0);
    std::vector<double> yd;
    dequantize_q8_0(yq.data(), nb, yd);
    for (int j = 0; j < nc; ++j) {
        const void* rp = rowq.data() + (size_t)j*nb*sizeof(block_mxfp4);
        ggml_vec_dot_mxfp4_q8_0(n, &r.vecdot[j], 0, rp, 0, yq.data(), 0, 1);
        std::vector<double> xd;
        dequantize_mxfp4(rp, nb, xd);
        const RefResult rr = reference_dot_blocked(xd, yd, QK_MXFP4);
        r.reference[j] = rr.dot;
        r.intermediate_mag[j] = rr.intermediate_mag;
    }
    return r;
}

} // namespace dfuzz

namespace dfuzz {

GemmResult run_mxfp4_gemm(const std::vector<std::vector<float>>& wrows,
                          const std::vector<std::vector<float>>& arows) {
    require_initialized("run_mxfp4_gemm");
    GemmResult r;
    const int nc = (int)wrows.size();
    const int nr = (int)arows.size();
    if (nc == 0 || nr == 0 || nc % 4 != 0 || nr % 4 != 0) return r;
    const int n  = (int)arows[0].size();
    const int nb = n / QK_MXFP4;
    if (nb == 0) return r;
    r.nr = nr; r.nc = nc;

    // Weight side: quantize each row, then interleave 4-wide (same packing the
    // gemv path uses, so the two are directly comparable).
    std::vector<uint8_t> rowq((size_t)nc * nb * sizeof(block_mxfp4));
    for (int j = 0; j < nc; ++j)
        quantize_row_mxfp4_ref(wrows[(size_t)j].data(),
                               rowq.data() + (size_t)j*nb*sizeof(block_mxfp4), n);
    std::vector<uint8_t> packed;
    repack_mxfp4_4x4(rowq.data(), nc, nb, packed);

    // Activation side: ggml's own matrix quantizer, which expects 4 rows
    // contiguous and produces block_q8_0x4. Using theirs rather than
    // reimplementing removes our largest remaining source of reference error.
    std::vector<float> aflat((size_t)nr * n);
    for (int m = 0; m < nr; ++m)
        std::memcpy(aflat.data() + (size_t)m*n, arows[(size_t)m].data(), (size_t)n*sizeof(float));

    const size_t q8x4_bytes = 4*sizeof(uint16_t) + (size_t)QK8_0*4;
    std::vector<uint8_t> aq((size_t)(nr/4) * nb * q8x4_bytes);
    for (int g = 0; g < nr/4; ++g)
        ggml_quantize_mat_q8_0_4x4(aflat.data() + (size_t)g*4*n,
                                   aq.data() + (size_t)g*nb*q8x4_bytes, n);

    r.simd.assign((size_t)nr*nc, 0.0f);
    r.generic.assign((size_t)nr*nc, 0.0f);
    ggml_gemm_mxfp4_4x4_q8_0        (n, r.simd.data(),    (size_t)nc, packed.data(), aq.data(), nr, nc);
    ggml_gemm_mxfp4_4x4_q8_0_generic(n, r.generic.data(), (size_t)nc, packed.data(), aq.data(), nr, nc);

    // Reference-free cross-check: run gemv per activation row on the SAME
    // repacked weights.
    r.gemv_rows.assign((size_t)nr*nc, 0.0f);
    r.reference.assign((size_t)nr*nc, 0.0);
    r.intermediate_mag.assign((size_t)nr*nc, 0.0);
    std::vector<double> yd;
    for (int m = 0; m < nr; ++m) {
        std::vector<uint8_t> yq((size_t)nb * 34);
        quantize_row_q8_0_ref(arows[(size_t)m].data(), yq.data(), n);
        ggml_gemv_mxfp4_4x4_q8_0(n, r.gemv_rows.data() + (size_t)m*nc, 0,
                                 packed.data(), yq.data(), 1, nc);
        dequantize_q8_0(yq.data(), nb, yd);
        for (int j = 0; j < nc; ++j) {
            std::vector<double> xd;
            dequantize_mxfp4(rowq.data() + (size_t)j*nb*sizeof(block_mxfp4), nb, xd);
            const RefResult rr = reference_dot_blocked(xd, yd, QK_MXFP4);
            r.reference[(size_t)m*nc + j]        = rr.dot;
            r.intermediate_mag[(size_t)m*nc + j] = rr.intermediate_mag;
        }
    }
    return r;
}

} // namespace dfuzz

namespace dfuzz {

GemmResult run_q4_K_gemm(const std::vector<std::vector<float>>& wrows,
                         const std::vector<std::vector<float>>& arows) {
    require_initialized("run_q4_K_gemm");
    GemmResult r;
    const int nc = (int)wrows.size();
    const int nr = (int)arows.size();
    if (nc == 0 || nr == 0 || nc % 8 != 0 || nr % 4 != 0) return r;
    const int n  = (int)arows[0].size();
    const int nb = n / 256;
    if (nb == 0) return r;
    r.nr = nr; r.nc = nc;

    std::vector<uint8_t> rowq((size_t)nc * nb * sizeof(block_q4_K));
    for (int j = 0; j < nc; ++j)
        quantize_row_q4_K_ref(wrows[(size_t)j].data(),
                              rowq.data() + (size_t)j*nb*sizeof(block_q4_K), n);
    std::vector<uint8_t> packed;
    repack_q4_K_8x8(rowq.data(), nc, nb, packed);

    std::vector<float> aflat((size_t)nr * n);
    for (int m = 0; m < nr; ++m)
        std::memcpy(aflat.data() + (size_t)m*n, arows[(size_t)m].data(), (size_t)n*sizeof(float));

    // block_q8_Kx4: 4 floats + 4*256 int8 + (256/4) int16
    const size_t q8Kx4_bytes = 4*sizeof(float) + 4*256 + (256/4)*sizeof(int16_t);
    std::vector<uint8_t> aq((size_t)(nr/4) * nb * q8Kx4_bytes);
    for (int g = 0; g < nr/4; ++g)
        ggml_quantize_mat_q8_K_4x8(aflat.data() + (size_t)g*4*n,
                                   aq.data() + (size_t)g*nb*q8Kx4_bytes, n);

    r.simd.assign((size_t)nr*nc, 0.0f);
    r.generic.assign((size_t)nr*nc, 0.0f);
    ggml_gemm_q4_K_8x8_q8_K        (n, r.simd.data(),    (size_t)nc, packed.data(), aq.data(), nr, nc);
    ggml_gemm_q4_K_8x8_q8_K_generic(n, r.generic.data(), (size_t)nc, packed.data(), aq.data(), nr, nc);

    r.gemv_rows.assign((size_t)nr*nc, 0.0f);
    r.reference.assign((size_t)nr*nc, 0.0);
    r.intermediate_mag.assign((size_t)nr*nc, 0.0);
    std::vector<double> yd;
    for (int m = 0; m < nr; ++m) {
        std::vector<uint8_t> yq((size_t)nb * 292);
        quantize_row_q8_K_ref(arows[(size_t)m].data(), yq.data(), n);
        ggml_gemv_q4_K_8x8_q8_K(n, r.gemv_rows.data() + (size_t)m*nc, 0,
                                packed.data(), yq.data(), 1, nc);
        dequantize_q8_K(yq.data(), nb, yd);
        for (int j = 0; j < nc; ++j) {
            const void* rp = rowq.data() + (size_t)j*nb*sizeof(block_q4_K);
            std::vector<double> xd;
            dequantize_q4_K(rp, nb, xd);
            const RefResult rr = reference_dot_blocked(xd, yd, 32);
            r.reference[(size_t)m*nc + j] = rr.dot;
            r.intermediate_mag[(size_t)m*nc + j] =
                std::max(rr.intermediate_mag, q4k_offset_magnitude(rp, yq.data(), nb));
        }
    }
    return r;
}

} // namespace dfuzz
