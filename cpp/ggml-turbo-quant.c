/*
 * TurboQuant: KV cache compression via PolarQuant + QJL
 * Based on: arXiv 2504.19874 (ICLR 2026)
 *
 * Implements LM_GGML_TYPE_TURBO3_0 (3-bit) and LM_GGML_TYPE_TURBO4_0 (4-bit)
 * for use as --cache-type-k turbo3 --cache-type-v turbo3 in llama-server.
 */

#include "ggml-quants.h"
#include "ggml-common.h"
#include "ggml-impl.h"

#include <math.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>

/* ---------- constants ---------- */

#define TURBO_SEED_ROTATION 42
#define TURBO_SEED_QJL      1042
#define TURBO_D             128  /* rotation group size = head_dim (independent of block size) */
#define TURBO_QJL_CONST     1.2533141373155003f  /* sqrt(pi/2) */

/* Optimal centroids from paper (scaled by 1/sqrt(d)) */
/* 1-bit: ±sqrt(2/(pi*d)) */
static const float CENTROIDS_1BIT[2] = { -0.070711f, 0.070711f };  /* for d=128 */

/* 2-bit: {±0.453, ±1.51} / sqrt(d) */
static const float CENTROIDS_2BIT[4] = { -0.133462f, -0.039994f, 0.039994f, 0.133462f };

/* 3-bit: Lloyd-Max for N(0, 1/128), pre-computed */
static const float CENTROIDS_3BIT[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

/* ---------- rotation matrix (lazy init) ---------- */

static float turbo_rotation[TURBO_D * TURBO_D];
static float turbo_rotation_t[TURBO_D * TURBO_D]; /* transpose */
static int   turbo_rotation_initialized = 0;

/* Simple LCG PRNG for deterministic rotation generation */
static uint64_t turbo_prng_state;

static void turbo_prng_seed(uint64_t seed) {
    turbo_prng_state = seed;
}

static double turbo_prng_normal(void) {
    /* Box-Muller transform from uniform LCG */
    turbo_prng_state = turbo_prng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u1 = (double)(turbo_prng_state >> 11) / (double)(1ULL << 53);
    if (u1 < 1e-15) u1 = 1e-15;
    turbo_prng_state = turbo_prng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u2 = (double)(turbo_prng_state >> 11) / (double)(1ULL << 53);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

static void turbo_init_rotation(void) {
    if (turbo_rotation_initialized) return;

    const int d = TURBO_D;

    /* Generate random Gaussian matrix */
    turbo_prng_seed(TURBO_SEED_ROTATION);
    float G[TURBO_D * TURBO_D];
    for (int i = 0; i < d * d; i++) {
        G[i] = (float)turbo_prng_normal();
    }

    /* QR decomposition via modified Gram-Schmidt */
    /* Q stored column-major in turbo_rotation */
    memcpy(turbo_rotation, G, d * d * sizeof(float));

    for (int j = 0; j < d; j++) {
        /* Normalize column j */
        float norm = 0.0f;
        for (int i = 0; i < d; i++) {
            norm += turbo_rotation[i * d + j] * turbo_rotation[i * d + j];
        }
        norm = sqrtf(norm);
        if (norm > 1e-10f) {
            for (int i = 0; i < d; i++) {
                turbo_rotation[i * d + j] /= norm;
            }
        }

        /* Orthogonalize remaining columns against j */
        for (int k = j + 1; k < d; k++) {
            float dot = 0.0f;
            for (int i = 0; i < d; i++) {
                dot += turbo_rotation[i * d + j] * turbo_rotation[i * d + k];
            }
            for (int i = 0; i < d; i++) {
                turbo_rotation[i * d + k] -= dot * turbo_rotation[i * d + j];
            }
        }
    }

    /* Compute transpose */
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            turbo_rotation_t[i * d + j] = turbo_rotation[j * d + i];
        }
    }

    turbo_rotation_initialized = 1;
}

/* ---------- QJL projection matrix (lazy init, seed-based) ---------- */

static float turbo_qjl_matrix[TURBO_D * TURBO_D];
static float turbo_qjl_matrix_t[TURBO_D * TURBO_D];
static int   turbo_qjl_initialized = 0;

static void turbo_init_qjl(void) {
    if (turbo_qjl_initialized) return;

    const int d = TURBO_D;
    turbo_prng_seed(TURBO_SEED_QJL);

    for (int i = 0; i < d * d; i++) {
        turbo_qjl_matrix[i] = (float)turbo_prng_normal();
    }

    /* Transpose */
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            turbo_qjl_matrix_t[i * d + j] = turbo_qjl_matrix[j * d + i];
        }
    }

    turbo_qjl_initialized = 1;
}

/* ---------- helper: matrix-vector multiply ---------- */

static void matvec(const float * M, const float * x, float * y, int d) {
    /* y = M @ x, M is row-major d×d */
    for (int i = 0; i < d; i++) {
        float sum = 0.0f;
        for (int j = 0; j < d; j++) {
            sum += M[i * d + j] * x[j];
        }
        y[i] = sum;
    }
}

/* ---------- nearest centroid ---------- */

static int nearest_centroid_2bit(float val) {
    /* Binary search on midpoints: {-0.133, -0.040, 0.040, 0.133} */
    if (val < -0.086728f) return 0;       /* midpoint(-0.133, -0.040) */
    if (val <  0.000000f) return 1;       /* midpoint(-0.040, 0.040) */
    if (val <  0.086728f) return 2;       /* midpoint(0.040, 0.133) */
    return 3;
}

static int nearest_centroid_3bit(float val) {
    /* 8 centroids, find nearest via midpoints */
    if (val < -0.154259f) return 0;
    if (val < -0.091775f) return 1;
    if (val < -0.043589f) return 2;
    if (val <  0.000000f) return 3;
    if (val <  0.043589f) return 4;
    if (val <  0.091775f) return 5;
    if (val <  0.154259f) return 6;
    return 7;
}

/* ---------- TURBO3_0: 2-bit PolarQuant + 1-bit QJL ---------- */

void quantize_row_turbo3_0_ref(const float * LM_GGML_RESTRICT x, block_turbo3_0 * LM_GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO3 == 0);
    const int nb = k / QK_TURBO3;

    for (int block = 0; block < nb; block++) {
        const float * src = x + block * QK_TURBO3;

        float norm_sq = 0.0f;
        for (int j = 0; j < QK_TURBO3; j++) norm_sq += src[j] * src[j];
        float norm = sqrtf(norm_sq);
        y[block].norm = LM_GGML_FP32_TO_FP16(norm);

        const float inv_norm = (norm > 1e-10f) ? 1.0f / norm : 0.0f;

        memset(y[block].qs, 0, QK_TURBO3 / 4);
        memset(y[block].signs, 0, QK_TURBO3 / 8);

        for (int j = 0; j < QK_TURBO3; j++) {
            float val = src[j] * inv_norm;
            uint8_t idx = (uint8_t)nearest_centroid_3bit(val);
            y[block].qs[j / 4]    |= (uint8_t)((idx & 0x3) << ((j % 4) * 2));
            y[block].signs[j / 8] |= (uint8_t)(((idx >> 2) & 0x1) << (j % 8));
        }
    }
}

void dequantize_row_turbo3_0(const block_turbo3_0 * LM_GGML_RESTRICT x, float * LM_GGML_RESTRICT y, int64_t k) {
    // Stub — Metal shader handles dequant on GPU.
    assert(k % QK_TURBO3 == 0);
    const int nb = k / QK_TURBO3;
    for (int block = 0; block < nb; block++) {
        float norm = LM_GGML_FP16_TO_FP32(x[block].norm);
        for (int j = 0; j < QK_TURBO3; j++) {
            uint8_t low2 = (x[block].qs[j/4] >> ((j%4)*2)) & 0x3;
            uint8_t hi1 = (x[block].signs[j/8] >> (j%8)) & 0x1;
            uint8_t idx = low2 | (hi1 << 2);
            y[block * QK_TURBO3 + j] = CENTROIDS_3BIT[idx] * norm;
        }
    }
}

size_t quantize_turbo3_0(const float * LM_GGML_RESTRICT src, void * LM_GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    LM_GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO3 == 0);

    size_t row_size = (n_per_row / QK_TURBO3) * sizeof(block_turbo3_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo3_0_ref(
            src + row * n_per_row,
            (block_turbo3_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ---------- TURBO4_0: 3-bit PolarQuant + 1-bit QJL ---------- */

void quantize_row_turbo4_0_ref(const float * LM_GGML_RESTRICT x, block_turbo4_0 * LM_GGML_RESTRICT y, int64_t k) {
    turbo_init_rotation();
    turbo_init_qjl();

    assert(k % QK_TURBO4 == 0);
    const int nb = k / QK_TURBO4;
    const int d  = QK_TURBO4;

    for (int block = 0; block < nb; block++) {
        const float * src = x + block * d;

        /* Step 1: Extract norm */
        float norm_sq = 0.0f;
        for (int i = 0; i < d; i++) norm_sq += src[i] * src[i];
        float norm = sqrtf(norm_sq);

        /* Normalize */
        float normalized[TURBO_D];
        if (norm > 1e-10f) {
            const float inv = 1.0f / norm;
            for (int i = 0; i < d; i++) normalized[i] = src[i] * inv;
        } else {
            memset(normalized, 0, d * sizeof(float));
        }

        /* Step 2: Rotate */
        float rotated[TURBO_D];
        matvec(turbo_rotation, normalized, rotated, d);

        /* Step 3: 3-bit quantization */
        uint8_t indices[TURBO_D];
        for (int i = 0; i < d; i++) {
            indices[i] = (uint8_t)nearest_centroid_3bit(rotated[i]);
        }

        /* Step 4: Residual */
        float reconstructed[TURBO_D];
        for (int i = 0; i < d; i++) {
            reconstructed[i] = CENTROIDS_3BIT[indices[i]];
        }
        float mse_recon[TURBO_D];
        matvec(turbo_rotation_t, reconstructed, mse_recon, d);

        float residual[TURBO_D];
        for (int i = 0; i < d; i++) {
            residual[i] = normalized[i] - mse_recon[i];
        }


        /* Step 5: QJL */
        float projected[TURBO_D];
        matvec(turbo_qjl_matrix, residual, projected, d);

        /* Pack */
        y[block].norm  = LM_GGML_FP32_TO_FP16(norm);

        /* Pack 3-bit indices: 8 indices per 3 bytes */
        memset(y[block].qs, 0, d * 3 / 8);
        for (int i = 0; i < d; i++) {
            int bit_offset = i * 3;
            int byte_idx   = bit_offset / 8;
            int bit_pos    = bit_offset % 8;
            uint16_t val   = (uint16_t)(indices[i] & 0x7);
            /* Write up to 2 bytes (3 bits might span a byte boundary) */
            y[block].qs[byte_idx] |= (uint8_t)(val << bit_pos);
            if (bit_pos > 5 && byte_idx + 1 < d * 3 / 8) {
                y[block].qs[byte_idx + 1] |= (uint8_t)(val >> (8 - bit_pos));
            }
        }

        /* Pack 1-bit QJL signs */
        memset(y[block].signs, 0, d / 8);
        for (int i = 0; i < d; i++) {
            if (projected[i] >= 0.0f) {
                y[block].signs[i / 8] |= (1 << (i % 8));
            }
        }
    }
}

void dequantize_row_turbo4_0(const block_turbo4_0 * LM_GGML_RESTRICT x, float * LM_GGML_RESTRICT y, int64_t k) {
    turbo_init_rotation();
    turbo_init_qjl();

    assert(k % QK_TURBO4 == 0);
    const int nb = k / QK_TURBO4;
    const int d  = QK_TURBO4;

    for (int block = 0; block < nb; block++) {
        float norm  = LM_GGML_FP16_TO_FP32(x[block].norm);

        /* Unpack 3-bit indices */
        uint8_t indices[TURBO_D];
        for (int i = 0; i < d; i++) {
            int bit_offset = i * 3;
            int byte_idx   = bit_offset / 8;
            int bit_pos    = bit_offset % 8;
            uint16_t raw   = (uint16_t)x[block].qs[byte_idx];
            if (byte_idx + 1 < d * 3 / 8) {
                raw |= (uint16_t)x[block].qs[byte_idx + 1] << 8;
            }
            indices[i] = (uint8_t)((raw >> bit_pos) & 0x7);
        }

        /* Unpack signs */
        float signs[TURBO_D];
        for (int i = 0; i < d; i++) {
            signs[i] = (x[block].signs[i / 8] & (1 << (i % 8))) ? 1.0f : -1.0f;
        }

        float rnorm = LM_GGML_FP16_TO_FP32(x[block].rnorm);
        const float qjl_scale = TURBO_QJL_CONST / (float)d * rnorm;

        /* PolarQuant dequant */
        float rotated_recon[TURBO_D];
        for (int i = 0; i < d; i++) {
            rotated_recon[i] = CENTROIDS_3BIT[indices[i]];
        }
        float mse_recon[TURBO_D];
        matvec(turbo_rotation_t, rotated_recon, mse_recon, d);

        /* QJL dequant */
        float qjl_recon[TURBO_D];
        matvec(turbo_qjl_matrix_t, signs, qjl_recon, d);
        for (int i = 0; i < d; i++) {
            qjl_recon[i] *= qjl_scale;
        }

        /* Combine */
        float * dst = y + block * d;
        for (int i = 0; i < d; i++) {
            dst[i] = (mse_recon[i] + qjl_recon[i]) * norm;
        }
    }
}

size_t quantize_turbo4_0(const float * LM_GGML_RESTRICT src, void * LM_GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    LM_GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO4 == 0);

    size_t row_size = (n_per_row / QK_TURBO4) * sizeof(block_turbo4_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo4_0_ref(
            src + row * n_per_row,
            (block_turbo4_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ---------- CPU wrappers (void * for lm_ggml_from_float_t) ---------- */

void quantize_row_turbo3_0(const float * LM_GGML_RESTRICT x, void * LM_GGML_RESTRICT y, int64_t k) {
    quantize_row_turbo3_0_ref(x, (block_turbo3_0 *)y, k);
}

void quantize_row_turbo4_0(const float * LM_GGML_RESTRICT x, void * LM_GGML_RESTRICT y, int64_t k) {
    quantize_row_turbo4_0_ref(x, (block_turbo4_0 *)y, k);
}

/* ---------- vec_dot: turbo3 · q8_0 (flash attention CPU path) ---------- */

#if defined(__ARM_NEON) && defined(__aarch64__)

void lm_ggml_vec_dot_turbo3_0_q8_0(int n, float * LM_GGML_RESTRICT s, size_t bs,
        const void * LM_GGML_RESTRICT vx, size_t bx,
        const void * LM_GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_TURBO3 == 0);
    assert(nrc == 1);
    LM_GGML_UNUSED(nrc);
    LM_GGML_UNUSED(bx);
    LM_GGML_UNUSED(by);
    LM_GGML_UNUSED(bs);

    const block_turbo3_0 * LM_GGML_RESTRICT x = (const block_turbo3_0 *)vx;
    const block_q8_0     * LM_GGML_RESTRICT y = (const block_q8_0 *)vy;

    const int nb = n / QK_TURBO3;

    // Load the 8 centroids as a 32-byte table for vqtbl2q_u8 byte lookup.
    // Each centroid is a float32 (4 bytes), so 8 centroids = 32 bytes = 2 x uint8x16.
    const float centroid_arr[8] = {
        CENTROIDS_3BIT[0], CENTROIDS_3BIT[1], CENTROIDS_3BIT[2], CENTROIDS_3BIT[3],
        CENTROIDS_3BIT[4], CENTROIDS_3BIT[5], CENTROIDS_3BIT[6], CENTROIDS_3BIT[7],
    };
    uint8x16x2_t ctbl;
    ctbl.val[0] = vld1q_u8((const uint8_t *)&centroid_arr[0]); // centroids 0-3 as bytes
    ctbl.val[1] = vld1q_u8((const uint8_t *)&centroid_arr[4]); // centroids 4-7 as bytes

    // Byte offset pattern: to look up centroid[idx], we need bytes at idx*4+0,1,2,3.
    // We build index vectors by computing idx*4 and adding {0,1,2,3} per lane.
    const uint8x16_t byte_off = {0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3};

    const uint8x8_t mask2 = vdup_n_u8(0x03);

    float32x4_t global_sum = vdupq_n_f32(0.0f);

    for (int i = 0; i < nb; i++) {
        const float turbo_norm = LM_GGML_FP16_TO_FP32(x[i].norm);
        const float q8_scale   = LM_GGML_FP16_TO_FP32(y[i].d);
        const float block_scale = turbo_norm * q8_scale;

        float32x4_t block_sum = vdupq_n_f32(0.0f);

        // Process 32 elements in 4 groups of 8.
        // qs layout:  8 bytes, qs[j/4] has 4 x 2-bit values (bits 0-1, 2-3, 4-5, 6-7)
        // signs layout: 4 bytes, signs[j/8] has 8 x 1-bit values
        for (int g = 0; g < 4; g++) {
            // --- Unpack 8 x 3-bit indices ---

            // Load 2 bytes from qs (elements g*8..g*8+7, packed 4 per byte)
            const uint8_t qs_byte0 = x[i].qs[g * 2 + 0]; // elements g*8+0..3
            const uint8_t qs_byte1 = x[i].qs[g * 2 + 1]; // elements g*8+4..7

            // Unpack 4 x 2-bit from each byte into a uint8x8 vector of 8 low2 values.
            // byte0: bits [1:0]=elem0, [3:2]=elem1, [5:4]=elem2, [7:6]=elem3
            // byte1: bits [1:0]=elem4, [3:2]=elem5, [5:4]=elem6, [7:6]=elem7
            uint8x8_t low2;
            {
                uint8x8_t v0 = vdup_n_u8(qs_byte0);
                uint8x8_t v1 = vdup_n_u8(qs_byte1);
                // Use negative shifts with vshl for right shift by {0,2,4,6} per element
                const int8x8_t neg_shifts = {0, -2, -4, -6, 0, -2, -4, -6};
                // Select byte0 for lanes 0-3, byte1 for lanes 4-7
                uint8x8_t merged = vext_u8(
                    vreinterpret_u8_s8(vshl_s8(vreinterpret_s8_u8(v0), neg_shifts)),
                    vreinterpret_u8_s8(vshl_s8(vreinterpret_s8_u8(v1), neg_shifts)),
                    4
                );
                low2 = vand_u8(merged, mask2);
            }

            // Load 1 byte from signs (elements g*8..g*8+7, 1 bit each)
            const uint8_t signs_byte = x[i].signs[g];

            // Unpack 8 x 1-bit hi values
            uint8x8_t hi1;
            {
                uint8x8_t sv = vdup_n_u8(signs_byte);
                const int8x8_t neg_shifts = {0, -1, -2, -3, -4, -5, -6, -7};
                hi1 = vand_u8(
                    vreinterpret_u8_s8(vshl_s8(vreinterpret_s8_u8(sv), neg_shifts)),
                    vdup_n_u8(0x01)
                );
            }

            // Combine: idx = low2 | (hi1 << 2), range 0-7
            uint8x8_t idx8 = vorr_u8(low2, vshl_n_u8(hi1, 2));

            // --- Centroid table lookup via vqtbl2q_u8 ---
            // Convert element index to byte offset: idx * 4
            uint8x8_t byte_idx_base = vshl_n_u8(idx8, 2); // idx * 4

            // For first 4 elements: build 16-byte lookup index
            // Lane k (k=0..3): needs bytes at byte_idx_base[k]*1 + {0,1,2,3}
            uint8x16_t bidx_lo, bidx_hi;
            {
                // Widen each of the 4 base indices to fill 4 consecutive lanes
                // e.g., base[0],base[0],base[0],base[0], base[1],base[1],base[1],base[1], ...
                // For elements 0-3 (lower half of idx8)
                uint8x8_t base_lo = byte_idx_base;
                // Duplicate each lane 4 times using zip and ext
                uint8x8x2_t z1 = vzip_u8(base_lo, base_lo); // {b0,b0,b1,b1,b2,b2,b3,b3}, {b4,b4,b5,b5,b6,b6,b7,b7}
                uint8x8x2_t z2_lo = vzip_u8(z1.val[0], z1.val[0]); // {b0,b0,b0,b0,b1,b1,b1,b1}, {b2,b2,b2,b2,b3,b3,b3,b3}
                uint8x8x2_t z2_hi = vzip_u8(z1.val[1], z1.val[1]); // {b4,b4,b4,b4,b5,b5,b5,b5}, {b6,b6,b6,b6,b7,b7,b7,b7}

                bidx_lo = vaddq_u8(vcombine_u8(z2_lo.val[0], z2_lo.val[1]), byte_off);
                bidx_hi = vaddq_u8(vcombine_u8(z2_hi.val[0], z2_hi.val[1]), byte_off);
            }

            // Look up centroid bytes from table
            uint8x16_t cbytes_lo = vqtbl2q_u8(ctbl, bidx_lo);
            uint8x16_t cbytes_hi = vqtbl2q_u8(ctbl, bidx_hi);

            // Reinterpret as float32x4
            float32x4_t cent_lo = vreinterpretq_f32_u8(cbytes_lo); // centroids for elements 0-3
            float32x4_t cent_hi = vreinterpretq_f32_u8(cbytes_hi); // centroids for elements 4-7

            // --- Load 8 x int8 q8 values and convert to float ---
            const int8_t * q8_ptr = y[i].qs + g * 8;
            int8x8_t q8_s8 = vld1_s8(q8_ptr);
            int16x8_t q8_s16 = vmovl_s8(q8_s8);
            float32x4_t q8_f_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q8_s16)));
            float32x4_t q8_f_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q8_s16)));

            // --- Multiply-accumulate ---
            block_sum = vfmaq_f32(block_sum, cent_lo, q8_f_lo);
            block_sum = vfmaq_f32(block_sum, cent_hi, q8_f_hi);
        }

        // Horizontal sum of block_sum * block_scale
        float32x4_t scaled = vmulq_n_f32(block_sum, block_scale);
        global_sum = vaddq_f32(global_sum, scaled);
    }

    // Final horizontal reduction
    *s = vaddvq_f32(global_sum);
}

#else /* scalar fallback for non-ARM platforms */

void lm_ggml_vec_dot_turbo3_0_q8_0(int n, float * LM_GGML_RESTRICT s, size_t bs,
        const void * LM_GGML_RESTRICT vx, size_t bx,
        const void * LM_GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_TURBO3 == 0);
    assert(nrc == 1);
    LM_GGML_UNUSED(nrc);
    LM_GGML_UNUSED(bx);
    LM_GGML_UNUSED(by);
    LM_GGML_UNUSED(bs);

    const block_turbo3_0 * LM_GGML_RESTRICT x = (const block_turbo3_0 *)vx;
    const block_q8_0     * LM_GGML_RESTRICT y = (const block_q8_0 *)vy;

    const int nb = n / QK_TURBO3;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        float turbo_norm = LM_GGML_FP16_TO_FP32(x[i].norm);
        float q8_scale   = LM_GGML_FP16_TO_FP32(y[i].d);

        float block_sum = 0.0f;
        for (int j = 0; j < QK_TURBO3; j++) {
            uint8_t low2 = (x[i].qs[j / 4] >> ((j % 4) * 2)) & 0x3;
            uint8_t hi1  = (x[i].signs[j / 8] >> (j % 8)) & 0x1;
            uint8_t idx  = low2 | (hi1 << 2);

            float centroid = CENTROIDS_3BIT[idx];
            int8_t q8_val  = y[i].qs[j];

            block_sum += centroid * (float)q8_val;
        }

        sumf += turbo_norm * q8_scale * block_sum;
    }

    *s = sumf;
}

#endif /* __ARM_NEON && __aarch64__ */
