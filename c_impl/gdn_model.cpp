#include "gdn_model.h"

#include "hls_stream.h"

#ifdef __SYNTHESIS__
#include <ap_float.h>
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GDN_WEIGHT_HEADER_BYTES 60

/* Compile-time constants for GDN-1.3B (used by on-chip state and parallelism) */
#define GDN_HEADS   8
#define GDN_DK    256   /* head_dim = query/key dimension */
#define GDN_DV    256   /* value_dim = hidden/num_heads   */
#define GDN_RECURRENT_LANES 32 /* recurrent-state column parallelism */
#define GDN_NORM_LANES       8 /* RMS/output-norm arithmetic lanes   */
#define GDN_CONV_LANES       4 /* depthwise-convolution lanes        */
#define GDN_OUTPUT_GATE_LANES 4 /* output norm/gate arithmetic lanes */
#define GDN_SWIGLU_LANES      4 /* independent SwiGLU lanes          */
#define GDN_PK     GDN_RECURRENT_LANES

/* iter19 (step 4): the decode kernel is specialized for the fixed GDN-1.3B shape.
 * These match the .gdnw header (verified by read: vocab 32000, hidden 2048,
 * layers 24, heads 8, v_heads 8, head_dim 256, inter 5632, conv 4, eps 1e-6).
 * Hardcoding them lets gdn_forward drop the runtime `config` pointer and use
 * constant loop bounds -- simplifying the top-level ap_CS_fsm that iter12 flagged
 * as a 95% routing-delay critical path. The host-side helpers keep reading
 * config; only the synthesized kernel path is specialized. */
#define GDN_HIDDEN    2048
#define GDN_LAYERS      24
#define GDN_HEAD_DIM  GDN_DK      /* 256 */
#define GDN_INTER     5632
#define GDN_CONV         4
#define GDN_VOCAB    32000
#define GDN_NORM_EPS  1e-6f       /* .gdnw header 0x358637BD, bit-exact as 1e-6f */
/* Compile-time form of gdn_aux_layer_stride() for the hardcoded shape; must equal
 * H + 2*nh + 2*nh*H + 3*H*cs + hd + H exactly (see gdn_aux_layer_stride). */
#define GDN_AUX_LAYER_STRIDE ((size_t)GDN_HIDDEN + 2*(size_t)GDN_HEADS \
    + 2*(size_t)GDN_HEADS*GDN_HIDDEN + 3*(size_t)GDN_HIDDEN*GDN_CONV \
    + GDN_HEAD_DIM + GDN_HIDDEN)
/* GEMV_CHANNELS lives in gdn_model.h (shared by the kernel and the host shard
 * builder / run-state); do not redefine it here. */

/* step 4 Stage B: the gdn_model.h workspace layout uses literal sizes; assert
 * they equal the GDN_* dims so the two can never silently drift. */
static_assert(GDN_WSF_HID == GDN_HIDDEN, "workspace hidden size drift");
static_assert(GDN_WSF_MLP == GDN_INTER, "workspace mlp size drift");
static_assert(GDN_WSF_STATE ==
    (size_t)GDN_LAYERS*GDN_HEADS*GDN_HEAD_DIM*(GDN_HIDDEN/GDN_HEADS),
    "workspace recurrent_state size drift");
static_assert(GDN_WSF_HEADBUF ==
    (size_t)GDN_LAYERS*3*(GDN_CONV-1)*GDN_HIDDEN, "workspace head_buffer drift");
static_assert(GDN_WSF_LOGITS == GDN_VOCAB, "workspace logits size drift");
static_assert(GDN_WSF_HEAD >= GDN_HEADS, "workspace head padding too small");
static_assert((GDN_WSF_STATE % (GDN_RECURRENT_STATE_PORTS * 32u)) == 0,
    "recurrent state must stripe into whole BF16 Beat512 words");
static_assert((GDN_HIDDEN % 32) == 0, "hidden dimension must pack into BF16 beats");
static_assert((GDN_INTER % 32) == 0, "intermediate dimension must pack into BF16 beats");
static_assert(((4 * GDN_HEAD_DIM) / GEMV_CHANNELS) % 8 == 0,
    "QKVG rows per channel/head must form eight-row groups");
static_assert((GDN_HIDDEN / GEMV_CHANNELS) % 8 == 0,
    "output/down rows per channel must form eight-row groups");
static_assert((GDN_INTER / (GEMV_CHANNELS / 2)) % 8 == 0,
    "gate/up rows per channel must form eight-row groups");
static_assert((GDN_VOCAB / GEMV_CHANNELS) % 8 == 0,
    "LM-head rows per channel must form eight-row groups");
static_assert((GDN_VOCAB / GEMV_CHANNELS) % 16 == 8,
    "LM-head channel-pair stitch assumes an eight-lane tail");
static_assert((GEMV_CHANNELS % 2) == 0,
    "LM-head channel-pair stitch requires paired channels");

union GDNFloatBits {
    float value;
    uint32_t bits;
};

static Fp32Bits fp32_to_bits(float value) {
#pragma HLS inline
    GDNFloatBits converted;
    converted.value = value;
    return Fp32Bits(converted.bits);
}

static float bits_to_fp32(Fp32Bits bits) {
#pragma HLS inline
    GDNFloatBits converted;
    converted.bits = (uint32_t)bits;
    return converted.value;
}

static float get_fp32_lane(const Beat512 &beat, uint32_t lane) {
#pragma HLS inline
#ifdef __SYNTHESIS__
    return bits_to_fp32(beat.range(32 * lane + 31, 32 * lane));
#else
    GDNFloatBits converted;
    memcpy(&converted.bits,
           reinterpret_cast<const unsigned char *>(&beat) + 4 * lane,
           sizeof(converted.bits));
    return converted.value;
#endif
}

static void set_fp32_lane(Beat512 &beat, uint32_t lane, float value) {
#pragma HLS inline
#ifdef __SYNTHESIS__
    beat.range(32 * lane + 31, 32 * lane) = fp32_to_bits(value);
#else
    GDNFloatBits converted;
    converted.value = value;
    memcpy(reinterpret_cast<unsigned char *>(&beat) + 4 * lane,
           &converted.bits, sizeof(converted.bits));
#endif
}

static Bf16Bits get_bf16_lane(const Beat512 &beat, uint32_t lane) {
#pragma HLS inline
#ifdef __SYNTHESIS__
    return beat.range(16 * lane + 15, 16 * lane);
#else
    uint16_t value;
    memcpy(&value,
           reinterpret_cast<const unsigned char *>(&beat) + 2 * lane,
           sizeof(value));
    return Bf16Bits(value);
#endif
}

static void set_bf16_lane(Beat512 &beat, uint32_t lane, Bf16Bits value) {
#pragma HLS inline
#ifdef __SYNTHESIS__
    beat.range(16 * lane + 15, 16 * lane) = value;
#else
    uint16_t native_value = (uint16_t)value;
    memcpy(reinterpret_cast<unsigned char *>(&beat) + 2 * lane,
           &native_value, sizeof(native_value));
#endif
}

/* HLS 2022.2 does not reliably constant-fold `half * 16 + lane` when `half`
 * is a pipelined loop variable.  Applying that index directly to Beat512
 * creates a 512-bit barrel select/read-modify-write cone for every unrolled
 * lane.  Slice the word once, then address only constant lanes inside a
 * partitioned 256-bit half.  The same form reduced the residual adapter from
 * an II-48 wide-word recurrence to a small II-1 pipeline. */
using Bf16Half = ap_uint<256>;

static Bf16Bits get_bf16_half_lane(const Bf16Half &half, uint32_t lane) {
#pragma HLS inline
    return half.range(16 * lane + 15, 16 * lane);
}

static void set_bf16_half_lane(Bf16Half &half, uint32_t lane,
                               Bf16Bits value) {
#pragma HLS inline
    half.range(16 * lane + 15, 16 * lane) = value;
}

static void split_bf16_beat(const Beat512 &beat, Bf16Half half[2]) {
#pragma HLS inline
#pragma HLS array_partition variable=half complete
    half[0] = beat.range(255, 0);
    half[1] = beat.range(511, 256);
}

static Beat512 join_bf16_halves(const Bf16Half half[2]) {
#pragma HLS inline
#pragma HLS array_partition variable=half complete
    Beat512 beat = 0;
    beat.range(255, 0) = half[0];
    beat.range(511, 256) = half[1];
    return beat;
}

static float bf16_to_fp32(Bf16Bits value) {
#pragma HLS inline
    return bits_to_fp32(Fp32Bits(value) << 16);
}

static Bf16Bits fp32_to_bf16_rne(float value) {
#pragma HLS inline
    Fp32Bits bits = fp32_to_bits(value);
    Fp32Bits magnitude = bits & 0x7fffffffu;
    if (magnitude >= 0x7f800000u) {
        if (magnitude == 0x7f800000u)
            return bits.range(31, 16); /* signed infinity */
        return Bf16Bits((bits[31] ? 0xffc0u : 0x7fc0u)); /* canonical qNaN */
    }
    Fp32Bits rounded = bits + 0x7fffu + bits[16];
    return rounded.range(31, 16);
}

#ifndef __SYNTHESIS__
static GDNMixedMulStats gdn_mixed_mul_stats;

void gdn_reset_mixed_mul_stats(void) {
    memset(&gdn_mixed_mul_stats, 0, sizeof(gdn_mixed_mul_stats));
}

GDNMixedMulStats gdn_get_mixed_mul_stats(void) {
    return gdn_mixed_mul_stats;
}

uint16_t gdn_test_fp32_to_bf16_rne_bits(uint32_t bits) {
    GDNFloatBits value;
    value.bits = bits;
    return (uint16_t)fp32_to_bf16_rne(value.value);
}

uint32_t gdn_test_bf16_to_fp32_bits(uint16_t bits) {
    return (uint32_t)fp32_to_bits(bf16_to_fp32(Bf16Bits(bits)));
}

void gdn_test_set_fp32_lane_bits(Beat512 *beat, uint32_t lane, uint32_t bits) {
    GDNFloatBits value;
    value.bits = bits;
    set_fp32_lane(*beat, lane, value.value);
}

uint32_t gdn_test_get_fp32_lane_bits(const Beat512 *beat, uint32_t lane) {
    return (uint32_t)fp32_to_bits(get_fp32_lane(*beat, lane));
}

void gdn_test_set_bf16_lane_bits(Beat512 *beat, uint32_t lane, uint16_t bits) {
    set_bf16_lane(*beat, lane, Bf16Bits(bits));
}

uint16_t gdn_test_get_bf16_lane_bits(const Beat512 *beat, uint32_t lane) {
    return (uint16_t)get_bf16_lane(*beat, lane);
}
#endif

/* Iter66 numerical contract: round each BF16 x BF16 product to BF16 before
 * widening it into the existing FP32 reduction tree. Only synthesis sees
 * ap_float; native execution uses the independently qualified AMD-FPO model
 * below, avoiding an xip_fpo runtime dependency in gdn_eval. */
#ifdef __SYNTHESIS__
using GDNNativeBf16 = ap_float<16, 8>;

static GDNNativeBf16 gdn_native_bf16_from_bits(Bf16Bits bits) {
#pragma HLS inline
    return GDNNativeBf16(
        GDNNativeBf16::sign_t(bits[15]),
        GDNNativeBf16::exponent_t(bits.range(14, 7)),
        GDNNativeBf16::mantissa_t(bits.range(6, 0)));
}

static Bf16Bits gdn_native_bf16_to_bits(GDNNativeBf16 value) {
#pragma HLS inline
    Bf16Bits bits = 0;
    bits[15] = value.sign_ref();
    bits.range(14, 7) = value.exponent_ref();
    bits.range(6, 0) = value.mantissa_ref();
    return bits;
}
#endif

static Bf16Bits gdn_native_bf16_product_bits(Bf16Bits lhs_bits,
                                              Bf16Bits rhs_bits) {
#pragma HLS inline
#ifdef __SYNTHESIS__
    const GDNNativeBf16 lhs = gdn_native_bf16_from_bits(lhs_bits);
    const GDNNativeBf16 rhs = gdn_native_bf16_from_bits(rhs_bits);
    const GDNNativeBf16 product = lhs * rhs;
    return gdn_native_bf16_to_bits(product);
#else
    uint16_t lhs = (uint16_t)lhs_bits;
    uint16_t rhs = (uint16_t)rhs_bits;
    const uint32_t lhs_exponent = (lhs >> 7) & 0xffu;
    const uint32_t rhs_exponent = (rhs >> 7) & 0xffu;
    const uint32_t lhs_fraction = lhs & 0x7fu;
    const uint32_t rhs_fraction = rhs & 0x7fu;
    const bool lhs_nan = lhs_exponent == 0xffu && lhs_fraction != 0;
    const bool rhs_nan = rhs_exponent == 0xffu && rhs_fraction != 0;
    const bool lhs_inf = lhs_exponent == 0xffu && lhs_fraction == 0;
    const bool rhs_inf = rhs_exponent == 0xffu && rhs_fraction == 0;
    const bool lhs_zero = lhs_exponent == 0;
    const bool rhs_zero = rhs_exponent == 0;
    const uint16_t sign = (lhs ^ rhs) & 0x8000u;

    gdn_mixed_mul_stats.calls++;
    if (lhs_nan || rhs_nan || lhs_inf || rhs_inf || lhs_zero || rhs_zero) {
        gdn_mixed_mul_stats.special_inputs++;
        if ((lhs_zero && lhs_fraction != 0) ||
            (rhs_zero && rhs_fraction != 0))
            gdn_mixed_mul_stats.flushed_inputs++;
    }
    if (lhs_nan || rhs_nan || (lhs_inf && rhs_zero) ||
        (rhs_inf && lhs_zero))
        return Bf16Bits(0x7fc0u);
    if (lhs_inf || rhs_inf)
        return Bf16Bits(sign | 0x7f80u);
    if (lhs_zero || rhs_zero)
        return Bf16Bits(sign);

    GDNFloatBits lhs_value;
    GDNFloatBits rhs_value;
    lhs_value.bits = (uint32_t)lhs << 16;
    rhs_value.bits = (uint32_t)rhs << 16;
    volatile float host_product = lhs_value.value * rhs_value.value;
    GDNFloatBits product;
    product.value = host_product;
    const uint32_t magnitude = product.bits & 0x7fffffffu;
    if (magnitude == 0x7f800000u) {
        gdn_mixed_mul_stats.overflows++;
        return Bf16Bits(product.bits >> 16);
    }

    /* AMD Floating-Point Operator DAZ/FTZ behavior: an exponent-minus-one
     * exact product is rescued only when RNE carries normalized significand
     * 0xff to 0x100. The exhaustive 2024.2 qualification measured the
     * equivalent FP32 magnitude boundary as 0x007fc000. */
    if (magnitude < 0x007fc000u) {
        gdn_mixed_mul_stats.flushed_outputs++;
        return Bf16Bits(product.bits >> 16) & Bf16Bits(0x8000u);
    }
    const uint32_t rounded = product.bits + 0x7fffu +
                             ((product.bits >> 16) & 1u);
    const uint16_t result = (uint16_t)(rounded >> 16);
    if ((result & 0x7f80u) == 0) {
        gdn_mixed_mul_stats.flushed_outputs++;
        return Bf16Bits(result & 0x8000u);
    }
    if ((result & 0x7fffu) == 0x7f80u)
        gdn_mixed_mul_stats.overflows++;
    return Bf16Bits(result);
#endif
}

float gdn_native_bf16_mul_to_fp32(Bf16Bits weight, Bf16Bits activation) {
#pragma HLS inline
    return bf16_to_fp32(gdn_native_bf16_product_bits(weight, activation));
}

/* One low/high recurrent-state column pair. Packing both halves into a single
 * 64-bit local-memory word lets four HBM ports advance concurrently without
 * requesting two accesses to the same cyclic URAM bank. */
struct GDNStatePair {
    float lo;
    float hi;
};

#ifndef __SYNTHESIS__
static float *gdn_native_final_hidden_debug = NULL;
static float *gdn_native_logits_debug = NULL;

void gdn_set_native_debug_buffers(float *final_hidden, float *logits) {
    gdn_native_final_hidden_debug = final_hidden;
    gdn_native_logits_debug = logits;
}
#endif

static void gdn_print_error(const char *message) {
#ifdef __SYNTHESIS__
    (void)message;
#else
    fprintf(stderr, "%s\n", message);
#endif
}

static void *gdn_malloc_bytes(size_t bytes) {
    void *ptr = malloc(bytes);
    if (ptr == NULL) {
        gdn_print_error("malloc failed");
    }
    return ptr;
}

static void *gdn_calloc_bytes(size_t count, size_t size) {
    void *ptr = calloc(count, size);
    if (ptr == NULL) {
        gdn_print_error("calloc failed");
    }
    return ptr;
}

static int gdn_read_exact(FILE *file, void *dst, size_t bytes) {
    return fread(dst, 1, bytes, file) == bytes;
}

static size_t gdn_total_weight_floats(const GDNWeightHeader *config) {
    size_t total = 0;
    size_t hidden = config->hidden_size;
    size_t num_heads = config->num_heads;
    size_t head_dim = config->head_dim;
    size_t intermediate = config->intermediate_size;
    size_t vocab = config->vocab_size;
    size_t conv = config->conv_size;
    size_t num_layers = config->num_layers;
    size_t layer_index;

    total += vocab * hidden;
    for (layer_index = 0; layer_index < num_layers; ++layer_index) {
        total += hidden;
        total += num_heads;
        total += num_heads;
        total += hidden * hidden;
        total += hidden * hidden;
        total += hidden * hidden;
        total += num_heads * hidden;
        total += num_heads * hidden;
        total += hidden * conv;
        total += hidden * conv;
        total += hidden * conv;
        total += hidden * hidden;
        total += head_dim;
        total += hidden * hidden;
        total += hidden;
        total += intermediate * hidden;
        total += intermediate * hidden;
        total += hidden * intermediate;
    }
    total += hidden;
    total += vocab * hidden;
    return total;
}

static size_t gdn_layer_weight_stride(const GDNWeightHeader *config) {
    size_t hidden = config->hidden_size;
    size_t num_heads = config->num_heads;
    size_t head_dim = config->head_dim;
    size_t intermediate = config->intermediate_size;
    size_t conv = config->conv_size;

    return hidden +
           num_heads +
           num_heads +
           hidden * hidden +
           hidden * hidden +
           hidden * hidden +
           num_heads * hidden +
           num_heads * hidden +
           hidden * conv +
           hidden * conv +
           hidden * conv +
           hidden * hidden +
           head_dim +
           hidden * hidden +
           hidden +
           intermediate * hidden +
           intermediate * hidden +
           hidden * intermediate;
}

static size_t gdn_layer_weight_offset(const GDNWeightHeader *config, uint32_t layer_index) {
    return (size_t)config->vocab_size * config->hidden_size +
           (size_t)layer_index * gdn_layer_weight_stride(config);
}

static size_t gdn_final_norm_offset(const GDNWeightHeader *config) {
    return gdn_layer_weight_offset(config, config->num_layers);
}

/* ---- Packed-BF16 compact shards for the 32-channel decode GEMV. ---- */
static uint16_t gdn_exact_bf16_bits(float value) {
    GDNFloatBits converted;
    converted.value = value;
    return (uint16_t)(converted.bits >> 16);
}

int gdn_validate_bf16_exact_weights(const float *wd,
                                    const GDNWeightHeader *config) {
    const size_t count = gdn_total_weight_floats(config);
    for (size_t i = 0; i < count; ++i) {
        GDNFloatBits converted;
        converted.value = wd[i];
        if ((converted.bits & 0xffffu) != 0) {
            fprintf(stderr,
                    "gdn: weight %zu is not BF16-exact (bits=0x%08x)\n",
                    i, converted.bits);
            return -1;
        }
    }
    return 0;
}

static size_t gdn_pack_bf16_rows(
    Beat512 *destination,
    const float *source,
    size_t rows,
    size_t columns
) {
    const size_t beats_per_row = columns / 32;
    size_t output = 0;
    for (size_t row_base = 0; row_base < rows; row_base += 8) {
        for (size_t weight_beat = 0;
             weight_beat < beats_per_row; ++weight_beat) {
            for (size_t row_lane = 0; row_lane < 8; ++row_lane) {
                Beat512 packed = 0;
                const float *row = source + (row_base + row_lane) * columns;
                for (uint32_t lane = 0; lane < 32; ++lane) {
                    set_bf16_lane(packed, lane,
                        Bf16Bits(gdn_exact_bf16_bits(
                            row[weight_beat * 32 + lane])));
                }
                destination[output++] = packed;
            }
        }
    }
    return output;
}

static int gdn_validate_bf16_rows(
    const Beat512 *packed,
    const float *source,
    size_t rows,
    size_t columns,
    const char *section,
    uint32_t layer,
    int channel
) {
    const size_t beats_per_row = columns / 32;
    size_t input = 0;
    for (size_t row_base = 0; row_base < rows; row_base += 8) {
        for (size_t weight_beat = 0;
             weight_beat < beats_per_row; ++weight_beat) {
            for (size_t row_lane = 0; row_lane < 8; ++row_lane) {
                const Beat512 word = packed[input++];
                const float *row = source + (row_base + row_lane) * columns;
                for (uint32_t lane = 0; lane < 32; ++lane) {
                    const uint16_t expected = gdn_exact_bf16_bits(
                        row[weight_beat * 32 + lane]);
                    const uint16_t actual =
                        (uint16_t)get_bf16_lane(word, lane);
                    if (actual != expected) {
                        fprintf(stderr,
                                "gdn: packed %s mismatch layer=%u channel=%d "
                                "row=%zu column=%zu expected=0x%04x actual=0x%04x\n",
                                section, layer, channel, row_base + row_lane,
                                weight_beat * 32 + lane, expected, actual);
                        return -1;
                    }
                }
            }
        }
    }
    return 0;
}

size_t gdn_weight_shard_beats(const GDNWeightHeader *config) {
    const size_t H = config->hidden_size;
    const size_t I = config->intermediate_size;
    const size_t V = config->vocab_size;
    const size_t per_layer =
          5 * (H / GEMV_CHANNELS) * (H / 32)
        + 2 * (I / GEMV_CHANNELS) * (H / 32)
        +     (H / GEMV_CHANNELS) * (I / 32);
    return (size_t)config->num_layers * per_layer
         + (V / GEMV_CHANNELS) * (H / 32);
}

size_t gdn_weight_shard_bytes(const GDNWeightHeader *config) {
    return gdn_weight_shard_beats(config) * sizeof(Beat512);
}

void gdn_build_weight_shards(const float *wd, const GDNWeightHeader *config,
                             Beat512 *const shards[]) {
    const size_t H = config->hidden_size;
    const size_t I = config->intermediate_size;
    const size_t nh = config->num_heads;
    const size_t hd = config->head_dim;
    const size_t cs = config->conv_size;
    size_t soff = 0;

    for (uint32_t layer = 0; layer < config->num_layers; ++layer) {
        const size_t base = gdn_layer_weight_offset(config, layer);
        const size_t q = base + H + 2 * nh;
        const size_t k = q + H * H;
        const size_t v = k + H * H;
        const size_t g = v + H * H + 2 * nh * H + 3 * H * cs;
        const size_t o = g + H * H + hd;
        const size_t mg = o + H * H + H;
        const size_t mu = mg + I * H;
        const size_t md = mu + I * H;
        const size_t qkvg_offset[4] = {q, k, v, g};
        const size_t qkvg_rows = (4 * hd) / GEMV_CHANNELS;

        for (int channel = 0; channel < GEMV_CHANNELS; ++channel) {
            const size_t kind = (size_t)channel / (GEMV_CHANNELS / 4);
            const size_t segment = (size_t)channel % (GEMV_CHANNELS / 4);
            size_t destination = soff;
            for (size_t head = 0; head < nh; ++head) {
                const size_t source_row = head * hd + segment * qkvg_rows;
                destination += gdn_pack_bf16_rows(
                    shards[channel] + destination,
                    wd + qkvg_offset[kind] + source_row * H,
                    qkvg_rows, H);
            }
        }
        soff += nh * qkvg_rows * (H / 32);

        const size_t o_rows = H / GEMV_CHANNELS;
        for (int channel = 0; channel < GEMV_CHANNELS; ++channel)
            gdn_pack_bf16_rows(shards[channel] + soff,
                wd + o + (size_t)channel * o_rows * H, o_rows, H);
        soff += o_rows * (H / 32);

        const size_t gu_rows = I / (GEMV_CHANNELS / 2);
        const size_t gu_offset[2] = {mg, mu};
        for (int channel = 0; channel < GEMV_CHANNELS; ++channel) {
            const size_t chunk = (size_t)channel >> 1;
            const size_t kind = (size_t)channel & 1;
            gdn_pack_bf16_rows(shards[channel] + soff,
                wd + gu_offset[kind] + chunk * gu_rows * H,
                gu_rows, H);
        }
        soff += gu_rows * (H / 32);

        const size_t down_rows = H / GEMV_CHANNELS;
        for (int channel = 0; channel < GEMV_CHANNELS; ++channel)
            gdn_pack_bf16_rows(shards[channel] + soff,
                wd + md + (size_t)channel * down_rows * I, down_rows, I);
        soff += down_rows * (I / 32);
    }

    const size_t lm_head = gdn_final_norm_offset(config) + H;
    const size_t lm_rows = config->vocab_size / GEMV_CHANNELS;
    for (int channel = 0; channel < GEMV_CHANNELS; ++channel)
        gdn_pack_bf16_rows(shards[channel] + soff,
            wd + lm_head + (size_t)channel * lm_rows * H, lm_rows, H);
    soff += lm_rows * (H / 32);

    if (soff != gdn_weight_shard_beats(config))
        gdn_print_error("packed weight shard size drift");
}

int gdn_validate_weight_shards(const float *wd, const GDNWeightHeader *config,
                               Beat512 *const shards[]) {
    const size_t H = config->hidden_size;
    const size_t I = config->intermediate_size;
    const size_t nh = config->num_heads;
    const size_t hd = config->head_dim;
    const size_t cs = config->conv_size;
    size_t soff = 0;

    if (H != nh * hd || H % 32 != 0 || I % 32 != 0 ||
        (4 * hd) % GEMV_CHANNELS != 0 || GEMV_CHANNELS % 4 != 0) {
        gdn_print_error("packed BF16 shard geometry is incompatible with the model");
        return -1;
    }

    for (uint32_t layer = 0; layer < config->num_layers; ++layer) {
        const size_t base = gdn_layer_weight_offset(config, layer);
        const size_t q = base + H + 2 * nh;
        const size_t k = q + H * H;
        const size_t v = k + H * H;
        const size_t g = v + H * H + 2 * nh * H + 3 * H * cs;
        const size_t o = g + H * H + hd;
        const size_t mg = o + H * H + H;
        const size_t mu = mg + I * H;
        const size_t md = mu + I * H;
        const size_t qkvg_offset[4] = {q, k, v, g};
        const size_t qkvg_rows = (4 * hd) / GEMV_CHANNELS;

        for (int channel = 0; channel < GEMV_CHANNELS; ++channel) {
            const size_t kind = (size_t)channel / (GEMV_CHANNELS / 4);
            const size_t segment = (size_t)channel % (GEMV_CHANNELS / 4);
            size_t source_offset = soff;
            for (size_t head = 0; head < nh; ++head) {
                const size_t source_row = head * hd + segment * qkvg_rows;
                if (gdn_validate_bf16_rows(
                        shards[channel] + source_offset,
                        wd + qkvg_offset[kind] + source_row * H,
                        qkvg_rows, H, "QKVG", layer, channel) != 0)
                    return -1;
                source_offset += qkvg_rows * (H / 32);
            }
        }
        soff += nh * qkvg_rows * (H / 32);

        const size_t o_rows = H / GEMV_CHANNELS;
        for (int channel = 0; channel < GEMV_CHANNELS; ++channel)
            if (gdn_validate_bf16_rows(shards[channel] + soff,
                    wd + o + (size_t)channel * o_rows * H,
                    o_rows, H, "O", layer, channel) != 0)
                return -1;
        soff += o_rows * (H / 32);

        const size_t gu_rows = I / (GEMV_CHANNELS / 2);
        const size_t gu_offset[2] = {mg, mu};
        for (int channel = 0; channel < GEMV_CHANNELS; ++channel) {
            const size_t chunk = (size_t)channel >> 1;
            const size_t kind = (size_t)channel & 1;
            if (gdn_validate_bf16_rows(shards[channel] + soff,
                    wd + gu_offset[kind] + chunk * gu_rows * H,
                    gu_rows, H, "GU", layer, channel) != 0)
                return -1;
        }
        soff += gu_rows * (H / 32);

        const size_t down_rows = H / GEMV_CHANNELS;
        for (int channel = 0; channel < GEMV_CHANNELS; ++channel)
            if (gdn_validate_bf16_rows(shards[channel] + soff,
                    wd + md + (size_t)channel * down_rows * I,
                    down_rows, I, "MLP-down", layer, channel) != 0)
                return -1;
        soff += down_rows * (I / 32);
    }

    const size_t lm_head = gdn_final_norm_offset(config) + H;
    const size_t lm_rows = config->vocab_size / GEMV_CHANNELS;
    for (int channel = 0; channel < GEMV_CHANNELS; ++channel)
        if (gdn_validate_bf16_rows(shards[channel] + soff,
                wd + lm_head + (size_t)channel * lm_rows * H,
                lm_rows, H, "LM-head", config->num_layers, channel) != 0)
            return -1;
    soff += lm_rows * (H / 32);

    if (soff != gdn_weight_shard_beats(config)) {
        gdn_print_error("packed BF16 shard final size mismatch");
        return -1;
    }
    return 0;
}

int gdn_scatter_recurrent_state(Beat512 *const state_stripes[],
                                const float *recurrent_state,
                                size_t recurrent_state_floats) {
    if ((recurrent_state_floats % GDN_DV) != 0) {
        gdn_print_error("recurrent state is not a whole 256-value row");
        return -1;
    }

    const size_t rows = recurrent_state_floats / GDN_DV;
    for (size_t row = 0; row < rows; ++row) {
        for (uint32_t port = 0; port < GDN_RECURRENT_STATE_PORTS; ++port) {
            const uint32_t island = port & 1u;
            const uint32_t high_half = port >> 1;
            for (uint32_t pair = 0; pair < 2; ++pair) {
                Beat512 packed = 0;
                for (uint32_t subhalf = 0; subhalf < 2; ++subhalf) {
                    for (uint32_t lane = 0; lane < 16; ++lane) {
                        const uint32_t global_v = high_half * 128u
                                                + pair * 64u
                                                + subhalf * 32u
                                                + island * 16u + lane;
                        const size_t source = row * GDN_DV + global_v;
                        GDNFloatBits value;
                        value.value = recurrent_state[source];
                        if ((value.bits & 0xffffu) != 0) {
                            fprintf(stderr,
                                    "gdn: recurrent state %zu is not BF16-exact "
                                    "(bits=0x%08x)\n",
                                    source, value.bits);
                            return -1;
                        }
                        set_bf16_lane(packed, subhalf * 16u + lane,
                                      Bf16Bits(value.bits >> 16));
                    }
                }
                state_stripes[port][row * 2 + pair] = packed;
            }
        }
    }
    return 0;
}

int gdn_pack_conv_tails_bf16(Beat512 *workspace_head_buffer,
                             const float *conv_tails,
                             size_t conv_tail_floats) {
    if (conv_tail_floats != GDN_WSF_HEADBUF) {
        gdn_print_error("convolution-tail size does not match workspace ABI");
        return -1;
    }

    for (size_t beat = 0; beat < GDN_WSF_HEADBUF / 16u; ++beat)
        workspace_head_buffer[beat] = 0;
    for (size_t stripe = 0; stripe < GDN_CONV_TAIL_STRIPES; ++stripe) {
        const size_t source_base = stripe * GDN_CONV_TAIL_FLOATS_PER_STRIPE;
        const size_t destination_base =
            stripe * GDN_CONV_TAIL_RESERVED_BEATS_PER_STRIPE;
        for (uint32_t beat = 0;
             beat < GDN_CONV_TAIL_BF16_BEATS_PER_STRIPE; ++beat) {
            Beat512 packed = 0;
            for (uint32_t lane = 0; lane < 32; ++lane) {
                const size_t source = source_base + beat * 32u + lane;
                GDNFloatBits value;
                value.value = conv_tails[source];
                if ((value.bits & 0xffffu) != 0) {
                    fprintf(stderr,
                            "gdn: convolution tail %zu is not BF16-exact "
                            "(bits=0x%08x)\n",
                            source, value.bits);
                    return -1;
                }
                set_bf16_lane(packed, lane, Bf16Bits(value.bits >> 16));
            }
            workspace_head_buffer[destination_base + beat] = packed;
        }
    }
    return 0;
}

/* Compact non-GEMV weights. The host supplies the selected embedding row in x;
 * all large projection weights and lm_head already live in the 32 GEMV shards.
 * Keeping only these small tensors avoids a second 5.6 GB weight copy. */
static size_t gdn_aux_layer_stride(const GDNWeightHeader *config) {
    size_t H = config->hidden_size;
    size_t nh = config->num_heads;
    size_t hd = config->head_dim;
    size_t cs = config->conv_size;
    return H + 2 * nh + 2 * nh * H + 3 * H * cs + hd + H;
}

size_t gdn_aux_weight_floats(const GDNWeightHeader *config) {
    return (size_t)config->num_layers * gdn_aux_layer_stride(config) +
           config->hidden_size;
}

void gdn_build_aux_weights(const float *wd, const GDNWeightHeader *config,
                           float *aux) {
    size_t H = config->hidden_size;
    size_t nh = config->num_heads;
    size_t hd = config->head_dim;
    size_t cs = config->conv_size;
    size_t dst = 0;

    for (uint32_t layer = 0; layer < config->num_layers; ++layer) {
        size_t src = gdn_layer_weight_offset(config, layer);
        size_t count = H + 2 * nh;
        memcpy(aux + dst, wd + src, count * sizeof(float));
        dst += count;
        src += count + 3 * H * H;

        count = 2 * nh * H + 3 * H * cs;
        memcpy(aux + dst, wd + src, count * sizeof(float));
        dst += count;
        src += count + H * H;

        memcpy(aux + dst, wd + src, hd * sizeof(float));
        dst += hd;
        src += hd + H * H;

        memcpy(aux + dst, wd + src, H * sizeof(float));
        dst += H;
    }

    memcpy(aux + dst, wd + gdn_final_norm_offset(config), H * sizeof(float));
}

static int gdn_validate_config(const GDNWeightHeader *config) {
    if (sizeof(GDNWeightHeader) != GDN_WEIGHT_HEADER_BYTES) {
        gdn_print_error("unexpected weight header size");
        return -1;
    }
    if (memcmp(config->magic, "GDNWv1", 6) != 0 || config->version != 1) {
        gdn_print_error("unsupported weight file");
        return -1;
    }
    if (config->num_heads != config->num_v_heads) {
        gdn_print_error("expected num_heads == num_v_heads");
        return -1;
    }
    if (config->hidden_size != config->num_heads * config->head_dim) {
        gdn_print_error("expected hidden_size == num_heads * head_dim");
        return -1;
    }
    return 0;
}

static void gdn_assign_weight_views(GDNModel *model) {
    size_t offset = 0;
    size_t hidden = model->config.hidden_size;
    size_t num_heads = model->config.num_heads;
    size_t head_dim = model->config.head_dim;
    size_t intermediate = model->config.intermediate_size;
    size_t vocab = model->config.vocab_size;
    size_t conv = model->config.conv_size;
    uint32_t layer_index;

    model->embeddings = model->weight_data + offset;
    offset += vocab * hidden;

    for (layer_index = 0; layer_index < model->config.num_layers; ++layer_index) {
        GDNLayerWeights *layer = &model->layers[layer_index];

        layer->attn_norm = model->weight_data + offset;
        offset += hidden;
        layer->a_log = model->weight_data + offset;
        offset += num_heads;
        layer->dt_bias = model->weight_data + offset;
        offset += num_heads;
        layer->q_proj = model->weight_data + offset;
        offset += hidden * hidden;
        layer->k_proj = model->weight_data + offset;
        offset += hidden * hidden;
        layer->v_proj = model->weight_data + offset;
        offset += hidden * hidden;
        layer->a_proj = model->weight_data + offset;
        offset += num_heads * hidden;
        layer->b_proj = model->weight_data + offset;
        offset += num_heads * hidden;
        layer->q_conv = model->weight_data + offset;
        offset += hidden * conv;
        layer->k_conv = model->weight_data + offset;
        offset += hidden * conv;
        layer->v_conv = model->weight_data + offset;
        offset += hidden * conv;
        layer->g_proj = model->weight_data + offset;
        offset += hidden * hidden;
        layer->o_norm = model->weight_data + offset;
        offset += head_dim;
        layer->o_proj = model->weight_data + offset;
        offset += hidden * hidden;
        layer->mlp_norm = model->weight_data + offset;
        offset += hidden;
        layer->mlp_gate_proj = model->weight_data + offset;
        offset += intermediate * hidden;
        layer->mlp_up_proj = model->weight_data + offset;
        offset += intermediate * hidden;
        layer->mlp_down_proj = model->weight_data + offset;
        offset += hidden * intermediate;
    }

    model->final_norm = model->weight_data + offset;
    offset += hidden;
    model->lm_head = model->weight_data + offset;
}

int gdn_model_load(GDNModel *model, const char *path) {
    FILE *file;
    size_t total_floats;

    memset(model, 0, sizeof(*model));

    file = fopen(path, "rb");
    if (file == NULL) {
        perror("fopen weights");
        return -1;
    }

    if (!gdn_read_exact(file, &model->config, GDN_WEIGHT_HEADER_BYTES)) {
        gdn_print_error("failed to read weight header");
        fclose(file);
        return -1;
    }
    if (gdn_validate_config(&model->config) != 0) {
        fclose(file);
        return -1;
    }

    total_floats = gdn_total_weight_floats(&model->config);
    model->weight_data = (float *)gdn_malloc_bytes(total_floats * sizeof(float));
    model->layers = (GDNLayerWeights *)gdn_calloc_bytes(model->config.num_layers, sizeof(GDNLayerWeights));
    if (model->weight_data == NULL || model->layers == NULL) {
        fclose(file);
        gdn_model_free(model);
        return -1;
    }

    /* Validate while reading so a non-BF16-exact blob (e.g. the retired FP32
     * export) aborts within the first chunk instead of after all 5.6 GB. */
    {
        const size_t chunk_floats = (size_t)4 * 1024 * 1024;
        size_t read_floats = 0;
        while (read_floats < total_floats) {
            size_t request = total_floats - read_floats;
            if (request > chunk_floats)
                request = chunk_floats;
            if (fread(model->weight_data + read_floats, sizeof(float),
                      request, file) != request) {
                gdn_print_error("failed to read weight payload");
                fclose(file);
                gdn_model_free(model);
                return -1;
            }
            for (size_t i = 0; i < request; ++i) {
                GDNFloatBits converted;
                converted.value = model->weight_data[read_floats + i];
                if ((converted.bits & 0xffffu) != 0) {
                    fprintf(stderr,
                            "gdn: weight %zu is not BF16-exact (bits=0x%08x)\n",
                            read_floats + i, converted.bits);
                    gdn_print_error(
                        "dense packing requires a BF16-exact FP32-word checkpoint");
                    fclose(file);
                    gdn_model_free(model);
                    return -1;
                }
            }
            read_floats += request;
        }
    }

    fclose(file);
    gdn_assign_weight_views(model);
    return 0;
}

void gdn_model_free(GDNModel *model) {
    free(model->weight_data);
    free(model->layers);
    memset(model, 0, sizeof(*model));
}

static int gdn_alloc_run_buffer(float **buffer, size_t count) {
    *buffer = (float *)gdn_malloc_bytes(count * sizeof(float));
    return (*buffer == NULL) ? -1 : 0;
}

static int gdn_alloc_beat_buffer(Beat512 **buffer, size_t count) {
    void *allocation = NULL;
    if (posix_memalign(&allocation, 64, count * sizeof(Beat512)) != 0)
        allocation = NULL;
    *buffer = reinterpret_cast<Beat512 *>(allocation);
    if (*buffer == NULL)
        gdn_print_error("64-byte-aligned Beat512 allocation failed");
    return (*buffer == NULL) ? -1 : 0;
}

int gdn_run_state_init(GDNRunState *state, const GDNModel *model, uint32_t max_tokens) {
    memset(state, 0, sizeof(*state));
    if (max_tokens == 0 || max_tokens > model->config.max_seq_len) {
        gdn_print_error("invalid max_tokens for run state");
        return -1;
    }

    /* Decode processes exactly one token per gdn_forward call. Keep the fixed
     * ABI-sized workspace allocation; only the embedding, recurrent state and
     * convolution-tail views are needed by the native driver. */
    if (gdn_alloc_beat_buffer(&state->workspace, GDN_WS_FLOATS / 16) != 0)
        return -1;
    float *workspace_floats = reinterpret_cast<float *>(state->workspace);
    state->x               = workspace_floats + GDN_WS_OFF_X;
    /* Decode persistence: recurrent_state holds ALL layers (24 x 2 MB = 48 MB)
     * and head_buffer is repurposed as the conv tail store: per layer, 3 convs
     * (q/k/v) x (conv_size-1) rows x hidden floats (~1.7 MB). */
    state->recurrent_state = workspace_floats + GDN_WS_OFF_REC_STATE;
    state->head_buffer     = workspace_floats + GDN_WS_OFF_HEAD_BUF;

    /* Stage 2: build the GEMV_CHANNELS compact weight shards (split the gemv
     * projection weights by output stripe) the decode datapath reads in
     * parallel. Same total size as one weight copy — no replication. */
    {
        size_t shard_beats = gdn_weight_shard_beats(&model->config);
        int c;
        for (c = 0; c < GEMV_CHANNELS; ++c) {
            size_t state_extra_beats =
                (c >= GDN_RECURRENT_STATE_FIRST_PORT &&
                 c < GDN_RECURRENT_STATE_FIRST_PORT + GDN_RECURRENT_STATE_PORTS)
                    ? GDN_RECURRENT_STATE_STRIPE_BF16_BEATS : 0;
            if (gdn_alloc_beat_buffer(&state->weight_shards[c],
                                      shard_beats + state_extra_beats) != 0)
                return -1;
        }
        gdn_build_weight_shards(model->weight_data, &model->config, state->weight_shards);
    }
    if (gdn_alloc_run_buffer(&state->aux_weights,
            gdn_aux_weight_floats(&model->config)) != 0) return -1;
    gdn_build_aux_weights(model->weight_data, &model->config, state->aux_weights);

    return 0;
}

void gdn_run_state_free(GDNRunState *state) {
    /* x, recurrent_state and head_buffer are views into workspace. */
    free(state->workspace);
    {
        int c;
        for (c = 0; c < GEMV_CHANNELS; ++c) free(state->weight_shards[c]);
    }
    free(state->aux_weights);
    memset(state, 0, sizeof(*state));
}

static float gdn_sigmoid(float x) {
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    {
        float z = expf(x);
        return z / (1.0f + z);
    }
}

static float gdn_silu(float x) {
    return x * gdn_sigmoid(x);
}

static float gdn_softplus(float x) {
    if (x > 20.0f) {
        return x;
    }
    if (x < -20.0f) {
        return expf(x);
    }
    return log1pf(expf(x));
}

/* Preserve the balanced pairwise reduction order while reusing one pipelined
 * adder per level. The fully unrolled form instantiated 255 adders at each
 * call site even though recurrent attention invokes these reductions
 * sequentially; that density is counterproductive beside the 32-port GEMV. */
static float gdn_tree_reduce_256(const float arr[256]) {
#pragma HLS inline off
    float l128[128];
    float l64[64];
    float l32[32];
    float l16[16];
    float l8[8];
    float l4[4];
    float l2[2];
    #pragma HLS bind_storage variable=l128 type=ram_2p impl=bram
    #pragma HLS bind_storage variable=l64  type=ram_2p impl=bram
    #pragma HLS bind_storage variable=l32  type=ram_2p impl=bram
    #pragma HLS array_partition variable=l16  complete
    #pragma HLS array_partition variable=l8   complete
    #pragma HLS array_partition variable=l4   complete
    #pragma HLS array_partition variable=l2   complete

    uint32_t i;
    L128: for (i = 0; i < 128; ++i) {
    #pragma HLS pipeline II=1
        l128[i] = arr[2*i] + arr[2*i+1];
    }
    L64: for (i = 0; i < 64; ++i) {
    #pragma HLS pipeline II=1
        l64[i] = l128[2*i] + l128[2*i+1];
    }
    L32: for (i = 0; i < 32; ++i) {
    #pragma HLS pipeline II=1
        l32[i] = l64[2*i] + l64[2*i+1];
    }
    L16: for (i = 0; i < 16; ++i) {
    #pragma HLS pipeline II=1
        l16[i] = l32[2*i] + l32[2*i+1];
    }
    L8:   for (i = 0; i < 8;   ++i) { _Pragma("HLS unroll") l8[i]   = l16[2*i]   + l16[2*i+1];   }
    L4:   for (i = 0; i < 4;   ++i) { _Pragma("HLS unroll") l4[i]   = l8[2*i]    + l8[2*i+1];    }
    L2:   for (i = 0; i < 2;   ++i) { _Pragma("HLS unroll") l2[i]   = l4[2*i]    + l4[2*i+1];    }
    return l2[0] + l2[1];
}

/* ============================================================
 * Element-wise helpers vectorised over Beat512 (16 FP32 lanes per beat).
 *
 * These wrap the common load/op/store patterns that used to live as
 * inline scalar loops in gdn_forward / gdn_attn_forward. Going through
 * Beat512 lets HLS use the 512-bit m_axi adapter for one wide read + one
 * wide write per pipeline iteration, instead of one narrow access per
 * float. Profiling on the prior bitstream showed those scalar loops
 * accounting for ~58% of per-layer cycles even though they're trivial
 * arithmetic; this rewrite is the actual fix.
 *
 * Callers pass element counts that are always divisible by 16:
 *   hidden_count = num_tokens × hidden (hidden=2048 ⇒ /16 OK)
 *   mlp_count    = num_tokens × intermediate (intermediate=5632 ⇒ /16 OK)
 * Both source and destination XRT buffers are page-aligned (≥ 4 KiB)
 * by xrt::bo allocation, so Beat512 alignment is satisfied.
 * ============================================================ */

static void gdn_rmsnorm_rows_bf16(
    Beat512 *out,
    const Beat512 *in,
    const float *weight,
    uint32_t num_rows,
    uint32_t num_cols,
    float eps
) {
    /* All transient operator boundaries are BF16. Arithmetic and reductions
     * are FP32; one raw Beat carries the two consecutive 16-value chunks that
     * are accumulated in lower-half then upper-half order. */
    uint32_t col_packs = num_cols / 32;

    /* Buffer the per-channel norm weight once (it is otherwise re-read every
     * row); partitioning follows the independently tuned norm lane count. */
    float w_loc[2048];
    #pragma HLS array_partition variable=w_loc cyclic factor=GDN_NORM_LANES
    rms_load_w: for (uint32_t c = 0; c < num_cols; ++c) {
    #pragma HLS loop_tripcount min=2048 max=2048
    #pragma HLS pipeline II=1
        w_loc[c] = weight[c];
    }

    rmsnorm_row: for (uint32_t row = 0; row < num_rows; ++row) {
    #pragma HLS loop_tripcount min=1 max=2048  /* num_tokens: 1..max_seq_len */
        /* Sum of squares stays FP32, matching the all-BF16 numerical
         * contract while retaining the lower-half/upper-half association. */
        float sum = 0.0f;
        rmsnorm_sq: for (uint32_t cp = 0; cp < col_packs; ++cp) {
        #pragma HLS loop_tripcount min=64 max=64
            Beat512 v = in[(size_t)row * col_packs + cp];
            Bf16Half v_half[2];
            #pragma HLS array_partition variable=v_half complete
            split_bf16_beat(v, v_half);
        rmsnorm_sq_half: for (uint32_t half = 0; half < 2; ++half) {
            #pragma HLS pipeline II=2
                float s = 0.0f;
                const Bf16Half current = v_half[half];
            sq_lane: for (int kk = 0; kk < 16; ++kk) {
            #pragma HLS unroll factor=GDN_NORM_LANES
                    float lane = bf16_to_fp32(
                        get_bf16_half_lane(current, (uint32_t)kk));
                    s += lane * lane;
                }
                sum += s;
            }
        }
        float scale = 1.0f / sqrtf((float)(sum / num_cols) + eps);
        rmsnorm_scale: for (uint32_t cp = 0; cp < col_packs; ++cp) {
        #pragma HLS loop_tripcount min=64 max=64
            Beat512 v = in[(size_t)row * col_packs + cp];
            Bf16Half v_half[2];
            Bf16Half o_half[2];
            #pragma HLS array_partition variable=v_half complete
            #pragma HLS array_partition variable=o_half complete
            split_bf16_beat(v, v_half);
        rmsnorm_scale_half: for (uint32_t half = 0; half < 2; ++half) {
            #pragma HLS pipeline II=2
                const Bf16Half current = v_half[half];
                Bf16Half result = 0;
            scl_lane: for (int kk = 0; kk < 16; ++kk) {
            #pragma HLS unroll factor=GDN_NORM_LANES
                    const uint32_t weight_index =
                        cp * 32u + half * 16u + (uint32_t)kk;
                    const float value = bf16_to_fp32(
                        get_bf16_half_lane(current, (uint32_t)kk));
                    set_bf16_half_lane(result, (uint32_t)kk,
                        fp32_to_bf16_rne(value * scale * w_loc[weight_index]));
                }
                o_half[half] = result;
            }
            out[(size_t)row * col_packs + cp] = join_bf16_halves(o_half);
        }
    }
}


/* Compile-time bounds for gdn_gemv_tiny's on-chip buffers (the a/b gate
 * projections: out_dim = num_heads = 8, in_dim = hidden = 2048). */
#define GDN_GEMV_TINY_OUT_MAX 8
#define GDN_GEMV_TINY_IN_MAX  2048
#define GDN_GEMV_TINY_OUT_LANES 2

static void gdn_gemv_tiny_mm2s(
    const float *weights,
    hls::stream<Beat512> &weight_stream,
    uint32_t in_dim,
    uint32_t out_dim
) {
#pragma HLS inline off
    const Beat512 *weight_words = reinterpret_cast<const Beat512 *>(weights);
    const uint32_t total_words = out_dim * (in_dim / 16);
gvt_mm2s: for (uint32_t i = 0; i < total_words; ++i) {
#pragma HLS loop_tripcount min=1024 max=1024
#pragma HLS pipeline II=1
        weight_stream.write(weight_words[i]);
    }
}

static void gdn_gemv_tiny_compute(
    float *out,
    const Beat512 *in,
    hls::stream<Beat512> &weight_stream,
    uint32_t in_dim,
    uint32_t out_dim
) {
    /* Decode-shape GEMV for the tiny a/b gate projections
     * (in_dim=hidden=2048, out_dim=num_heads=8). Three buffered steps:
     *   1. load the single activation row into resident a_loc (read once);
     *   2. preload all out_dim weight rows to BRAM as one CONTIGUOUS burst (a/b are
     *      [out_dim][in_dim] row-major) — avoids the per-(c,kc) strided HBM reads;
     *   3. one k-pass computing two output rows at a time, each with its own
     *      accumulator + the SAME balanced-tree-per-16-chunk sequential reduction.
     * Bit-exact to the prior per-output reduction (each acc[c] keeps the chunk
     * order); removes the 8x per-output pipeline restart + redundant activation
     * reads that made the prior form ~0.18 ms/call. */
    uint32_t k_packs = in_dim / 16;   /* FP32 auxiliary-weight packs */
    uint32_t activation_packs = in_dim / 32;
    uint32_t c, kc, i;

    /* (1) resident activation — read the token's in[] once, reused by every output */
    float a_loc[GDN_GEMV_TINY_IN_MAX];
    #pragma HLS array_partition variable=a_loc cyclic factor=16
    gvt_la: for (kc = 0; kc < activation_packs; ++kc) {
    #pragma HLS loop_tripcount min=64 max=64
        Beat512 a = in[kc];
        Bf16Half a_half[2];
        #pragma HLS array_partition variable=a_half complete
        split_bf16_beat(a, a_half);
    gvt_la_half: for (uint32_t half = 0; half < 2; ++half) {
        #pragma HLS pipeline II=1
            const Bf16Half current = a_half[half];
        gvt_la_i: for (i = 0; i < 16; ++i) {
        #pragma HLS unroll
                a_loc[kc * 32 + half * 16 + i] = bf16_to_fp32(
                    get_bf16_half_lane(current, i));
            }
        }
    }

    /* (2) preload weights to BRAM — one contiguous burst over [out_dim][in_dim] */
    float w_loc[GDN_GEMV_TINY_OUT_MAX][GDN_GEMV_TINY_IN_MAX];
    #pragma HLS array_partition variable=w_loc dim=1 cyclic factor=GDN_GEMV_TINY_OUT_LANES
    #pragma HLS array_partition variable=w_loc dim=2 cyclic factor=16
    gvt_lw_c: for (c = 0; c < out_dim; ++c) {
    #pragma HLS loop_tripcount min=8 max=8
        gvt_lw_kc: for (kc = 0; kc < k_packs; ++kc) {
        #pragma HLS loop_tripcount min=128 max=128
        #pragma HLS pipeline II=1
            Beat512 w = weight_stream.read();
            gvt_lw_i: for (i = 0; i < 16; ++i) {
            #pragma HLS unroll
                w_loc[c][kc * 16 + i] = get_fp32_lane(w, i);
            }
        }
    }

    /* (3) one k-pass with two physical output lanes. Keep each output's kc
     * accumulation order unchanged, but pipeline groups of independent outputs
     * instead of pipelining gvt_k. Pipelining gvt_k made Vitis completely
     * unroll gvt_c (all 8 outputs), ignoring its partial-unroll factor. */
    float acc[GDN_GEMV_TINY_OUT_MAX];
    #pragma HLS array_partition variable=acc cyclic factor=GDN_GEMV_TINY_OUT_LANES
    gvt_init: for (c = 0; c < out_dim; ++c) {
    #pragma HLS unroll factor=GDN_GEMV_TINY_OUT_LANES
        acc[c] = 0.0f;
    }
    gvt_k: for (kc = 0; kc < k_packs; ++kc) {
    #pragma HLS loop_tripcount min=128 max=128
        gvt_c_group: for (uint32_t cg = 0;
                          cg < GDN_GEMV_TINY_OUT_MAX;
                          cg += GDN_GEMV_TINY_OUT_LANES) {
        #pragma HLS loop_tripcount min=4 max=4
        #pragma HLS pipeline II=2
            gvt_c_lane: for (uint32_t co = 0;
                             co < GDN_GEMV_TINY_OUT_LANES; ++co) {
            #pragma HLS unroll
                uint32_t c_lane = cg + co;
                if (c_lane < out_dim) {
                    float p[16];
                    #pragma HLS array_partition variable=p complete
                    gvt_mul: for (i = 0; i < 16; ++i) {
                    #pragma HLS unroll
                        p[i] = a_loc[kc * 16 + i]
                             * w_loc[c_lane][kc * 16 + i];
                    }
                    /* Same balanced 4-level tree as before. */
                    float s2_0 = p[0]  + p[1],  s2_1 = p[2]  + p[3];
                    float s2_2 = p[4]  + p[5],  s2_3 = p[6]  + p[7];
                    float s2_4 = p[8]  + p[9],  s2_5 = p[10] + p[11];
                    float s2_6 = p[12] + p[13], s2_7 = p[14] + p[15];
                    float s4_0 = s2_0 + s2_1, s4_1 = s2_2 + s2_3;
                    float s4_2 = s2_4 + s2_5, s4_3 = s2_6 + s2_7;
                    float s8_0 = s4_0 + s4_1, s8_1 = s4_2 + s4_3;
                    acc[c_lane] += s8_0 + s8_1;
                }
            }
        }
    }
    gvt_st: for (c = 0; c < out_dim; ++c) {
    #pragma HLS unroll factor=GDN_GEMV_TINY_OUT_LANES
        /* Tiny-GEMV is an operator boundary in the all-BF16 contract. Keep the
         * scalar interface for the recurrent actor, but store a BF16-exact
         * value in that FP32 slot. */
        out[c] = bf16_to_fp32(fp32_to_bf16_rne(acc[c]));
    }
}

static void gdn_gemv_tiny(
    float *out,
    const Beat512 *in,
    const float *weights,
    uint32_t in_dim,
    uint32_t out_dim
) {
#pragma HLS inline off
    hls::stream<Beat512> weight_stream;
#pragma HLS stream variable=weight_stream depth=64
#pragma HLS bind_storage variable=weight_stream type=fifo impl=bram
#pragma HLS dataflow disable_start_propagation
    gdn_gemv_tiny_mm2s(weights, weight_stream, in_dim, out_dim);
    gdn_gemv_tiny_compute(out, in, weight_stream, in_dim, out_dim);
}

/* No dispatch wrapper — gdn_forward calls gdn_gemv directly for the large
 * decode projections (defined below) and gdn_gemv_tiny for the small
 * a/b_proj shapes (out_dim=8). Direct calls let HLS allocate and report only
 * the path actually used. */

static void gdn_depthwise_conv_silu_head_kind(
    Beat512 head_out[GDN_HEAD_DIM / 32],
    const Beat512 head_value[4][GDN_HEAD_DIM / 32],
    const Beat512 conv_weights[3][(GDN_HIDDEN * GDN_CONV) / 16],
    const Beat512 conv_tails[3][((GDN_CONV - 1) * GDN_HIDDEN) / 32],
    uint32_t head,
    uint32_t kind
) {
#pragma HLS inline off
#pragma HLS aggregate variable=conv_weights compact=bit
#pragma HLS aggregate variable=conv_tails compact=bit
    float w_loc[GDN_HEAD_DIM][GDN_CONV];
#pragma HLS array_partition variable=w_loc dim=2 complete
#pragma HLS array_partition variable=w_loc dim=1 cyclic factor=GDN_CONV_LANES
    float in_window[GDN_CONV][GDN_HEAD_DIM];
#pragma HLS array_partition variable=in_window dim=1 complete
#pragma HLS array_partition variable=in_window dim=2 cyclic factor=GDN_CONV_LANES

    const uint32_t head_packs = GDN_HEAD_DIM / 32;
    const uint32_t hidden_packs = GDN_HIDDEN / 32;
    const uint32_t weight_packs_per_head =
        (GDN_HEAD_DIM * GDN_CONV) / 16;

iter39_head_conv_load_w: for (uint32_t wb = 0;
                              wb < weight_packs_per_head; ++wb) {
#pragma HLS loop_tripcount min=64 max=64
#pragma HLS pipeline II=1
        Beat512 wp = conv_weights[kind][head * weight_packs_per_head + wb];
    iter39_head_conv_load_w_col: for (uint32_t col4 = 0; col4 < 4; ++col4) {
#pragma HLS unroll
        iter39_head_conv_load_w_k: for (uint32_t k = 0; k < GDN_CONV; ++k) {
#pragma HLS unroll
                w_loc[wb * 4 + col4][k] =
                    get_fp32_lane(wp, col4 * GDN_CONV + k);
            }
        }
    }

iter39_head_conv_restore_k: for (uint32_t k = 0; k + 1 < GDN_CONV; ++k) {
    iter39_head_conv_restore_p: for (uint32_t p = 0; p < head_packs; ++p) {
#pragma HLS loop_tripcount min=8 max=8
            Beat512 value = conv_tails[kind][(size_t)k * hidden_packs
                                           + head * head_packs + p];
            Bf16Half value_half[2];
            #pragma HLS array_partition variable=value_half complete
            split_bf16_beat(value, value_half);
        iter39_head_conv_restore_half: for (uint32_t half = 0;
                                             half < 2; ++half) {
#pragma HLS pipeline II=1
                const Bf16Half current = value_half[half];
            iter39_head_conv_restore_lane: for (uint32_t lane = 0;
                                                lane < 16; ++lane) {
#pragma HLS unroll
                    const uint32_t packed_lane = half * 16u + lane;
                    in_window[k + 1][p * 32u + packed_lane] =
                        bf16_to_fp32(get_bf16_half_lane(current, lane));
                }
            }
        }
    }

iter39_head_conv_shift: for (uint32_t p = 0; p < head_packs; ++p) {
#pragma HLS loop_tripcount min=8 max=8
        Beat512 value = head_value[kind][p];
        Bf16Half value_half[2];
        #pragma HLS array_partition variable=value_half complete
        split_bf16_beat(value, value_half);
    iter39_head_conv_shift_half: for (uint32_t half = 0; half < 2; ++half) {
#pragma HLS pipeline II=1
            const Bf16Half current = value_half[half];
        iter39_head_conv_shift_lane: for (uint32_t lane = 0; lane < 16; ++lane) {
#pragma HLS unroll
                const uint32_t packed_lane = half * 16u + lane;
                uint32_t col = p * 32u + packed_lane;
                in_window[0][col] = in_window[1][col];
                in_window[1][col] = in_window[2][col];
                in_window[2][col] = in_window[3][col];
                in_window[3][col] =
                    bf16_to_fp32(get_bf16_half_lane(current, lane));
            }
        }
    }

iter39_head_conv_compute: for (uint32_t p = 0; p < head_packs; ++p) {
#pragma HLS loop_tripcount min=8 max=8
        float o_lane[32];
#pragma HLS array_partition variable=o_lane complete
    iter39_head_conv_group: for (uint32_t base = 0; base < 32;
                                 base += GDN_CONV_LANES) {
#pragma HLS loop_tripcount min=8 max=8
#pragma HLS pipeline II=1
        iter39_head_conv_lane: for (uint32_t lane = 0;
                                    lane < GDN_CONV_LANES; ++lane) {
#pragma HLS unroll
                uint32_t out_lane = base + lane;
                uint32_t col = p * 32u + out_lane;
                float sum = in_window[0][col] * w_loc[col][0]
                          + in_window[1][col] * w_loc[col][1]
                          + in_window[2][col] * w_loc[col][2]
                          + in_window[3][col] * w_loc[col][3];
                o_lane[out_lane] = gdn_silu(sum);
            }
        }
        Beat512 result = 0;
    iter39_head_conv_pack: for (uint32_t lane = 0; lane < 32; ++lane) {
#pragma HLS unroll
            set_bf16_lane(result, lane, fp32_to_bf16_rne(o_lane[lane]));
        }
        head_out[p] = result;
    }

}

/* Keep the shared mem_weights_mm0 AXI master outside the QKVG dataflow region:
 * Vitis HLS permits only one reader process for a bundled m_axi interface.
 * These fixed-trip transfers stage exactly the three convolution tensors and
 * persistent tails needed by the bounded head consumer. */
static void gdn_read_qkvg_conv_context(
    hls::stream<Beat512> &context,
    const float *q_weights,
    const float *k_weights,
    const float *v_weights,
    const Beat512 *q_tail,
    const Beat512 *k_tail,
    const Beat512 *v_tail
) {
#pragma HLS inline off
    const Beat512 *q_weight_words = reinterpret_cast<const Beat512 *>(q_weights);
    const Beat512 *k_weight_words = reinterpret_cast<const Beat512 *>(k_weights);
    const Beat512 *v_weight_words = reinterpret_cast<const Beat512 *>(v_weights);
qkvg_context_read_q_weight: for (uint32_t p = 0; p < 512; ++p) {
#pragma HLS pipeline II=1
        context.write(q_weight_words[p]);
    }
qkvg_context_read_k_weight: for (uint32_t p = 0; p < 512; ++p) {
#pragma HLS pipeline II=1
        context.write(k_weight_words[p]);
    }
qkvg_context_read_v_weight: for (uint32_t p = 0; p < 512; ++p) {
#pragma HLS pipeline II=1
        context.write(v_weight_words[p]);
    }
qkvg_context_read_q_tail: for (uint32_t p = 0; p < 192; ++p) {
#pragma HLS pipeline II=1
        context.write(q_tail[p]);
    }
qkvg_context_read_k_tail: for (uint32_t p = 0; p < 192; ++p) {
#pragma HLS pipeline II=1
        context.write(k_tail[p]);
    }
qkvg_context_read_v_tail: for (uint32_t p = 0; p < 192; ++p) {
#pragma HLS pipeline II=1
        context.write(v_tail[p]);
    }
}

static void gdn_store_qkvg_conv_context_stream(
    Beat512 conv_weights[3][(GDN_HIDDEN * GDN_CONV) / 16],
    Beat512 conv_tails[3][((GDN_CONV - 1) * GDN_HIDDEN) / 32],
    hls::stream<Beat512> &context
) {
#pragma HLS inline off
#pragma HLS aggregate variable=conv_weights compact=bit
#pragma HLS aggregate variable=conv_tails compact=bit

iter39_context_q_weight: for (uint32_t p = 0;
                              p < (GDN_HIDDEN * GDN_CONV) / 16; ++p) {
#pragma HLS loop_tripcount min=512 max=512
#pragma HLS pipeline II=1
        conv_weights[0][p] = context.read();
    }
iter39_context_k_weight: for (uint32_t p = 0;
                              p < (GDN_HIDDEN * GDN_CONV) / 16; ++p) {
#pragma HLS loop_tripcount min=512 max=512
#pragma HLS pipeline II=1
        conv_weights[1][p] = context.read();
    }
iter39_context_v_weight: for (uint32_t p = 0;
                              p < (GDN_HIDDEN * GDN_CONV) / 16; ++p) {
#pragma HLS loop_tripcount min=512 max=512
#pragma HLS pipeline II=1
        conv_weights[2][p] = context.read();
    }
iter39_context_q_tail: for (uint32_t p = 0;
                            p < ((GDN_CONV - 1) * GDN_HIDDEN) / 32; ++p) {
#pragma HLS loop_tripcount min=192 max=192
#pragma HLS pipeline II=1
        conv_tails[0][p] = context.read();
    }
iter39_context_k_tail: for (uint32_t p = 0;
                            p < ((GDN_CONV - 1) * GDN_HIDDEN) / 32; ++p) {
#pragma HLS loop_tripcount min=192 max=192
#pragma HLS pipeline II=1
        conv_tails[1][p] = context.read();
    }
iter39_context_v_tail: for (uint32_t p = 0;
                            p < ((GDN_CONV - 1) * GDN_HIDDEN) / 32; ++p) {
#pragma HLS loop_tripcount min=192 max=192
#pragma HLS pipeline II=1
        conv_tails[2][p] = context.read();
    }
}

static void gdn_load_qkvg_conv_context(
    Beat512 conv_weights[3][(GDN_HIDDEN * GDN_CONV) / 16],
    Beat512 conv_tails[3][((GDN_CONV - 1) * GDN_HIDDEN) / 32],
    const float *q_weights,
    const float *k_weights,
    const float *v_weights,
    const Beat512 *q_tail,
    const Beat512 *k_tail,
    const Beat512 *v_tail
) {
#pragma HLS inline off
    hls::stream<Beat512> context;
#pragma HLS stream variable=context depth=64
#pragma HLS bind_storage variable=context type=fifo impl=bram
#pragma HLS dataflow disable_start_propagation
    gdn_read_qkvg_conv_context(context, q_weights, k_weights, v_weights,
                               q_tail, k_tail, v_tail);
    gdn_store_qkvg_conv_context_stream(conv_weights, conv_tails, context);
}

static void gdn_store_qkvg_conv_tails(
    Beat512 *q_tail,
    Beat512 *k_tail,
    Beat512 *v_tail,
    const Beat512 conv_tails[3][((GDN_CONV - 1) * GDN_HIDDEN) / 32]
) {
#pragma HLS inline off
#pragma HLS aggregate variable=conv_tails compact=bit
    const uint32_t hidden_packs = GDN_HIDDEN / 32;
    const uint32_t tail_packs = ((GDN_CONV - 1) * GDN_HIDDEN) / 32;

iter39_context_store_q: for (uint32_t p = 0; p < tail_packs; ++p) {
#pragma HLS loop_tripcount min=192 max=192
#pragma HLS pipeline II=1
        uint32_t source = (p < 2 * hidden_packs)
                        ? p + hidden_packs : p - 2 * hidden_packs;
        q_tail[p] = conv_tails[0][source];
    }
iter39_context_store_k: for (uint32_t p = 0; p < tail_packs; ++p) {
#pragma HLS loop_tripcount min=192 max=192
#pragma HLS pipeline II=1
        uint32_t source = (p < 2 * hidden_packs)
                        ? p + hidden_packs : p - 2 * hidden_packs;
        k_tail[p] = conv_tails[1][source];
    }
iter39_context_store_v: for (uint32_t p = 0; p < tail_packs; ++p) {
#pragma HLS loop_tripcount min=192 max=192
#pragma HLS pipeline II=1
        uint32_t source = (p < 2 * hidden_packs)
                        ? p + hidden_packs : p - 2 * hidden_packs;
        v_tail[p] = conv_tails[2][source];
    }
}

static void gdn_read_recurrent_scalar_word(
    const float *layer_a_log,
    hls::stream<Beat512> &scalar_word
) {
#pragma HLS inline off
    const Beat512 *source = reinterpret_cast<const Beat512 *>(layer_a_log);
    scalar_word.write(source[0]);
}

static void gdn_store_recurrent_scalar_word(
    hls::stream<Beat512> &scalar_word,
    float a_log_storage[GDN_HEADS],
    float dt_bias_storage[GDN_HEADS]
) {
#pragma HLS inline off
    Beat512 values = scalar_word.read();
recur_scalar_local_lane: for (uint32_t head = 0; head < GDN_HEADS; ++head) {
#pragma HLS unroll
        a_log_storage[head] = get_fp32_lane(values, head);
        dt_bias_storage[head] = get_fp32_lane(values, GDN_HEADS + head);
    }
}

static void gdn_load_recurrent_scalars(
    float a_log_storage[GDN_HEADS],
    float dt_bias_storage[GDN_HEADS],
    const float *layer_a_log
) {
#pragma HLS inline off
    hls::stream<Beat512> scalar_word;
#pragma HLS stream variable=scalar_word depth=2
#pragma HLS dataflow disable_start_propagation
    gdn_read_recurrent_scalar_word(layer_a_log, scalar_word);
    gdn_store_recurrent_scalar_word(scalar_word,
                                    a_log_storage, dt_bias_storage);
}

/* -----------------------------------------------------------------------
 * Frequency-oriented recurrent islands. The packed state layout has a
 * natural two-way physical cut; each actor owns 16 columns from two ports,
 * retaining 32 aggregate MAC lanes without a crossbar.
 *
 * Algebraic fusion (Gupta et al.):
 *   S_new = g*S_old + k_norm * Δv^T
 *   o = (1/√d) * S_new^T * q_norm
 *     = (1/√d) * (g * S_old^T * q_norm + (q_norm^T * k_norm) * Δv)
 *     = q_scale * (g * ô + α * Δv)
 * where ô_i = Σ_j S_old[j][i]*q_norm[j], α = q_norm^T * k_norm.
 * This lets us compute retrieval (r) and partial output (ô) in a single
 * read pass, then apply a scalar correction, reducing state passes from
 * 4 to 2.
 * ----------------------------------------------------------------------- */
#define GDN_RECURRENT_ISLAND_LANES 16
#define GDN_RECURRENT_ISLAND_COLS  (GDN_DV / 4)

static void gdn_recurrent_duplicate_qkv(
    hls::stream<Beat512> &q_in,
    hls::stream<Beat512> &k_in,
    hls::stream<Beat512> &v_in,
    hls::stream<Beat512> &q0,
    hls::stream<Beat512> &k0,
    hls::stream<Beat512> &v0,
    hls::stream<Beat512> &q1,
    hls::stream<Beat512> &k1,
    hls::stream<Beat512> &v1
) {
#pragma HLS inline off
recur_island_broadcast_head: for (uint32_t head = 0;
                                  head < GDN_HEADS; ++head) {
#pragma HLS loop_tripcount min=8 max=8
    recur_island_broadcast_pack: for (uint32_t p = 0;
                                      p < GDN_DK / 32; ++p) {
#pragma HLS loop_tripcount min=8 max=8
#pragma HLS pipeline II=1
            Beat512 qv = q_in.read();
            Beat512 kv = k_in.read();
            Beat512 vv = v_in.read();
            q0.write(qv);
            k0.write(kv);
            v0.write(vv);
            q1.write(qv);
            k1.write(kv);
            v1.write(vv);
        }
    }
}

#ifndef __SYNTHESIS__
/* Iter66m diagnostic: append one per-head scalar-chain record to the file
 * named by GDN_HEAD_SCALAR_DUMP. Native-only; never synthesized. */
static void gdn_debug_record_head_scalars(uint32_t layer, uint32_t head,
                                          int island, float a_val, float b_val,
                                          float a_log, float dt_bias,
                                          float decay, float beta) {
    const char *path = getenv("GDN_HEAD_SCALAR_DUMP");
    if (path == NULL) return;
    FILE *out = fopen(path, "a");
    if (out == NULL) return;
    GDNFloatBits pack[6];
    pack[0].value = a_val;
    pack[1].value = b_val;
    pack[2].value = a_log;
    pack[3].value = dt_bias;
    pack[4].value = decay;
    pack[5].value = beta;
    fprintf(out, "%u %u %d 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x\n",
            layer, head, island, pack[0].bits, pack[1].bits, pack[2].bits,
            pack[3].bits, pack[4].bits, pack[5].bits);
    fclose(out);
}

/* Iter66l diagnostic helper: shift a float by N ULP, N read once from the
 * named environment variable. Native-only; never synthesized. */
static float gdn_debug_nudge_ulp(float value, const char *env_name) {
    const char *setting = getenv(env_name);
    if (setting == NULL) return value;
    const int steps = atoi(setting);
    if (steps == 0) return value;
    GDNFloatBits raw;
    raw.value = value;
    raw.bits = (uint32_t)((int64_t)raw.bits + steps);
    return raw.value;
}
#endif

template <int ISLAND>
static void gdn_recurrent_attention_island(
    hls::stream<Beat512> &q_stream,
    hls::stream<Beat512> &k_stream,
    hls::stream<Beat512> &v_stream,
    hls::stream<Beat512> &state_low_stream,
    hls::stream<Beat512> &state_high_stream,
    hls::stream<Beat512> &out_stream,
    Beat512 *recurrent_state_low,
    Beat512 *recurrent_state_high,
    const float *a,
    const float *b,
    const float *layer_a_log,
    const float *layer_dt_bias,
    uint32_t layer_index
) {
#pragma HLS inline off
    GDNStatePair state_pair[GDN_DK][GDN_RECURRENT_ISLAND_COLS];
#pragma HLS aggregate variable=state_pair compact=bit
#pragma HLS bind_storage variable=state_pair type=RAM_2P impl=URAM
#pragma HLS array_partition variable=state_pair dim=2 cyclic factor=GDN_RECURRENT_ISLAND_LANES

    Beat512 *state_out_low = recurrent_state_low;
    Beat512 *state_out_high = recurrent_state_high;
    const float q_scale = 1.0f / sqrtf((float)GDN_DK);

recur_island_head: for (uint32_t head_index = 0;
                        head_index < GDN_HEADS; ++head_index) {
#pragma HLS loop_tripcount min=8 max=8
        Beat512 q_head[GDN_DK / 32];
        Beat512 k_head[GDN_DK / 32];
        Beat512 v_head[GDN_DV / 32];
#pragma HLS aggregate variable=q_head compact=bit
#pragma HLS aggregate variable=k_head compact=bit
#pragma HLS aggregate variable=v_head compact=bit

    recur_island_load_qkv: for (uint32_t p = 0;
                                p < GDN_DK / 32; ++p) {
#pragma HLS loop_tripcount min=8 max=8
#pragma HLS pipeline II=1
            q_head[p] = q_stream.read();
            k_head[p] = k_stream.read();
            v_head[p] = v_stream.read();
        }

        const size_t head_state_base_bf16 =
            ((size_t)layer_index * GDN_HEADS + head_index) *
            GDN_DK * (GDN_DV / 32) / GDN_RECURRENT_STATE_PORTS;

        /* One BF16 state Beat contains the two 16-column chunks that used to
         * arrive as two FP32 Beats.  A nested 512-Beat/2-half loop looked
         * compact in C, but HLS scheduled the outer loop sequentially at seven
         * cycles per Beat (3,584 cycles/head).  Flatten the logical halves and
         * cache only the upper 256 bits: one stream read on each even
         * transaction, one URAM-bank write on every transaction, II=1.  The
         * resulting 1,024 cycles/head exactly restores Iter61's state-load
         * service time while retaining half the HBM traffic. */
        Bf16Half cached_low_upper = 0;
        Bf16Half cached_high_upper = 0;
    recur_island_load_state: for (uint32_t transaction = 0;
                                  transaction < GDN_DK * 4; ++transaction) {
#pragma HLS loop_tripcount min=1024 max=1024
#pragma HLS pipeline II=1
#pragma HLS dependence variable=state_pair inter false
            const uint32_t block = transaction >> 1;
            const uint32_t subhalf = transaction & 1;
            const uint32_t row = block >> 1;
            const uint32_t pair = block & 1;
            Bf16Half current_low;
            Bf16Half current_high;
            if (subhalf == 0) {
                const Beat512 state_low = state_low_stream.read();
                const Beat512 state_high = state_high_stream.read();
                current_low = state_low.range(255, 0);
                current_high = state_high.range(255, 0);
                cached_low_upper = state_low.range(511, 256);
                cached_high_upper = state_high.range(511, 256);
            } else {
                current_low = cached_low_upper;
                current_high = cached_high_upper;
            }
            const uint32_t local_base = pair * 32u + subhalf * 16u;
        recur_island_load_state_lane: for (uint32_t lane = 0;
                                            lane < GDN_RECURRENT_ISLAND_LANES;
                                            ++lane) {
#pragma HLS unroll
                GDNStatePair value;
                value.lo = bf16_to_fp32(
                    get_bf16_half_lane(current_low, lane));
                value.hi = bf16_to_fp32(
                    get_bf16_half_lane(current_high, lane));
                state_pair[row][local_base + lane] = value;
            }
        }

        float q_loc[GDN_DK];
        float k_loc[GDN_DK];
        float v_loc[GDN_DV];
        float qsq_arr[GDN_DK];
        float ksq_arr[GDN_DK];
        float alpha_prod[GDN_DK];
#pragma HLS array_partition variable=qsq_arr cyclic factor=2
#pragma HLS array_partition variable=ksq_arr cyclic factor=2
#pragma HLS array_partition variable=alpha_prod cyclic factor=2
#pragma HLS bind_storage variable=qsq_arr type=ram_2p impl=bram
#pragma HLS bind_storage variable=ksq_arr type=ram_2p impl=bram
#pragma HLS bind_storage variable=alpha_prod type=ram_2p impl=bram

    recur_island_load_qk: for (uint32_t j = 0; j < GDN_DK; ++j) {
#pragma HLS loop_tripcount min=256 max=256
#pragma HLS pipeline II=1
            float qj = bf16_to_fp32(get_bf16_lane(q_head[j >> 5], j & 31));
            float kj = bf16_to_fp32(get_bf16_lane(k_head[j >> 5], j & 31));
            q_loc[j] = qj;
            k_loc[j] = kj;
            qsq_arr[j] = qj * qj;
            ksq_arr[j] = kj * kj;
        }

        float q_sq = gdn_tree_reduce_256(qsq_arr);
        float k_sq = gdn_tree_reduce_256(ksq_arr);
        float q_inv = 1.0f / sqrtf(q_sq + 1e-6f);
        float k_inv = 1.0f / sqrtf(k_sq + 1e-6f);
#ifndef __SYNTHESIS__
        /* Iter66l diagnostic (native only, env-gated, no synthesis effect):
         * nudge one per-head scalar by N FP32 ULP and compare the resulting
         * state-dump fingerprint against the measured hardware-vs-native one
         * (130 lanes; 120/7/1/2 at 1/2/3/4 BF16 ULP; uniform intra-head
         * scatter). The arithmetic audit cleared every primitive, so the
         * remaining candidate is a per-head scalar whose reduction chain
         * diverges; this identifies which scalar and at what magnitude. */
        k_inv = gdn_debug_nudge_ulp(k_inv, "GDN_NUDGE_KINV");
#endif

    recur_island_norm_qk: for (uint32_t j = 0; j < GDN_DK; ++j) {
#pragma HLS loop_tripcount min=256 max=256
#pragma HLS pipeline II=1
            q_loc[j] *= q_inv;
            k_loc[j] *= k_inv;
        }
    recur_island_load_v: for (uint32_t i = 0; i < GDN_DV; ++i) {
#pragma HLS loop_tripcount min=256 max=256
#pragma HLS pipeline II=1
            v_loc[i] = bf16_to_fp32(get_bf16_lane(v_head[i >> 5], i & 31));
        }

        const float beta = gdn_sigmoid(b[head_index]);
        const float decay_in = a[head_index] + layer_dt_bias[head_index];
        const float decay_val = -expf(layer_a_log[head_index]) *
                                gdn_softplus(decay_in);
        const float decay = expf(decay_val);
#ifndef __SYNTHESIS__
        /* Iter66m diagnostic (native only, env-gated): record the real
         * per-head scalar chain so an isolated cosim can recompute it from
         * these exact operands. The earlier transcendental sweep used an
         * 8,192-point grid, which can miss the isolated inputs where two
         * conforming exp/log implementations legally differ. */
        gdn_debug_record_head_scalars(layer_index, head_index, ISLAND,
                                      a[head_index], b[head_index],
                                      layer_a_log[head_index],
                                      layer_dt_bias[head_index],
                                      decay, beta);
#endif

    recur_island_alpha_product: for (uint32_t j = 0; j < GDN_DK; ++j) {
#pragma HLS loop_tripcount min=256 max=256
#pragma HLS pipeline II=1
            alpha_prod[j] = q_loc[j] * k_loc[j];
        }
        const float alpha = gdn_tree_reduce_256(alpha_prod);

        float retrieval_lo[GDN_RECURRENT_ISLAND_COLS];
        float retrieval_hi[GDN_RECURRENT_ISLAND_COLS];
        float partial_lo[GDN_RECURRENT_ISLAND_COLS];
        float partial_hi[GDN_RECURRENT_ISLAND_COLS];
        float delta_lo[GDN_RECURRENT_ISLAND_COLS];
        float delta_hi[GDN_RECURRENT_ISLAND_COLS];
        float output_lo[GDN_RECURRENT_ISLAND_COLS];
        float output_hi[GDN_RECURRENT_ISLAND_COLS];
#pragma HLS array_partition variable=retrieval_lo cyclic factor=GDN_RECURRENT_ISLAND_LANES
#pragma HLS array_partition variable=retrieval_hi cyclic factor=GDN_RECURRENT_ISLAND_LANES
#pragma HLS array_partition variable=partial_lo cyclic factor=GDN_RECURRENT_ISLAND_LANES
#pragma HLS array_partition variable=partial_hi cyclic factor=GDN_RECURRENT_ISLAND_LANES
#pragma HLS array_partition variable=delta_lo cyclic factor=GDN_RECURRENT_ISLAND_LANES
#pragma HLS array_partition variable=delta_hi cyclic factor=GDN_RECURRENT_ISLAND_LANES
#pragma HLS array_partition variable=output_lo cyclic factor=GDN_RECURRENT_ISLAND_LANES
#pragma HLS array_partition variable=output_hi cyclic factor=GDN_RECURRENT_ISLAND_LANES

    recur_island_init: for (uint32_t i = 0;
                            i < GDN_RECURRENT_ISLAND_COLS; ++i) {
#pragma HLS loop_tripcount min=64 max=64
#pragma HLS pipeline II=1
#pragma HLS unroll factor=GDN_RECURRENT_ISLAND_LANES
            retrieval_lo[i] = 0.0f;
            retrieval_hi[i] = 0.0f;
            partial_lo[i] = 0.0f;
            partial_hi[i] = 0.0f;
        }

    recur_island_read: for (uint32_t block = 0;
                            block < GDN_DK * 4; ++block) {
#pragma HLS loop_tripcount min=1024 max=1024
#pragma HLS pipeline II=1
            const uint32_t row = block >> 2;
            const uint32_t local_base =
                (block & 3) * GDN_RECURRENT_ISLAND_LANES;
            const float kj = k_loc[row];
            const float qj = q_loc[row];
        recur_island_read_lane: for (uint32_t lane = 0;
                                      lane < GDN_RECURRENT_ISLAND_LANES;
                                      ++lane) {
#pragma HLS unroll
                GDNStatePair state_value =
                    state_pair[row][local_base + lane];
                const uint32_t col = local_base + lane;
                /* Iter67: this loop ran at II=2, not the requested II=1, in
                 * every csynth report on disk. HLS names the reason exactly:
                 *
                 *   [HLS 200-880] Unable to enforce a carried dependence
                 *   constraint (II = 1, distance = 4, offset = 1) ... on
                 *   array 'partial_hi'
                 *
                 * `local_base` is (block & 3) * 16, so each accumulator is
                 * revisited every fourth iteration. II=1 therefore needs an
                 * adder latency of at most 4, and this file's default binding
                 * is 5 (see the fulldsp latency=5 accumulator in the GEMV
                 * cluster). Naming each sum and binding it to latency 4 buys
                 * II=1 without touching the arithmetic: the expression, its
                 * operand order, and the sequential row-major accumulation are
                 * all unchanged, so the trajectory stays bit-exact. Splitting
                 * the accumulators into parity banks would also give II=1, at
                 * distance 8, but it reassociates the sum and would retire the
                 * exact golden -- do not substitute it without regenerating
                 * the references and repeating the quality work. */
                float acc_r_lo = retrieval_lo[col] + state_value.lo * kj;
                float acc_r_hi = retrieval_hi[col] + state_value.hi * kj;
                float acc_p_lo = partial_lo[col] + state_value.lo * qj;
                float acc_p_hi = partial_hi[col] + state_value.hi * qj;
#pragma HLS bind_op variable=acc_r_lo op=fadd impl=fabric latency=4
#pragma HLS bind_op variable=acc_r_hi op=fadd impl=fabric latency=4
#pragma HLS bind_op variable=acc_p_lo op=fadd impl=fabric latency=4
#pragma HLS bind_op variable=acc_p_hi op=fadd impl=fabric latency=4
                retrieval_lo[col] = acc_r_lo;
                retrieval_hi[col] = acc_r_hi;
                partial_lo[col] = acc_p_lo;
                partial_hi[col] = acc_p_hi;
            }
        }

    /* Process 16 columns per island per cycle. Across both concurrently
     * running islands this retains the original aggregate 32-column width. */
    recur_island_delta: for (uint32_t logical = 0;
                             logical < 2 * GDN_RECURRENT_ISLAND_COLS;
                             ++logical) {
#pragma HLS loop_tripcount min=128 max=128
#pragma HLS pipeline II=1
#pragma HLS unroll factor=GDN_RECURRENT_ISLAND_LANES
            const bool high = logical >= GDN_RECURRENT_ISLAND_COLS;
            const uint32_t local = high
                ? logical - GDN_RECURRENT_ISLAND_COLS : logical;
            const uint32_t physical =
                (high ? GDN_DV / 2 : 0) +
                (local / GDN_RECURRENT_ISLAND_LANES) *
                    (2 * GDN_RECURRENT_ISLAND_LANES) +
                ISLAND * GDN_RECURRENT_ISLAND_LANES +
                (local % GDN_RECURRENT_ISLAND_LANES);
            const float retrieval = high ? retrieval_hi[local]
                                         : retrieval_lo[local];
            const float partial = high ? partial_hi[local]
                                       : partial_lo[local];
            const float delta = beta * (v_loc[physical] - decay * retrieval);
            const float output = q_scale *
                (decay * partial + alpha * delta);
            if (high) {
                delta_hi[local] = delta;
                output_hi[local] = output;
            } else {
                delta_lo[local] = delta;
                output_lo[local] = output;
            }
        }

    recur_island_output_half: for (uint32_t half = 0; half < 2; ++half) {
        recur_island_output_block: for (uint32_t block = 0;
                                        block < 4; ++block) {
#pragma HLS pipeline II=1
                Beat512 output_word = 0;
            recur_island_output_lane: for (uint32_t lane = 0;
                                            lane < GDN_RECURRENT_ISLAND_LANES;
                                            ++lane) {
#pragma HLS unroll
                    uint32_t local =
                        block * GDN_RECURRENT_ISLAND_LANES + lane;
                    set_fp32_lane(output_word, lane,
                        half == 0 ? output_lo[local] : output_hi[local]);
                }
                out_stream.write(output_word);
            }
        }

    recur_island_update: for (uint32_t packed_index = 0;
                              packed_index < GDN_DK * 2; ++packed_index) {
#pragma HLS loop_tripcount min=512 max=512
            const uint32_t row = packed_index >> 1;
            const uint32_t pair = packed_index & 1;
            Bf16Half state_low_half[2];
            Bf16Half state_high_half[2];
#pragma HLS array_partition variable=state_low_half complete
#pragma HLS array_partition variable=state_high_half complete
        recur_island_update_half: for (uint32_t subhalf = 0;
                                         subhalf < 2; ++subhalf) {
#pragma HLS pipeline II=1
#pragma HLS dependence variable=state_pair inter false
                const uint32_t local_base = pair * 32u + subhalf * 16u;
                const float kj = k_loc[row];
                Bf16Half packed_low = 0;
                Bf16Half packed_high = 0;
            recur_island_update_lane: for (uint32_t lane = 0;
                                            lane < GDN_RECURRENT_ISLAND_LANES;
                                            ++lane) {
#pragma HLS unroll
                    const uint32_t local = local_base + lane;
                    GDNStatePair old_state = state_pair[row][local];
                    const float updated_lo =
                        decay * old_state.lo + kj * delta_lo[local];
                    const float updated_hi =
                        decay * old_state.hi + kj * delta_hi[local];
                    GDNStatePair updated_state;
                    updated_state.lo = updated_lo;
                    updated_state.hi = updated_hi;
                    state_pair[row][local] = updated_state;
                    set_bf16_half_lane(packed_low, lane,
                                       fp32_to_bf16_rne(updated_lo));
                    set_bf16_half_lane(packed_high, lane,
                                       fp32_to_bf16_rne(updated_hi));
                }
                state_low_half[subhalf] = packed_low;
                state_high_half[subhalf] = packed_high;
            }
            state_out_low[head_state_base_bf16 + packed_index] =
                join_bf16_halves(state_low_half);
            state_out_high[head_state_base_bf16 + packed_index] =
                join_bf16_halves(state_high_half);
        }
    }
}

static void gdn_recurrent_merge_islands(
    hls::stream<Beat512> &out0,
    hls::stream<Beat512> &out1,
    Beat512 *attn_out
) {
#pragma HLS inline off
recur_island_merge_head: for (uint32_t head = 0;
                              head < GDN_HEADS; ++head) {
#pragma HLS loop_tripcount min=8 max=8
    recur_island_merge_half: for (uint32_t half = 0; half < 2; ++half) {
        recur_island_merge_block: for (uint32_t block = 0;
                                       block < 4; ++block) {
#pragma HLS pipeline II=1
                const Beat512 island0 = out0.read();
                const Beat512 island1 = out1.read();
                Beat512 packed = 0;
            recur_island_merge_lane: for (uint32_t lane = 0; lane < 16; ++lane) {
#pragma HLS unroll
                    set_bf16_lane(packed, lane,
                        fp32_to_bf16_rne(get_fp32_lane(island0, lane)));
                    set_bf16_lane(packed, lane + 16,
                        fp32_to_bf16_rne(get_fp32_lane(island1, lane)));
                }
                const size_t destination = (size_t)head * (GDN_DV / 32) +
                                           half * 4 + block;
                attn_out[destination] = packed;
            }
        }
    }
}

static void gdn_recurrent_attention_islands_dataflow(
    hls::stream<Beat512> &q_stream,
    hls::stream<Beat512> &k_stream,
    hls::stream<Beat512> &v_stream,
    hls::stream<Beat512> &state_stream0,
    hls::stream<Beat512> &state_stream1,
    hls::stream<Beat512> &state_stream2,
    hls::stream<Beat512> &state_stream3,
    Beat512 *attn_out,
    Beat512 *recurrent_state0,
    Beat512 *recurrent_state1,
    Beat512 *recurrent_state2,
    Beat512 *recurrent_state3,
    const float *a,
    const float *b,
    const float *layer_a_log,
    const float *layer_dt_bias,
    uint32_t layer_index
) {
#pragma HLS inline off
    hls::stream<Beat512> q0, k0, v0, q1, k1, v1;
    hls::stream<Beat512> out0, out1;
#pragma HLS stream variable=q0 depth=32
#pragma HLS stream variable=k0 depth=32
#pragma HLS stream variable=v0 depth=32
#pragma HLS stream variable=q1 depth=32
#pragma HLS stream variable=k1 depth=32
#pragma HLS stream variable=v1 depth=32
#pragma HLS stream variable=out0 depth=16
#pragma HLS stream variable=out1 depth=16
    /* These eight queues exist only inside the SLR2 recurrent wrapper. At
     * depths 16/32 their BRAM implementation consumed about 112 RAMB18s and
     * pushed both outer SLRs above 92% BRAM while SLR1 remained at 62.5%.
     * LUTRAM costs only about 3.6K LUTs here; keep the 69 high-traffic GEMV
     * decouplers in BRAM, where LUTRAM would cost roughly 60K LUTs. */
#pragma HLS bind_storage variable=q0 type=fifo impl=lutram
#pragma HLS bind_storage variable=k0 type=fifo impl=lutram
#pragma HLS bind_storage variable=v0 type=fifo impl=lutram
#pragma HLS bind_storage variable=q1 type=fifo impl=lutram
#pragma HLS bind_storage variable=k1 type=fifo impl=lutram
#pragma HLS bind_storage variable=v1 type=fifo impl=lutram
#pragma HLS bind_storage variable=out0 type=fifo impl=lutram
#pragma HLS bind_storage variable=out1 type=fifo impl=lutram

#pragma HLS dataflow disable_start_propagation
    gdn_recurrent_duplicate_qkv(q_stream, k_stream, v_stream,
                                q0, k0, v0, q1, k1, v1);
    gdn_recurrent_attention_island<0>(
        q0, k0, v0, state_stream0, state_stream2, out0,
        recurrent_state0, recurrent_state2,
        a, b, layer_a_log, layer_dt_bias, layer_index);
    gdn_recurrent_attention_island<1>(
        q1, k1, v1, state_stream1, state_stream3, out1,
        recurrent_state1, recurrent_state3,
        a, b, layer_a_log, layer_dt_bias, layer_index);
    gdn_recurrent_merge_islands(out0, out1, attn_out);
}

static void gdn_recurrent_attention_islands(
    hls::stream<Beat512> &q_stream,
    hls::stream<Beat512> &k_stream,
    hls::stream<Beat512> &v_stream,
    hls::stream<Beat512> &state_stream0,
    hls::stream<Beat512> &state_stream1,
    hls::stream<Beat512> &state_stream2,
    hls::stream<Beat512> &state_stream3,
    Beat512 *attn_out,
    Beat512 *recurrent_state0,
    Beat512 *recurrent_state1,
    Beat512 *recurrent_state2,
    Beat512 *recurrent_state3,
    const float *a,
    const float *b,
    const float *layer_a_log,
    const float *layer_dt_bias,
    uint32_t layer_index,
    bool enabled
) {
#pragma HLS inline off
    if (!enabled)
        return;
    gdn_recurrent_attention_islands_dataflow(
        q_stream, k_stream, v_stream,
        state_stream0, state_stream1, state_stream2, state_stream3,
        attn_out,
        recurrent_state0, recurrent_state1,
        recurrent_state2, recurrent_state3,
        a, b, layer_a_log, layer_dt_bias, layer_index);
}

#undef GDN_RECURRENT_ISLAND_COLS
#undef GDN_RECURRENT_ISLAND_LANES

static void gdn_output_norm_and_gate(
    Beat512 *out,
    const Beat512 *attn,
    const Beat512 *gate,
    const float *weight,
    uint32_t num_tokens,
    uint32_t num_heads,
    uint32_t head_dim,
    float eps
) {
    /* Attention and gate are BF16-resident. A head spans eight 32-lane words;
     * arithmetic and the original 16-value reduction chunks remain FP32. */
    uint32_t hd_packs = head_dim / 32;

    /* Pre-load the per-head norm weight once and reuse for every (token, head). */
    float weight_loc[GDN_DV];
    #pragma HLS array_partition variable=weight_loc cyclic factor=GDN_NORM_LANES
    uint32_t windex;
    onorm_load_w: for (windex = 0; windex < head_dim; ++windex) {
    #pragma HLS loop_tripcount min=256 max=256
    #pragma HLS pipeline II=1
        weight_loc[windex] = weight[windex];
    }

    uint32_t token_index;
    onorm_token: for (token_index = 0; token_index < num_tokens; ++token_index) {
    #pragma HLS loop_tripcount min=1 max=2048  /* num_tokens: 1..max_seq_len */
        uint32_t head_index;
        onorm_head: for (head_index = 0; head_index < num_heads; ++head_index) {
        #pragma HLS loop_tripcount min=8 max=8  /* num_heads=8 */
            size_t base = (size_t)(token_index * num_heads + head_index) * hd_packs;

            /* On-chip buffers break the AXI read-after-write hazard on attn and
             * avoid a second AXI read of gate per element. */
            float attn_loc[GDN_DV];
            float gate_loc[GDN_DV];
            #pragma HLS array_partition variable=attn_loc cyclic factor=GDN_NORM_LANES
            #pragma HLS array_partition variable=gate_loc cyclic factor=GDN_NORM_LANES

            /* Phase 1: load BF16 attention into local FP32 storage and retain
             * the old lower-then-upper 16-value reduction association. */
            float sum = 0.0f;
            onorm_sq: for (uint32_t ip = 0; ip < hd_packs; ++ip) {
            #pragma HLS loop_tripcount min=8 max=8
                Beat512 v = attn[base + ip];
                Bf16Half v_half[2];
                #pragma HLS array_partition variable=v_half complete
                split_bf16_beat(v, v_half);
            onorm_sq_half: for (uint32_t half = 0; half < 2; ++half) {
            #pragma HLS pipeline II=2
                    float s = 0.0f;
                    const Bf16Half current = v_half[half];
                onorm_sq_lane: for (int kk = 0; kk < 16; ++kk) {
                #pragma HLS unroll factor=GDN_NORM_LANES
                        const uint32_t lane = half * 16u + (uint32_t)kk;
                        float a = bf16_to_fp32(
                            get_bf16_half_lane(current, (uint32_t)kk));
                        attn_loc[ip * 32u + lane] = a;
                        s += a * a;
                    }
                    sum += s;
                }
            }

            /* Phase 2: load BF16 gate into local FP32 storage. */
            onorm_load_g: for (uint32_t ip = 0; ip < hd_packs; ++ip) {
            #pragma HLS loop_tripcount min=8 max=8
                Beat512 g = gate[base + ip];
                Bf16Half g_half[2];
                #pragma HLS array_partition variable=g_half complete
                split_bf16_beat(g, g_half);
            onorm_load_g_half: for (uint32_t half = 0; half < 2; ++half) {
            #pragma HLS pipeline II=2
                    const Bf16Half current = g_half[half];
                onorm_g_lane: for (int kk = 0; kk < 16; ++kk) {
                #pragma HLS unroll factor=GDN_NORM_LANES
                        const uint32_t lane = half * 16u + (uint32_t)kk;
                        gate_loc[ip * 32u + lane] =
                            bf16_to_fp32(
                                get_bf16_half_lane(current, (uint32_t)kk));
                    }
                }
            }

            float scale = 1.0f / sqrtf(sum / (float)head_dim + eps);

            /* Phase 3: combine in FP32 and round once into BF16 storage. */
            onorm_gate: for (uint32_t ip = 0; ip < hd_packs; ++ip) {
            #pragma HLS loop_tripcount min=8 max=8
                float o_lane[32];
                #pragma HLS array_partition variable=o_lane complete
                /* Retain four physical lanes here: fully parallel output-gate
                 * arithmetic saves too few token cycles for its DSP cost. */
                onorm_gate_group: for (int kb = 0; kb < 32;
                                       kb += GDN_OUTPUT_GATE_LANES) {
                #pragma HLS loop_tripcount min=8 max=8
                #pragma HLS pipeline II=1
                    onorm_gate_lane: for (int kl = 0;
                                          kl < GDN_OUTPUT_GATE_LANES; ++kl) {
                    #pragma HLS unroll
                        int kk = kb + kl;
                        uint32_t index = ip * 32u + (uint32_t)kk;
                        float normalized =
                            attn_loc[index] * scale * weight_loc[index];
                        float gate_value = gate_loc[index];
                        o_lane[kk] = normalized * gate_value
                                   * gdn_sigmoid(gate_value);
                    }
                }
                Beat512 o = 0;
                onorm_pack_out: for (int kk = 0; kk < 32; ++kk) {
                #pragma HLS unroll
                    set_bf16_lane(o, kk, fp32_to_bf16_rne(o_lane[kk]));
                }
                out[base + ip] = o;
            }
        }
    }
}

/* SwiGLU at a BF16 operator boundary. Both operands widen to FP32, the
 * nonlinear arithmetic stays FP32, and the result is RNE-rounded once.  The
 * output is separate so it can feed the one fixed GEMV-input BRAM directly. */
static void gdn_swiglu(Beat512 *out, const Beat512 *gate,
                       const Beat512 *up, size_t count) {
    const size_t count32 = count >> 5;
    swiglu_loop: for (size_t i = 0; i < count32; ++i) {
    #pragma HLS loop_tripcount min=176 max=360448
        Beat512 g = gate[i];
        Beat512 u = up[i];
        Bf16Bits result_lane[32];
        #pragma HLS array_partition variable=result_lane complete
        swiglu_group: for (int jb = 0; jb < 32;
                           jb += GDN_SWIGLU_LANES) {
        #pragma HLS loop_tripcount min=8 max=8
        #pragma HLS pipeline II=1
            swiglu_lane: for (int jl = 0;
                              jl < GDN_SWIGLU_LANES; ++jl) {
            #pragma HLS unroll
                int j = jb + jl;
                const float gv = bf16_to_fp32(get_bf16_lane(g, j));
                const float uv = bf16_to_fp32(get_bf16_lane(u, j));
                /* Round in the four-lane compute pipeline.  Keeping FP32
                 * results until a separately unrolled 32-lane pack caused HLS
                 * to instantiate 32 complete RNE/special-case converters even
                 * though only four arithmetic lanes are live. */
                result_lane[j] = fp32_to_bf16_rne(gdn_silu(gv) * uv);
            }
        }
        Beat512 packed = 0;
        swiglu_pack_out: for (int j = 0; j < 32; ++j) {
        #pragma HLS unroll
            set_bf16_lane(packed, j, result_lane[j]);
        }
        out[i] = packed;
    }
}

/* Deinterleave [chunk0 gate,up, chunk1 gate,up, ...] from the unified GU
 * command into the existing natural-row gate and up buffers. */
static void gdn_unpack_gu_local(
    Beat512 *gate, Beat512 *up, const Beat512 *gu
) {
#pragma HLS inline
gu_unpack_block: for (uint32_t block = 0; block < GEMV_CHANNELS; ++block) {
    uint32_t chunk = block >> 1;
    uint32_t kind = block & 1;
gu_unpack_pack: for (uint32_t p = 0;
                     p < (GDN_INTER / (GEMV_CHANNELS / 2)) / 32; ++p) {
#pragma HLS loop_tripcount min=11 max=11
#pragma HLS pipeline II=1
        Beat512 value = gu[block * ((GDN_INTER / (GEMV_CHANNELS / 2)) / 32) + p];
        uint32_t destination =
            chunk * ((GDN_INTER / (GEMV_CHANNELS / 2)) / 32) + p;
        if (kind == 0)
            gate[destination] = value;
        else
            up[destination] = value;
    }
}
}

static ap_uint<256> gdn_bf16_add_half(
    ap_uint<256> residual, ap_uint<256> projection
) {
#pragma HLS inline
    ap_uint<256> result = 0;
add_half_lane: for (uint32_t lane = 0; lane < 16; ++lane) {
#pragma HLS unroll
        const Bf16Bits residual_bits =
            residual.range(16 * lane + 15, 16 * lane);
        const Bf16Bits projection_bits =
            projection.range(16 * lane + 15, 16 * lane);
        float sum_value = bf16_to_fp32(residual_bits)
                        + bf16_to_fp32(projection_bits);
#pragma HLS bind_op variable=sum_value op=fadd impl=fulldsp latency=5
        const Bf16Bits sum_bits = fp32_to_bf16_rne(sum_value);
        result.range(16 * lane + 15, 16 * lane) = sum_bits;
    }
    return result;
}

static Beat512 gdn_bf16_add_beat(
    const Beat512 residual_beat, const Beat512 projection_beat
) {
#pragma HLS inline
    ap_uint<256> residual_half[2];
    ap_uint<256> projection_half[2];
    ap_uint<256> result_half[2];
#pragma HLS array_partition variable=residual_half complete
#pragma HLS array_partition variable=projection_half complete
#pragma HLS array_partition variable=result_half complete
    residual_half[0] = residual_beat.range(255, 0);
    residual_half[1] = residual_beat.range(511, 256);
    projection_half[0] = projection_beat.range(255, 0);
    projection_half[1] = projection_beat.range(511, 256);
add_local_half: for (uint32_t half = 0; half < 2; ++half) {
#pragma HLS pipeline II=1
        result_half[half] = gdn_bf16_add_half(
            residual_half[half], projection_half[half]);
    }
    Beat512 sum;
    sum.range(255, 0) = result_half[0];
    sum.range(511, 256) = result_half[1];
    return sum;
}

static void gdn_beat_add_local(
    Beat512 *out, const Beat512 *residual, const Beat512 *projection,
    uint32_t count32
) {
#pragma HLS inline
add_local: for (uint32_t i = 0; i < count32; ++i) {
#pragma HLS loop_tripcount min=64 max=64
        out[i] = gdn_bf16_add_beat(residual[i], projection[i]);
    }
}

#ifdef GDN_BF16_ADD_HLS_TEST
Beat512 gdn_bf16_add_hls_top(
    const Beat512 residual, const Beat512 projection
) {
#pragma HLS interface ap_none port=residual
#pragma HLS interface ap_none port=projection
#pragma HLS interface ap_none port=return
    return gdn_bf16_add_beat(residual, projection);
}
#endif

/* Forward decl: the decode-only clustered GEMV engine. */
static void gdn_gemv(
    Beat512 *out, hls::stream<Beat512> &logits_stream, const Beat512 *in,
    const Beat512 *w0, const Beat512 *w1, const Beat512 *w2, const Beat512 *w3,
    const Beat512 *w4, const Beat512 *w5, const Beat512 *w6, const Beat512 *w7,
    const Beat512 *w8, const Beat512 *w9, const Beat512 *w10, const Beat512 *w11,
    const Beat512 *w12, const Beat512 *w13, const Beat512 *w14, const Beat512 *w15,
    const Beat512 *w16, const Beat512 *w17, const Beat512 *w18, const Beat512 *w19,
    const Beat512 *w20, const Beat512 *w21, const Beat512 *w22, const Beat512 *w23,
    const Beat512 *w24, const Beat512 *w25, const Beat512 *w26, const Beat512 *w27,
    Beat512 *w28, Beat512 *w29, Beat512 *w30, Beat512 *w31,
    uint32_t w_pack_off,
    uint32_t in_dim, uint32_t out_dim,
    bool qkvg_recurrent_mode,
    Beat512 *attn_out, Beat512 *gate_out,
    const Beat512 conv_weights[3][(GDN_HIDDEN * GDN_CONV) / 16],
    Beat512 conv_tails[3][((GDN_CONV - 1) * GDN_HIDDEN) / 32],
    const float a[GDN_HEADS],
    const float b[GDN_HEADS],
    const float layer_a_log[GDN_HEADS],
    const float layer_dt_bias[GDN_HEADS],
    uint32_t layer_index);

/* Vitis HLS 2022.1 does not synthesize arrays of pointers. This expands each
 * call to the explicit scalar pointer interface above. */
#define GDN_GEMV_SHARD_ARGUMENTS \
    weight_data_mm0, weight_data_mm1, weight_data_mm2, weight_data_mm3, \
    weight_data_mm4, weight_data_mm5, weight_data_mm6, weight_data_mm7, \
    weight_data_mm8, weight_data_mm9, weight_data_mm10, weight_data_mm11, \
    weight_data_mm12, weight_data_mm13, weight_data_mm14, weight_data_mm15, \
    weight_data_mm16, weight_data_mm17, weight_data_mm18, weight_data_mm19, \
    weight_data_mm20, weight_data_mm21, weight_data_mm22, weight_data_mm23, \
    weight_data_mm24, weight_data_mm25, weight_data_mm26, weight_data_mm27, \
    weight_data_mm28, weight_data_mm29, weight_data_mm30, weight_data_mm31

int gdn_forward(
    const float *aux_weights,
    Beat512 *workspace,
    const Beat512 *weight_data_mm0,
    const Beat512 *weight_data_mm1,
    const Beat512 *weight_data_mm2,
    const Beat512 *weight_data_mm3,
    const Beat512 *weight_data_mm4,
    const Beat512 *weight_data_mm5,
    const Beat512 *weight_data_mm6,
    const Beat512 *weight_data_mm7,
    const Beat512 *weight_data_mm8,
    const Beat512 *weight_data_mm9,
    const Beat512 *weight_data_mm10,
    const Beat512 *weight_data_mm11,
    const Beat512 *weight_data_mm12,
    const Beat512 *weight_data_mm13,
    const Beat512 *weight_data_mm14,
    const Beat512 *weight_data_mm15,
    const Beat512 *weight_data_mm16,
    const Beat512 *weight_data_mm17,
    const Beat512 *weight_data_mm18,
    const Beat512 *weight_data_mm19,
    const Beat512 *weight_data_mm20,
    const Beat512 *weight_data_mm21,
    const Beat512 *weight_data_mm22,
    const Beat512 *weight_data_mm23,
    const Beat512 *weight_data_mm24,
    const Beat512 *weight_data_mm25,
    const Beat512 *weight_data_mm26,
    const Beat512 *weight_data_mm27,
    Beat512 *weight_data_mm28,
    Beat512 *weight_data_mm29,
    Beat512 *weight_data_mm30,
    Beat512 *weight_data_mm31
) {
    /* Depths match gdn-1.3b-f32.gdnw: hidden=2048 heads=8 head_dim=256
    intermediate=5632 layers=24 conv=4 max_seq_len=2048 vocab=32000 */
    /* This U55C shell exposes 32 HMSS masters. Every scalar weight, activation,
     * and state buffer shares weight port 0 and is allocated in HBM0; ports 1..31
     * remain read-only. A combined x/w0 loader is the sole dataflow reader of
     * port 0. The host writes the selected token embedding directly into x.
     *
     * DO NOT reduce mm0's outstanding depths. iter14/iter15 tried 64->8 read and
     * write (to shrink the 29-deep BRAM cascade RQS_TIMING-6 flagged on iter13's
     * worst path) and BOTH links were REFUSED by the router with
     * [Route 35-3] not routable, where iter13 at 64 completed route_design.
     * iter15 isolated it: link cfg byte-identical to iter13, mm0 depth the only
     * variable. It saves 50 BRAM and costs routability -- the freed BRAM came
     * out of SLR1/SLR2 (SLR0 actually GAINED 7 tiles), which let the placer
     * compact the design, pull cluster 5 back into SLR0 and use SLR2 less. The
     * whole margin between routing and refusal is ~22 K cells of SLR0
     * occupancy. See doc/optimization_log.md sec iter15. */
    #pragma HLS interface m_axi port=aux_weights depth=2000000 offset=slave bundle=mem_weights_mm0 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=64 max_write_burst_length=64 num_write_outstanding=64
    /* Thirty-two compact GEMV shards, each on an independent 512-bit master.
     * The clustered datapath consumes one Beat512 beat per master per cycle. */
    /* One compact GDN-1.3B shard is 43,728,896 floats (166.8125 MiB).
     * Do not use the full-model float count here: it exceeds one AXI address
     * range and corrupts the metadata consumed by the Vitis platform linker. */
    #pragma HLS interface m_axi port=weight_data_mm0 depth=1366528 offset=slave bundle=mem_weights_mm0 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=64 max_write_burst_length=64 num_write_outstanding=64
    #pragma HLS interface m_axi port=weight_data_mm1 depth=1366528 offset=slave bundle=mem_weights_mm1 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=4
    #pragma HLS interface m_axi port=weight_data_mm2 depth=1366528 offset=slave bundle=mem_weights_mm2 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=4
    #pragma HLS interface m_axi port=weight_data_mm3 depth=1366528 offset=slave bundle=mem_weights_mm3 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=4
    #pragma HLS interface m_axi port=weight_data_mm4 depth=1366528 offset=slave bundle=mem_weights_mm4 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=4
    #pragma HLS interface m_axi port=weight_data_mm5 depth=1366528 offset=slave bundle=mem_weights_mm5 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=4
    #pragma HLS interface m_axi port=weight_data_mm6 depth=1366528 offset=slave bundle=mem_weights_mm6 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=4
    #pragma HLS interface m_axi port=weight_data_mm7 depth=1366528 offset=slave bundle=mem_weights_mm7 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=4
    #pragma HLS interface m_axi port=weight_data_mm8 depth=1366528 offset=slave bundle=mem_weights_mm8 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=4
    #pragma HLS interface m_axi port=weight_data_mm9 depth=1366528 offset=slave bundle=mem_weights_mm9 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=4
    #pragma HLS interface m_axi port=weight_data_mm10 depth=1366528 offset=slave bundle=mem_weights_mm10 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=4
    #pragma HLS interface m_axi port=weight_data_mm11 depth=1366528 offset=slave bundle=mem_weights_mm11 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=4
    #pragma HLS interface m_axi port=weight_data_mm12 depth=1366528 offset=slave bundle=mem_weights_mm12 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=4
    #pragma HLS interface m_axi port=weight_data_mm13 depth=1366528 offset=slave bundle=mem_weights_mm13 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=4
    #pragma HLS interface m_axi port=weight_data_mm14 depth=1366528 offset=slave bundle=mem_weights_mm14 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=4
    #pragma HLS interface m_axi port=weight_data_mm15 depth=1366528 offset=slave bundle=mem_weights_mm15 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=4
    #pragma HLS interface m_axi port=weight_data_mm16 depth=1366528 offset=slave bundle=mem_weights_mm16 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=4
    #pragma HLS interface m_axi port=weight_data_mm17 depth=1366528 offset=slave bundle=mem_weights_mm17 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=4
    #pragma HLS interface m_axi port=weight_data_mm18 depth=1366528 offset=slave bundle=mem_weights_mm18 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=4
    #pragma HLS interface m_axi port=weight_data_mm19 depth=1366528 offset=slave bundle=mem_weights_mm19 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=4
    #pragma HLS interface m_axi port=weight_data_mm20 depth=1366528 offset=slave bundle=mem_weights_mm20 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=4
    #pragma HLS interface m_axi port=weight_data_mm21 depth=1366528 offset=slave bundle=mem_weights_mm21 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=4
    #pragma HLS interface m_axi port=weight_data_mm22 depth=1366528 offset=slave bundle=mem_weights_mm22 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=4
    #pragma HLS interface m_axi port=weight_data_mm23 depth=1366528 offset=slave bundle=mem_weights_mm23 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=4
    #pragma HLS interface m_axi port=weight_data_mm24 depth=1366528 offset=slave bundle=mem_weights_mm24 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=4
    #pragma HLS interface m_axi port=weight_data_mm25 depth=1366528 offset=slave bundle=mem_weights_mm25 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=4
    #pragma HLS interface m_axi port=weight_data_mm26 depth=1366528 offset=slave bundle=mem_weights_mm26 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=4
    #pragma HLS interface m_axi port=weight_data_mm27 depth=1366528 offset=slave bundle=mem_weights_mm27 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=4
    #pragma HLS interface m_axi port=weight_data_mm28 depth=1464832 offset=slave bundle=mem_weights_mm28 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=4 max_write_burst_length=64 num_write_outstanding=8
    #pragma HLS interface m_axi port=weight_data_mm29 depth=1464832 offset=slave bundle=mem_weights_mm29 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=4 max_write_burst_length=64 num_write_outstanding=8
    #pragma HLS interface m_axi port=weight_data_mm30 depth=1464832 offset=slave bundle=mem_weights_mm30 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=4 max_write_burst_length=64 num_write_outstanding=8
    #pragma HLS interface m_axi port=weight_data_mm31 depth=1464832 offset=slave bundle=mem_weights_mm31 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=4 max_write_burst_length=64 num_write_outstanding=8
    /* step 4 Stage B: the 15 activation/state buffers are packed into this one
     * workspace pointer (GDN_WS_OFF_* layout in gdn_model.h), replacing 15 m_axi
     * ports and their control_s_axi base-address registers. Read+write, HBM0. */
    #pragma HLS interface m_axi port=workspace depth=817810 offset=slave bundle=mem_weights_mm0 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=64 max_write_burst_length=64 num_write_outstanding=64
    #pragma HLS interface s_axilite port=return
    /* All projection shapes must time-share the one routed 32-reader,
     * 16-cluster engine. Local activation memories otherwise encourage HLS to
     * specialize one complete dataflow graph per input/output buffer size. */
    #pragma HLS allocation function instances=gdn_gemv limit=1

    /* step 4: fixed GDN-1.3B decode shape, one token per call. config/max_tokens/
     * num_tokens are gone from the signature; these constants replace them so the
     * synthesized loops have literal bounds and control_s_axi loses those regs. */
    const uint32_t hidden = GDN_HIDDEN;
    const uint32_t num_heads = GDN_HEADS;
    const uint32_t head_dim = GDN_HEAD_DIM;
    const uint32_t intermediate = GDN_INTER;
    const uint32_t num_tokens = 1;
    uint32_t layer_index;
    size_t mlp_count;
    const float *final_norm = aux_weights +
        (size_t)GDN_LAYERS * GDN_AUX_LAYER_STRIDE;

    /* Iter32: only persistent state and the host handoff remain in workspace.
     * Preserve every external offset so the committed host/ABI stays unchanged,
     * but keep the complete transient activation lifetime in six BRAM-backed,
     * 16-bank buffers. The two 5632-entry buffers hold q/k during attention and
     * are reused for the MLP gate/up vectors after recurrence consumes q/k. */
    Beat512 *workspace_x = workspace + GDN_WS_OFF_X / 16;
    Beat512 *head_buffer = workspace + GDN_WS_OFF_HEAD_BUF / 16;
    Beat512 x_storage[GDN_HIDDEN / 32];
    /* Residual ping-pong storage.  The rejected in-place adapter made HLS
     * flatten the 64-Beat/two-half loop into a read-after-write recurrence on
     * x_storage (II=5).  Each layer writes the output-projection residual here
     * and the MLP-down residual back to x_storage, making alias freedom
     * structural while preserving the exact add/RNE order. */
    Beat512 x_alt_storage[GDN_HIDDEN / 32];
    /* Every projection reads this same physical BRAM.  HLS specializes a
     * dataflow function for each distinct caller-local array even when an
     * allocation limit is present; using different norm/attention/gate arrays
     * cloned the complete 16-cluster engine three times in the rejected
     * Iter64 csynth. Producers write this buffer directly, so this is not an
     * activation-packing pass. */
    Beat512 gemv_in_storage[GDN_INTER / 32];
    Beat512 q_mlp_gate_storage[GDN_INTER / 32];
    Beat512 k_mlp_up_storage[GDN_INTER / 32];
    Beat512 gate_storage[GDN_HIDDEN / 32];
    Beat512 gemv_out_storage[(2 * GDN_INTER) / 32];
    Beat512 conv_weight_storage[3][(GDN_HIDDEN * GDN_CONV) / 16];
    Beat512 conv_tail_storage[3][((GDN_CONV - 1) * GDN_HIDDEN) / 32];
    float a_storage[GDN_HEADS];
    float b_storage[GDN_HEADS];
    float a_log_storage[GDN_HEADS];
    float dt_bias_storage[GDN_HEADS];
#pragma HLS bind_storage variable=x_storage type=ram_2p impl=bram
#pragma HLS bind_storage variable=x_alt_storage type=ram_2p impl=bram
#pragma HLS bind_storage variable=gemv_in_storage type=ram_2p impl=bram
#pragma HLS bind_storage variable=q_mlp_gate_storage type=ram_2p impl=bram
#pragma HLS bind_storage variable=k_mlp_up_storage type=ram_2p impl=bram
#pragma HLS bind_storage variable=gate_storage type=ram_2p impl=bram
#pragma HLS bind_storage variable=gemv_out_storage type=ram_2p impl=bram

    /* Iter61: LM-head logit queue. Deep enough to hold the whole GDN_VOCAB
     * vector, because gdn_gemv is called from this sequential region: the
     * producer finishes before this function drains it, so there is no
     * concurrency to shrink the depth. 2048 x 512-bit in URAM is 8 blocks out
     * of 960, of which only 48 are in use. Bound to URAM deliberately so this
     * costs no BRAM -- BRAM sits at 90-91% in SLR0/SLR1. */
    static hls::stream<Beat512> logits_stream;
#pragma HLS stream variable=logits_stream depth=2048
#pragma HLS bind_storage variable=logits_stream type=fifo impl=uram
#pragma HLS bind_storage variable=conv_weight_storage type=ram_2p impl=bram
#pragma HLS bind_storage variable=conv_tail_storage type=ram_2p impl=bram
#pragma HLS array_partition variable=conv_weight_storage complete dim=1
#pragma HLS array_partition variable=conv_tail_storage complete dim=1
#pragma HLS aggregate variable=conv_weight_storage compact=bit
#pragma HLS aggregate variable=conv_tail_storage compact=bit
#pragma HLS array_partition variable=a_storage complete dim=1
#pragma HLS array_partition variable=b_storage complete dim=1
#pragma HLS array_partition variable=a_log_storage complete dim=1
#pragma HLS array_partition variable=dt_bias_storage complete dim=1

    float *a = a_storage;
    float *b = b_storage;

    mlp_count = (size_t)num_tokens * intermediate;

    /* Compact-shard geometry (Beat512 units): the first per-layer segment is one
     * head-major Q/K/V/gate command (four old HxH stripe lengths), followed by
     * O, one pair-interleaved gate/up command, and MLP-down. */
    size_t shard_hh = (size_t)(hidden / GEMV_CHANNELS) * (hidden / 32);
    size_t shard_qkvg = 4 * shard_hh;
    size_t shard_ih = (size_t)(intermediate / GEMV_CHANNELS) * (hidden / 32);
    size_t shard_gu = 2 * shard_ih;
    size_t shard_di = (size_t)(hidden / GEMV_CHANNELS) * (intermediate / 32);
    size_t shard_per_layer = 5 * shard_hh + 2 * shard_ih + shard_di;

    /* The workspace ABI remains FP32. Import the one embedding row once and
     * immediately establish the all-BF16 transient contract on chip. */
    {
    load_embedding_local: for (uint32_t p = 0; p < GDN_HIDDEN / 32; ++p) {
#pragma HLS loop_tripcount min=64 max=64
#pragma HLS pipeline II=1
            const Beat512 lo = workspace_x[2 * p];
            const Beat512 hi = workspace_x[2 * p + 1];
            Beat512 packed = 0;
        load_embedding_lane: for (uint32_t lane = 0; lane < 16; ++lane) {
#pragma HLS unroll
                set_bf16_lane(packed, lane,
                    fp32_to_bf16_rne(get_fp32_lane(lo, lane)));
                set_bf16_lane(packed, lane + 16,
                    fp32_to_bf16_rne(get_fp32_lane(hi, lane)));
            }
            x_storage[p] = packed;
        }
    }

    layer_loop: for (layer_index = 0; layer_index < GDN_LAYERS; ++layer_index) {
    #pragma HLS loop_tripcount min=24 max=24  /* num_layers=24 */
        size_t layer_offset = (size_t)layer_index * GDN_AUX_LAYER_STRIDE;
        const float *layer_attn_norm = aux_weights + layer_offset;
        const float *layer_a_log;
        const float *layer_a_proj;
        const float *layer_b_proj;
        const float *layer_q_conv;
        const float *layer_k_conv;
        const float *layer_v_conv;
        const float *layer_o_norm;
        const float *layer_mlp_norm;

        /* Non-GEMV tensors are packed contiguously in aux_weights. */
        layer_offset += hidden;                          /* past attn_norm */
        layer_a_log = aux_weights + layer_offset;
        layer_offset += 2 * num_heads;                  /* a_log + dt_bias */
        layer_a_proj = aux_weights + layer_offset;
        layer_offset += (size_t)num_heads * hidden;
        layer_b_proj = aux_weights + layer_offset;
        layer_offset += (size_t)num_heads * hidden;
        layer_q_conv = aux_weights + layer_offset;
        layer_offset += (size_t)hidden * GDN_CONV;
        layer_k_conv = aux_weights + layer_offset;
        layer_offset += (size_t)hidden * GDN_CONV;
        layer_v_conv = aux_weights + layer_offset;
        layer_offset += (size_t)hidden * GDN_CONV;
        layer_o_norm = aux_weights + layer_offset;
        layer_offset += head_dim;
        layer_mlp_norm = aux_weights + layer_offset;

        /* Running compact-shard offset (Beat512); order qkvg,o,gu,mlp_down —
         * matches gdn_build_weight_shards. */
        size_t soff = (size_t)layer_index * shard_per_layer;

        gdn_rmsnorm_rows_bf16(gemv_in_storage, x_storage, layer_attn_norm,
                              num_tokens, hidden, GDN_NORM_EPS);
        /* Recurrence consumes each convolved head inside the QKVG dataflow
         * graph. Stage every auxiliary scalar before that graph starts so the
         * shared mem_weights_mm0 adapter retains a single active reader. */
        gdn_gemv_tiny(a, gemv_in_storage, layer_a_proj,
                      hidden, num_heads);
        gdn_gemv_tiny(b, gemv_in_storage, layer_b_proj,
                      hidden, num_heads);
        gdn_load_recurrent_scalars(a_log_storage, dt_bias_storage,
                                   layer_a_log);

        /* Per-(layer, conv) slice of the persistent conv tail in head_buffer:
         * 3 convs/layer x (conv_size-1) rows x hidden floats. Iter39B passes
         * these into the QKVG result sink so head h convolution overlaps GEMV
         * production of head h+1. */
        size_t tail_stride = (size_t)(GDN_CONV - 1) * hidden / 16;
        Beat512 *q_tail = head_buffer +
            ((size_t)layer_index * 3 + 0) * tail_stride;
        Beat512 *k_tail = head_buffer +
            ((size_t)layer_index * 3 + 1) * tail_stride;
        Beat512 *v_tail = head_buffer +
            ((size_t)layer_index * 3 + 2) * tail_stride;

        gdn_load_qkvg_conv_context(
            conv_weight_storage, conv_tail_storage,
            layer_q_conv, layer_k_conv, layer_v_conv,
            q_tail, k_tail, v_tail);
        /* q_mlp_gate_storage is dead until the later GU projection, so reuse
         * it for the recurrent attention result. Keeping the fixed GEMV input
         * and recurrent output in distinct BRAMs is required by HLS dataflow. */
        gdn_gemv(gemv_out_storage, logits_stream, gemv_in_storage,
                 GDN_GEMV_SHARD_ARGUMENTS,
                 (uint32_t)soff, hidden, 4 * hidden,
                 true,
                 q_mlp_gate_storage, gate_storage,
                 conv_weight_storage, conv_tail_storage,
                 a, b, a_log_storage, dt_bias_storage, layer_index);
        gdn_store_qkvg_conv_tails(q_tail, k_tail, v_tail,
                                  conv_tail_storage);
        soff += shard_qkvg;
        gdn_output_norm_and_gate(gemv_in_storage, q_mlp_gate_storage,
                                 gate_storage,
                                 layer_o_norm, num_tokens, num_heads,
                                 head_dim, GDN_NORM_EPS);
        gdn_gemv(gemv_out_storage, logits_stream, gemv_in_storage,
                 GDN_GEMV_SHARD_ARGUMENTS,
                 (uint32_t)soff, hidden, hidden,
                 false,
                 q_mlp_gate_storage, gate_storage,
                 conv_weight_storage, conv_tail_storage,
                 a, b, a_log_storage, dt_bias_storage, layer_index);
        gdn_beat_add_local(x_alt_storage, x_storage, gemv_out_storage,
                           hidden / 32);
        soff += shard_hh;

        gdn_rmsnorm_rows_bf16(gemv_in_storage, x_alt_storage, layer_mlp_norm,
                              num_tokens, hidden, GDN_NORM_EPS);
        gdn_gemv(gemv_out_storage, logits_stream, gemv_in_storage,
                 GDN_GEMV_SHARD_ARGUMENTS,
                 (uint32_t)soff, hidden, 2 * intermediate,
                 false,
                 q_mlp_gate_storage, gate_storage,
                 conv_weight_storage, conv_tail_storage,
                 a, b, a_log_storage, dt_bias_storage, layer_index);
        gdn_unpack_gu_local(q_mlp_gate_storage, k_mlp_up_storage,
                            gemv_out_storage);
        soff += shard_gu;
        gdn_swiglu(gemv_in_storage, q_mlp_gate_storage,
                   k_mlp_up_storage, mlp_count);
        gdn_gemv(gemv_out_storage, logits_stream, gemv_in_storage,
                 GDN_GEMV_SHARD_ARGUMENTS,
                 (uint32_t)soff, intermediate, hidden,
                 false,
                 q_mlp_gate_storage, gate_storage,
                 conv_weight_storage, conv_tail_storage,
                 a, b, a_log_storage, dt_bias_storage, layer_index);
        gdn_beat_add_local(x_storage, x_alt_storage, gemv_out_storage,
                           hidden / 32);
    }

    gdn_rmsnorm_rows_bf16(gemv_in_storage, x_storage, final_norm,
                          num_tokens, hidden, GDN_NORM_EPS);
#ifndef __SYNTHESIS__
    if (gdn_native_final_hidden_debug != NULL) {
        for (uint32_t p = 0; p < GDN_HIDDEN / 32; ++p) {
            for (uint32_t lane = 0; lane < 32; ++lane) {
                gdn_native_final_hidden_debug[p * 32 + lane] =
                    bf16_to_fp32(get_bf16_lane(gemv_in_storage[p], lane));
            }
        }
    }
#endif
    /* The LM-head store streams its reorder buffer out as the full FP32 logit
     * vector; the greedy pick happens on the host. */
    {
        size_t lm_soff = (size_t)GDN_LAYERS * shard_per_layer;
        gdn_gemv(gemv_out_storage, logits_stream, gemv_in_storage,
                 GDN_GEMV_SHARD_ARGUMENTS,
                 (uint32_t)lm_soff, hidden, GDN_VOCAB,
                 false,
                 q_mlp_gate_storage, gate_storage,
                 conv_weight_storage, conv_tail_storage,
                 a, b, a_log_storage, dt_bias_storage, layer_index);
    }

    /* Iter67: the on-chip argmax is back, so hand the host a token id again.
     * gemv_out_storage[0] lane 0 holds it; one 512-bit line to the workspace
     * token slot lets a generation loop read 4 bytes instead of pulling the
     * whole 128 KB logit vector across PCIe every step. The logit drain below
     * is unchanged and still serves teacher-forced scoring, which needs the
     * full vector and is not latency-bound. */
    {
        Beat512 *token_out = workspace + GDN_WS_OFF_X_NORM / 16;
        token_out[0] = gemv_out_storage[0];
    }

    /* Drain the LM-head logit queue to HBM. This is the only new AXI traffic
     * in Iter61 and it lives here, at the top level, next to the token write
     * above -- the same master and the same code region that Iter57 already
     * routes. Full 512-bit lines only; GDN_VOCAB is a multiple of 16. */
    {
        Beat512 *logits_out = workspace + GDN_WS_OFF_LOGITS / 16;
    drain_logits: for (uint32_t k = 0; k < GDN_VOCAB / 16; ++k) {
#pragma HLS loop_tripcount min=2000 max=2000
#pragma HLS pipeline II=1
            logits_out[k] = logits_stream.read();
        }
    }
    return 0;
}

#undef GDN_GEMV_SHARD_ARGUMENTS

/* Decode-only host entry: forward exactly one token against the persistent
 * per-layer recurrent + conv state held in the run-state buffers (loaded from
 * the GPU .gdnstate export). gdn_forward is decode-only — it restores each
 * layer's state at the start and saves the update at the end. */
int gdn_decode_step_host(const GDNModel *model, GDNRunState *state, const int32_t *token) {
    if (*token < 0 || (uint32_t)*token >= model->config.vocab_size) {
        gdn_print_error("token id out of range");
        return -1;
    }
    memcpy(state->x,
           model->embeddings + (size_t)*token * model->config.hidden_size,
           (size_t)model->config.hidden_size * sizeof(float));
    return gdn_forward(
        state->aux_weights,
        state->workspace,
        state->weight_shards[0],  state->weight_shards[1],
        state->weight_shards[2],  state->weight_shards[3],
        state->weight_shards[4],  state->weight_shards[5],
        state->weight_shards[6],  state->weight_shards[7],
        state->weight_shards[8],  state->weight_shards[9],
        state->weight_shards[10], state->weight_shards[11],
        state->weight_shards[12], state->weight_shards[13],
        state->weight_shards[14], state->weight_shards[15],
        state->weight_shards[16], state->weight_shards[17],
        state->weight_shards[18], state->weight_shards[19],
        state->weight_shards[20], state->weight_shards[21],
        state->weight_shards[22], state->weight_shards[23],
        state->weight_shards[24], state->weight_shards[25],
        state->weight_shards[26], state->weight_shards[27],
        state->weight_shards[28], state->weight_shards[29],
        state->weight_shards[30], state->weight_shards[31]
    );
}

void gdn_compute_logits(const GDNModel *model, const float *hidden, float *logits_out) {
    uint32_t vocab = model->config.vocab_size;
    uint32_t hidden_size = model->config.hidden_size;
    uint32_t vocab_index;

    /* The native-BF16 GEMV rounds both operands and every product to BF16.
     * This remains an independent scalar LM-head check with a different
     * reduction order, but it models the same per-product rounding point. */
    float *rounded = (float *)malloc((size_t)hidden_size * sizeof(float));
    if (rounded == NULL) {
        gdn_print_error("gdn_compute_logits: allocation failed");
        return;
    }
    {
        uint32_t i;
        for (i = 0; i < hidden_size; ++i) {
            rounded[i] = bf16_to_fp32(fp32_to_bf16_rne(hidden[i]));
        }
    }

    for (vocab_index = 0; vocab_index < vocab; ++vocab_index) {
        const float *weight_row = model->lm_head + (size_t)vocab_index * hidden_size;
        float sum = 0.0f;
        uint32_t hidden_index;
        for (hidden_index = 0; hidden_index < hidden_size; ++hidden_index) {
            const Bf16Bits activation_bits =
                fp32_to_bf16_rne(rounded[hidden_index]);
            const Bf16Bits weight_bits =
                fp32_to_bits(weight_row[hidden_index]).range(31, 16);
            sum += gdn_native_bf16_mul_to_fp32(
                weight_bits, activation_bits);
        }
        logits_out[vocab_index] = sum;
    }
    free(rounded);
}


/* Decode-only GEMV constants shared by the routed 32-port implementation. */
#define IN_DIM_MAX    5632   /* max in_dim (intermediate=5632) — sizes a_loc */
/* GEMV_CHANNELS lives in gdn_model.h (shared by the kernel and the host). */

/* 32-port GEMV topology: 32 MM2S readers feed sixteen two-port compute
 * clusters. The smaller clusters preserve one weight beat per port per cycle
 * while reducing each independently placeable FP32 block by roughly half.
 * Activations ripple through one BRAM copy per cluster, and results merge
 * through SLR-local collectors. */
#define GEMV32_MAX_RESULT_PACKS 2048

/* The existing balanced association, factored out so both ports of a paired
 * half-dot reuse it unchanged: 8 pairwise adds, then 4, 2, and the final sum.
 * Inlined, so each call site gets its own tree rather than sharing one. */
static float gemv32_tree16(const float prod[16]) {
#pragma HLS inline
    float s0 = prod[0] + prod[1];
    float s1 = prod[2] + prod[3];
    float s2 = prod[4] + prod[5];
    float s3 = prod[6] + prod[7];
    float s4 = prod[8] + prod[9];
    float s5 = prod[10] + prod[11];
    float s6 = prod[12] + prod[13];
    float s7 = prod[14] + prod[15];
#pragma HLS bind_op variable=s0 op=fadd impl=fulldsp
#pragma HLS bind_op variable=s1 op=fadd impl=fulldsp
#pragma HLS bind_op variable=s2 op=fadd impl=fulldsp
#pragma HLS bind_op variable=s3 op=fadd impl=fulldsp
#pragma HLS bind_op variable=s4 op=fadd impl=fulldsp
#pragma HLS bind_op variable=s5 op=fadd impl=fulldsp
#pragma HLS bind_op variable=s6 op=fadd impl=fulldsp
#pragma HLS bind_op variable=s7 op=fadd impl=fulldsp
    float a0 = s0 + s1;
    float a1 = s2 + s3;
    float a2 = s4 + s5;
    float a3 = s6 + s7;
    float b0 = a0 + a1;
    float b1 = a2 + a3;
#pragma HLS bind_op variable=a0 op=fadd impl=fulldsp
#pragma HLS bind_op variable=a1 op=fadd impl=fulldsp
#pragma HLS bind_op variable=a2 op=fadd impl=fulldsp
#pragma HLS bind_op variable=a3 op=fadd impl=fulldsp
#pragma HLS bind_op variable=b0 op=fadd impl=fulldsp
#pragma HLS bind_op variable=b1 op=fadd impl=fulldsp
    return b0 + b1;
}

static void gemv32_pair_dot16(const Beat512 &w0, const Beat512 &w1,
                              uint32_t weight_lane_base, const Beat512 &xv,
                              float &d0, float &d1) {
#pragma HLS inline
    float p0[16], p1[16];
#pragma HLS array_partition variable=p0 complete
#pragma HLS array_partition variable=p1 complete
gemv32_pair_mul: for (int i = 0; i < 16; ++i) {
#pragma HLS unroll
        /* Iter66: native ap_float<16,8> multiplication deliberately rounds the
         * scalar product to BF16. The BF16 result widens by bit placement into
         * the unchanged FP32 tree. No FP32 multiplier or hand-built normalize
         * network is present in the synthesized GEMV product path. */
        const Bf16Bits xa =
            get_bf16_lane(xv, weight_lane_base + i);
        p0[i] = gdn_native_bf16_mul_to_fp32(
            get_bf16_lane(w0, weight_lane_base + i), xa);
        p1[i] = gdn_native_bf16_mul_to_fp32(
            get_bf16_lane(w1, weight_lane_base + i), xa);
    }
    d0 = gemv32_tree16(p0);
    d1 = gemv32_tree16(p1);
}

/* One two-port product engine.  The eight row contexts in gemv32_cluster2
 * call this pipeline on successive clocks; keeping it out of line and
 * limiting it to one instance prevents HLS from cloning the exact mixed
 * multiplier body once per unrolled context. */
/* Two compile-time-distinct half-dots, each `inline off`, so HLS emits two
 * separate RTL modules the placer can spread independently. One `inline off`
 * function called twice would be *shared* -- one instance at II=2 -- not
 * duplicated, which is why these are distinct functions rather than one taking
 * a lane-base argument.
 *
 * This is the direct answer to Iter62b's failure: report_design_analysis named
 * the replicated `gemv32_four_dots` hierarchy as a leading contributor in every
 * level-7 congestion window, so the monolithic 64-product cone is split in two. */
/* Iter63b measured the cost of making these separate modules: `inline off` on
 * both halves added **+17,488 FF** design-wide (the isolated cluster predicted
 * +1,195 x 16 = 19,120), and hardware build 721 then failed *placement* at
 * 7h14m with 40 shell cells unplaced in SLR1 -- `axi_gpio_null` and
 * `interconnect_axilite_user`. SLR1 was already at 94.58% CLB in Iter62b, so
 * the boundary registers pushed the platform's own control logic out.
 *
 * The hierarchy is what would have let the placer spread the 27,764-LUT
 * `gemv32_four_dots` cone that report_design_analysis named, but hierarchy costs
 * boundary registers, and this design cannot afford them in SLR1. Inlined here,
 * so the split survives as code structure without the physical cost. */
static void gemv32_half_dot_low(const Beat512 &weight0,
                                const Beat512 &weight1,
                                const Beat512 &activation,
                                float &d0_lo, float &d1_lo) {
#pragma HLS inline
    gemv32_pair_dot16(weight0, weight1, 0, activation, d0_lo, d1_lo);
}

static void gemv32_half_dot_high(const Beat512 &weight0,
                                 const Beat512 &weight1,
                                 const Beat512 &activation,
                                 float &d0_hi, float &d1_hi) {
#pragma HLS inline
    gemv32_pair_dot16(weight0, weight1, 16, activation, d0_hi, d1_hi);
}

static void gemv32_four_dots(const Beat512 &weight0,
                             const Beat512 &weight1,
                             const Beat512 &activation,
                             float &d0_lo, float &d0_hi,
                             float &d1_lo, float &d1_hi) {
#pragma HLS inline off
#pragma HLS pipeline II=1
    gemv32_half_dot_low (weight0, weight1, activation, d0_lo, d1_lo);
    gemv32_half_dot_high(weight0, weight1, activation, d0_hi, d1_hi);
}

static float gemv32_reduce_parts(float p0, float p1, float p2, float p3) {
#pragma HLS inline
    float s0 = p0 + p1;
    float s1 = p2 + p3;
    float total = s0 + s1;
    /* Iter65: keep the shared four-part reduction in DSPs.  Integrated
     * csynth shares six fadd cores between steady retirement and final flush,
     * saving 680 LUTs per cluster without changing II, cycles, or Fmax. */
#pragma HLS bind_op variable=s0 op=fadd impl=fulldsp
#pragma HLS bind_op variable=s1 op=fadd impl=fulldsp
#pragma HLS bind_op variable=total op=fadd impl=fulldsp
    return total;
}

/* The packed rows revisit one logical accumulator every eight clocks.  Rotate
 * the eight contexts through fixed scalar registers so the current context is
 * always slot zero.  This exposes the true distance-eight FP32 recurrence to
 * HLS without a runtime-indexed mux or eight compile-time copies of the MAC
 * control cone. */
static void gemv32_rotate_context(float ring[8], float next) {
#pragma HLS inline
#pragma HLS array_partition variable=ring complete
    const float oldest = ring[0];
gemv32_rotate_context_slot: for (int slot = 0; slot < 7; ++slot) {
#pragma HLS unroll
        ring[slot] = ring[slot + 1];
    }
    (void)oldest;
    ring[7] = next;
}

static void gemv32_load_x_and_w0(const Beat512 *x, const Beat512 *w0,
                                 size_t weight_base,
                                 hls::stream<Beat512> &xr,
                                 hls::stream<Beat512> &ws0,
                                 uint32_t k_packs, uint32_t n_packs) {
#pragma HLS inline off
gemv32_lx: for (uint32_t kp = 0; kp < k_packs; ++kp) {
#pragma HLS loop_tripcount min=128 max=352
#pragma HLS pipeline II=1
        xr.write(x[kp]);
    }
gemv32_w0: for (uint32_t i = 0; i < n_packs; ++i) {
#pragma HLS loop_tripcount min=4096 max=64000
#pragma HLS pipeline II=1
        ws0.write(w0[weight_base + i]);
    }
}

template <int CHANNEL>
static void gemv32_mm2s(const Beat512 *w, size_t base,
                        hls::stream<Beat512> &ws, uint32_t n_packs) {
#pragma HLS inline off
    (void)CHANNEL;
gemv32_mm2s_loop: for (uint32_t i = 0; i < n_packs; ++i) {
#pragma HLS loop_tripcount min=4096 max=64000
#pragma HLS pipeline II=1
        ws.write(w[base + i]);
    }
}

/* Ports 28--31 own both one GEMV shard and one packed recurrent-state stripe.
 * HLS permits only one read process per bundled m_axi interface, so this actor
 * is the sole reader for both address ranges. In QKVG mode it emits one head's
 * weights, then emits that head's 512 BF16 state words while the collectors and
 * convolution actor finish the head. Iter40C drains a shallow BRAM decoupler
 * into the recurrent actor's existing head-local state buffer before the MAC
 * pass, so no second whole-head FIFO or stream-controlled MAC cone is needed. */
template <int CHANNEL>
static void gemv32_mm2s_with_state(
    const Beat512 *w,
    const Beat512 *state,
    size_t weight_base,
    hls::stream<Beat512> &ws,
    hls::stream<Beat512> &state_stream,
    uint32_t n_packs,
    uint32_t layer_index,
    bool qkvg_recurrent_mode
) {
#pragma HLS inline off
    (void)CHANNEL;
    if (!qkvg_recurrent_mode) {
    gemv32_state_owner_weight_only: for (uint32_t i = 0;
                                          i < n_packs; ++i) {
#pragma HLS loop_tripcount min=4096 max=64000
#pragma HLS pipeline II=1
            ws.write(w[weight_base + i]);
        }
        return;
    }

    const uint32_t weight_packs_per_head = n_packs / GDN_HEADS;
    const uint32_t state_packs_per_head =
        GDN_DK * (GDN_DV / 32) / GDN_RECURRENT_STATE_PORTS;
gemv32_state_owner_head: for (uint32_t head = 0;
                               head < GDN_HEADS; ++head) {
#pragma HLS loop_tripcount min=8 max=8
    gemv32_state_owner_weight: for (uint32_t i = 0;
                                     i < weight_packs_per_head; ++i) {
#pragma HLS loop_tripcount min=2048 max=2048
#pragma HLS pipeline II=1
            ws.write(w[weight_base + head * weight_packs_per_head + i]);
        }

        /* The per-owner state FIFO holds this complete 512-word burst.
         * Consequently the owner can always advance to the next head's
         * weights, whose first row flushes the cluster's pending final row
         * for this head. This forward-only schedule avoids a credit cycle. */
        size_t state_base = ((size_t)layer_index * GDN_HEADS + head)
                          * state_packs_per_head;
    gemv32_state_owner_prefetch: for (uint32_t i = 0;
                                       i < state_packs_per_head; ++i) {
#pragma HLS loop_tripcount min=512 max=512
#pragma HLS pipeline II=1
            state_stream.write(state[state_base + i]);
        }
    }
}

static void gemv32_drain_x(hls::stream<Beat512> &xr, uint32_t k_packs) {
#pragma HLS inline off
gemv32_dx: for (uint32_t kp = 0; kp < k_packs; ++kp) {
#pragma HLS loop_tripcount min=128 max=352
#pragma HLS pipeline II=1
        (void)xr.read();
    }
}

static void gemv32_cluster2(hls::stream<Beat512> &ws0,
                            hls::stream<Beat512> &ws1,
                            hls::stream<Beat512> &x_in,
                            hls::stream<Beat512> &x_out,
                            hls::stream<Beat512> &ys,
                            uint32_t k_packs, uint32_t rows_per_ch) {
#pragma HLS inline off
#pragma HLS allocation function instances=gemv32_four_dots limit=1
    /* One BF16 activation beat (32 lanes) now matches one weight beat, so the
     * even/odd pair collapses to a single array and the ripple carries half the
     * beats: in_dim/32 rather than in_dim/16. */
    Beat512 x_bf16[IN_DIM_MAX / 32];
#pragma HLS bind_storage variable=x_bf16 type=ram_1p impl=bram
    gemv32_cl_load: for (uint32_t kp = 0; kp < k_packs; ++kp) {
#pragma HLS loop_tripcount min=64 max=176
#pragma HLS pipeline II=1
        Beat512 v = x_in.read();
        x_out.write(v);
        x_bf16[kp] = v;
    }

    float p00[8], p01[8], p02[8], p03[8];
    float p10[8], p11[8], p12[8], p13[8];
#pragma HLS array_partition variable=p00 complete
#pragma HLS array_partition variable=p01 complete
#pragma HLS array_partition variable=p02 complete
#pragma HLS array_partition variable=p03 complete
#pragma HLS array_partition variable=p10 complete
#pragma HLS array_partition variable=p11 complete
#pragma HLS array_partition variable=p12 complete
#pragma HLS array_partition variable=p13 complete
gemv32_cl_init_contexts: for (int slot = 0; slot < 8; ++slot) {
#pragma HLS pipeline II=1
        p00[slot] = 0.0f;
        p01[slot] = 0.0f;
        p02[slot] = 0.0f;
        p03[slot] = 0.0f;
        p10[slot] = 0.0f;
        p11[slot] = 0.0f;
        p12[slot] = 0.0f;
        p13[slot] = 0.0f;
    }
    Beat512 yp0 = 0;
    Beat512 yp1 = 0;
    /* One BF16 activation beat per weight beat (was two FP32 beats each). */
    const uint32_t weight_beats_per_row = k_packs;
    const uint32_t row_groups = rows_per_ch / 8;
    const uint32_t total_weight_beats = rows_per_ch * weight_beats_per_row;
    uint32_t group = 0;
    uint32_t wb = 0;
    uint32_t context = 0;

gemv32_cl_weight_stream: for (uint32_t flat = 0;
                               flat < total_weight_beats; ++flat) {
#pragma HLS loop_tripcount min=4096 max=64000
/* Iter66e probe: free-running pipeline. The per-stage clock-enable network
 * HLS generates for a standard pipeline reaches 5.1K-9.3K loads per cluster
 * (measured, diagnosis 1164) and its cones sat in every routing-failure
 * conflict set from Iter66b through Iter66d. style=frp keeps the datapath
 * always running with a valid pipeline instead, removing that CE network at
 * the source. Native semantics are unchanged; csynth decides eligibility. */
#pragma HLS pipeline II=1 style=frp
        const Beat512 activation = x_bf16[wb];
        bool emit_result = false;
        Beat512 emitted_word = 0;

        /* Retire the previous group before slot zero is reset for the current
         * row.  Shift-and-insert packs sequential rows without a variable
         * 512-bit lane mux. */
        if (group != 0 && wb == 0) {
            if (context == 0 && ((group - 1) & 1) == 0) {
                yp0 = 0;
                yp1 = 0;
            }
            const float result0 = gemv32_reduce_parts(
                p00[0], p01[0], p02[0], p03[0]);
            const float result1 = gemv32_reduce_parts(
                p10[0], p11[0], p12[0], p13[0]);
            yp0 >>= 32;
            yp1 >>= 32;
            set_fp32_lane(yp0, 15, result0);
            set_fp32_lane(yp1, 15, result1);
            /* A cluster emits port-zero then port-one for each output pack.
             * Use the otherwise idle following weight cycle for the second
             * write so the steady loop still requests only one AXIS write. */
            if (context == 7 && (group & 1) == 0) {
                emit_result = true;
                emitted_word = yp0;
            }
        }
        if (group >= 2 && (group & 1) == 0 && wb == 1 && context == 0) {
            emit_result = true;
            emitted_word = yp1;
        }
        if (emit_result)
            ys.write(emitted_word);

        const Beat512 weight0 = ws0.read();
        const Beat512 weight1 = ws1.read();
        float d0_lo, d0_hi, d1_lo, d1_hi;
        gemv32_four_dots(weight0, weight1, activation,
                         d0_lo, d0_hi, d1_lo, d1_hi);

        const bool even_bank = (wb & 1) == 0;
        const bool first_for_bank = wb < 2;
        const float old0_lo = even_bank ? p00[0] : p02[0];
        const float old0_hi = even_bank ? p01[0] : p03[0];
        const float old1_lo = even_bank ? p10[0] : p12[0];
        const float old1_hi = even_bank ? p11[0] : p13[0];
        float next0_lo = (first_for_bank ? 0.0f : old0_lo) + d0_lo;
        float next0_hi = (first_for_bank ? 0.0f : old0_hi) + d0_hi;
        float next1_lo = (first_for_bank ? 0.0f : old1_lo) + d1_lo;
        float next1_hi = (first_for_bank ? 0.0f : old1_hi) + d1_hi;
#pragma HLS bind_op variable=next0_lo op=fadd impl=fulldsp
#pragma HLS bind_op variable=next0_hi op=fadd impl=fulldsp
#pragma HLS bind_op variable=next1_lo op=fadd impl=fulldsp
#pragma HLS bind_op variable=next1_hi op=fadd impl=fulldsp

        gemv32_rotate_context(p00, even_bank ? next0_lo : p00[0]);
        gemv32_rotate_context(p01, even_bank ? next0_hi : p01[0]);
        gemv32_rotate_context(p02, even_bank ? p02[0] : next0_lo);
        gemv32_rotate_context(p03, even_bank ? p03[0] : next0_hi);
        gemv32_rotate_context(p10, even_bank ? next1_lo : p10[0]);
        gemv32_rotate_context(p11, even_bank ? next1_hi : p11[0]);
        gemv32_rotate_context(p12, even_bank ? p12[0] : next1_lo);
        gemv32_rotate_context(p13, even_bank ? p13[0] : next1_hi);

        if (context == 7) {
            context = 0;
            if (wb + 1 == weight_beats_per_row) {
                wb = 0;
                group++;
            } else {
                wb++;
            }
        } else {
            context++;
        }
    }

    const uint32_t final_group = row_groups - 1;
    if ((final_group & 1) == 0) {
        yp0 = 0;
        yp1 = 0;
    }
gemv32_cl_flush: for (uint32_t context = 0; context < 8; ++context) {
#pragma HLS loop_tripcount min=8 max=8
        const float result0 = gemv32_reduce_parts(
            p00[0], p01[0], p02[0], p03[0]);
        const float result1 = gemv32_reduce_parts(
            p10[0], p11[0], p12[0], p13[0]);
        yp0 >>= 32;
        yp1 >>= 32;
        set_fp32_lane(yp0, 15, result0);
        set_fp32_lane(yp1, 15, result1);
        gemv32_rotate_context(p00, p00[0]);
        gemv32_rotate_context(p01, p01[0]);
        gemv32_rotate_context(p02, p02[0]);
        gemv32_rotate_context(p03, p03[0]);
        gemv32_rotate_context(p10, p10[0]);
        gemv32_rotate_context(p11, p11[0]);
        gemv32_rotate_context(p12, p12[0]);
        gemv32_rotate_context(p13, p13[0]);
    }
    if ((final_group & 1) == 0) {
        /* An odd number of eight-row groups leaves only lanes 0--7 valid in
         * the final logical output Beat.  The shift-in packer temporarily
         * holds those values in lanes 8--15; move them down before padding. */
        yp0 >>= 256;
        yp1 >>= 256;
    }
    ys.write(yp0);
    ys.write(yp1);
}

#ifdef GDN_CLUSTER_HLS_TEST
void gdn_packed_cluster_hls_top(hls::stream<Beat512> &ws0,
                                hls::stream<Beat512> &ws1,
                                hls::stream<Beat512> &x_in,
                                hls::stream<Beat512> &x_out,
                                hls::stream<Beat512> &ys,
                                uint32_t k_packs,
                                uint32_t rows_per_ch) {
    gemv32_cluster2(ws0, ws1, x_in, x_out, ys, k_packs, rows_per_ch);
}
#endif

static void gemv32_collect6(hls::stream<Beat512> &ys0,
                            hls::stream<Beat512> &ys1,
                            hls::stream<Beat512> &ys2,
                            hls::stream<Beat512> &ys3,
                            hls::stream<Beat512> &ys4,
                            hls::stream<Beat512> &ys5,
                            hls::stream<Beat512> &local,
                            uint32_t opacks_per_ch) {
#pragma HLS inline off
gemv32_c6_p: for (uint32_t p = 0; p < opacks_per_ch; ++p) {
#pragma HLS loop_tripcount min=4 max=63
    gemv32_c6_a: for (int i = 0; i < 2; ++i) {
#pragma HLS pipeline II=1
            local.write(ys0.read());
        }
    gemv32_c6_b: for (int i = 0; i < 2; ++i) {
#pragma HLS pipeline II=1
            local.write(ys1.read());
        }
    gemv32_c6_c: for (int i = 0; i < 2; ++i) {
#pragma HLS pipeline II=1
            local.write(ys2.read());
        }
    gemv32_c6_d: for (int i = 0; i < 2; ++i) {
#pragma HLS pipeline II=1
            local.write(ys3.read());
        }
    gemv32_c6_e: for (int i = 0; i < 2; ++i) {
#pragma HLS pipeline II=1
            local.write(ys4.read());
        }
    gemv32_c6_f: for (int i = 0; i < 2; ++i) {
#pragma HLS pipeline II=1
            local.write(ys5.read());
        }
    }
}

static void gemv32_collect4(hls::stream<Beat512> &ys0,
                            hls::stream<Beat512> &ys1,
                            hls::stream<Beat512> &ys2,
                            hls::stream<Beat512> &ys3,
                            hls::stream<Beat512> &local,
                            uint32_t opacks_per_ch) {
#pragma HLS inline off
gemv32_c4_p: for (uint32_t p = 0; p < opacks_per_ch; ++p) {
#pragma HLS loop_tripcount min=4 max=63
    gemv32_c4_a: for (int i = 0; i < 2; ++i) {
#pragma HLS pipeline II=1
            local.write(ys0.read());
        }
    gemv32_c4_b: for (int i = 0; i < 2; ++i) {
#pragma HLS pipeline II=1
            local.write(ys1.read());
        }
    gemv32_c4_c: for (int i = 0; i < 2; ++i) {
#pragma HLS pipeline II=1
            local.write(ys2.read());
        }
    gemv32_c4_d: for (int i = 0; i < 2; ++i) {
#pragma HLS pipeline II=1
            local.write(ys3.read());
        }
    }
}

/* Materialize a real II=1 register stage at each local-collector boundary.
 * The physical hook places these small actors in SLR1 and marks their
 * sequential leaves as USER_SLL_REG, giving SSI placement an explicit
 * destination register instead of a direct BRAM/control crossing. */
template <int RELAY>
static void gemv32_boundary_relay(hls::stream<Beat512> &source,
                                  hls::stream<Beat512> &destination,
                                  uint32_t words) {
#pragma HLS inline off
    (void)RELAY;
gemv32_boundary_relay_word: for (uint32_t word = 0; word < words; ++word) {
#pragma HLS loop_tripcount min=32 max=756
#pragma HLS pipeline II=1
        Beat512 value = source.read();
        destination.write(value);
    }
}

static void gemv32_collect_final(hls::stream<Beat512> &slr0,
                                 hls::stream<Beat512> &slr1,
                                 hls::stream<Beat512> &slr2,
                                 hls::stream<Beat512> &result,
                                 uint32_t opacks_per_ch) {
#pragma HLS inline off
gemv32_cf_p: for (uint32_t p = 0; p < opacks_per_ch; ++p) {
#pragma HLS loop_tripcount min=4 max=63
gemv32_cf_0: for (int i = 0; i < 8; ++i) {
#pragma HLS pipeline II=1
            result.write(slr0.read());
        }
gemv32_cf_1: for (int i = 0; i < 12; ++i) {
#pragma HLS pipeline II=1
            result.write(slr1.read());
        }
gemv32_cf_2: for (int i = 0; i < 12; ++i) {
#pragma HLS pipeline II=1
            result.write(slr2.read());
        }
    }
}

/* The routed microbenchmark emits pack-major/channel-minor results. Full GDN
 * requires ordinary row-major activations, so buffer the small result tensor
 * in URAM and restore the original layout. */
static void gemv32_store(hls::stream<Beat512> &result, Beat512 *out,
                         hls::stream<Beat512> &logits_stream,
                         uint32_t rows_per_ch, uint32_t opacks_per_ch,
                         uint32_t total_opacks) {
#pragma HLS inline off
    Beat512 reorder[GEMV32_MAX_RESULT_PACKS];
#pragma HLS bind_storage variable=reorder type=ram_2p impl=uram
gemv32_store_fill: for (uint32_t i = 0; i < total_opacks; ++i) {
#pragma HLS loop_tripcount min=128 max=2016
#pragma HLS pipeline II=1
        reorder[i] = result.read();
    }

    if (rows_per_ch == GDN_VOCAB / GEMV_CHANNELS) {
        /* Iter61: emit the full GDN_VOCAB logit vector into a stream that the
         * top level drains and writes to HBM. This branch is entered only by
         * the LM head (rows_per_ch == 1000), so every other GEMV call pushes
         * nothing and needs no special casing.
         *
         * The point of the stream is that this block gains no AXI port. Iter58b
         * and Iter59 gave gemv32_store its own m_axi write pointer; Iter59 then
         * failed route_design at global congestion level 7. gemv32_store is
         * distributed across all three SLRs by the island pblocks, so a memory
         * port here is a memory port everywhere. Filling a FIFO instead keeps
         * every AXI access at the top level, beside the token-id write that
         * Iter57 already performs and routes.
         *
         * Natural row order: channel c owns rows [c*rows_per_ch, (c+1)*...),
         * and GDN_VOCAB is a multiple of 16, so the vector is a whole number of
         * Beat512 lines with no partial line.  Each 1000-row stripe comprises
         * 62 complete FP32 Beats plus one eight-lane tail.  Process adjacent
         * channel pairs sequentially: stream the even channel's full Beats,
         * then stitch its tail to the odd channel with a 256-bit carry.  This
         * uses one sequential URAM read per cycle instead of 16 unrolled,
         * dynamically indexed reads from a two-port memory.
         *
         * Iter63 retires the argmax that used to follow this loop, so this is
         * now the only consumer of the reorder buffer and the sole LM-head
         * output. The token trajectory is unchanged: the host applies the same
         * maximum-value / lowest-index-on-ties rule. */
#ifndef __SYNTHESIS__
        uint32_t debug_index = 0;
#endif
        /* Iter67: the greedy pick is fused back into the emission loops rather
         * than restored as Iter61's separate pass. The stream below already
         * visits every logit exactly once in natural vocabulary order, so the
         * argmax costs a 16-wide compare per beat and no extra cycles at all
         * -- Iter63 removed a second 2,016-iteration sweep of the reorder
         * buffer, and nothing here reinstates it.
         *
         * Sixteen independent lane accumulators keep the loop-carried
         * dependency to one compare per lane instead of a tree, which is what
         * let Iter61 hold II=1 in this block. Indices increase monotonically
         * within a lane, so strict '>' keeps the lowest index on ties; the
         * merge then applies the same rule across lanes. This is exactly the
         * host's maximum-value / lowest-index rule, so the trajectory is
         * unchanged and the native gate stays bit-exact. */
        float lane_best[16];
        uint32_t lane_best_index[16];
#pragma HLS array_partition variable=lane_best complete dim=1
#pragma HLS array_partition variable=lane_best_index complete dim=1
    gemv32_argmax_init: for (uint32_t lane = 0; lane < 16; ++lane) {
#pragma HLS unroll
            lane_best[lane] = -3.402823466e38f;
            lane_best_index[lane] = 0;
        }

        gemv32_logits_channel_pair: for (uint32_t pair = 0;
                                             pair < GEMV_CHANNELS / 2;
                                             ++pair) {
#pragma HLS loop_tripcount min=16 max=16
            const uint32_t even_channel = pair * 2;
            const uint32_t odd_channel = even_channel + 1;

        gemv32_logits_even_full: for (uint32_t p = 0;
                                           p < (GDN_VOCAB / GEMV_CHANNELS) / 16;
                                           ++p) {
#pragma HLS loop_tripcount min=62 max=62
#pragma HLS pipeline II=1
                const Beat512 line =
                    reorder[(size_t)p * GEMV_CHANNELS + even_channel];
#ifndef __SYNTHESIS__
                if (gdn_native_logits_debug != NULL) {
                    for (uint32_t lane = 0; lane < 16; ++lane) {
                        gdn_native_logits_debug[debug_index + lane] =
                            get_fp32_lane(line, lane);
                    }
                }
                debug_index += 16;
#endif
                /* Emitted beat number is pair*125 + p: each pair contributes
                 * 62 even-channel beats plus 63 stitched ones, and 2 x 1000
                 * rows is exactly 125 whole Beat512 lines. Global vocabulary
                 * index is therefore emitted_beat*16 + lane. */
                const uint32_t even_base = (pair * 125u + p) * 16u;
            gemv32_argmax_even: for (uint32_t lane = 0; lane < 16; ++lane) {
#pragma HLS unroll
                    const float candidate = get_fp32_lane(line, lane);
                    if (candidate > lane_best[lane]) {
                        lane_best[lane] = candidate;
                        lane_best_index[lane] = even_base + lane;
                    }
                }
                logits_stream.write(line);
            }

            const Beat512 even_tail = reorder[
                (size_t)((GDN_VOCAB / GEMV_CHANNELS) / 16) *
                    GEMV_CHANNELS + even_channel];
            ap_uint<256> carry = even_tail.range(255, 0);

        gemv32_logits_odd_stitch: for (uint32_t p = 0;
                                            p < ((GDN_VOCAB /
                                                  GEMV_CHANNELS) + 15) / 16;
                                            ++p) {
#pragma HLS loop_tripcount min=63 max=63
#pragma HLS pipeline II=1
                const Beat512 source =
                    reorder[(size_t)p * GEMV_CHANNELS + odd_channel];
                Beat512 line = 0;
                line.range(255, 0) = carry;
                line.range(511, 256) = source.range(255, 0);
#ifndef __SYNTHESIS__
                if (gdn_native_logits_debug != NULL) {
                    for (uint32_t lane = 0; lane < 16; ++lane) {
                        gdn_native_logits_debug[debug_index + lane] =
                            get_fp32_lane(line, lane);
                    }
                }
                debug_index += 16;
#endif
                const uint32_t odd_base = (pair * 125u + 62u + p) * 16u;
            gemv32_argmax_odd: for (uint32_t lane = 0; lane < 16; ++lane) {
#pragma HLS unroll
                    const float candidate = get_fp32_lane(line, lane);
                    if (candidate > lane_best[lane]) {
                        lane_best[lane] = candidate;
                        lane_best_index[lane] = odd_base + lane;
                    }
                }
                logits_stream.write(line);
                if (p + 1 < ((GDN_VOCAB / GEMV_CHANNELS) + 15) / 16) {
                    carry = source.range(511, 256);
                }
            }
        }

        /* Cross-lane merge: lower index wins a tie, matching the host rule. */
        float best = lane_best[0];
        uint32_t best_index = lane_best_index[0];
    gemv32_argmax_merge: for (uint32_t lane = 1; lane < 16; ++lane) {
            const float candidate = lane_best[lane];
            const uint32_t candidate_index = lane_best_index[lane];
            if (candidate > best ||
                (candidate == best && candidate_index < best_index)) {
                best = candidate;
                best_index = candidate_index;
            }
        }
        /* Lane 0 of out[0] carries the token id. The LM-head branch writes
         * nothing else to `out`, and the top level copies this one lane to the
         * workspace token slot -- no AXI port is added inside this block,
         * which is the constraint Iter58b and Iter59 violated. */
        out[0] = 0;
        set_fp32_lane(out[0], 0, (float)best_index);
        return;
    }

    if ((rows_per_ch & 31) == 0) {
    gemv32_store_c: for (uint32_t c = 0; c < GEMV_CHANNELS; ++c) {
        gemv32_store_p: for (uint32_t p = 0; p < opacks_per_ch / 2; ++p) {
#pragma HLS loop_tripcount min=2 max=11
#pragma HLS pipeline II=1
                const Beat512 lo =
                    reorder[(size_t)(2 * p) * GEMV_CHANNELS + c];
                const Beat512 hi =
                    reorder[(size_t)(2 * p + 1) * GEMV_CHANNELS + c];
                Beat512 packed = 0;
            gemv32_store_bf16_lane: for (uint32_t lane = 0; lane < 16; ++lane) {
#pragma HLS unroll
                    set_bf16_lane(packed, lane,
                        fp32_to_bf16_rne(get_fp32_lane(lo, lane)));
                    set_bf16_lane(packed, lane + 16,
                        fp32_to_bf16_rne(get_fp32_lane(hi, lane)));
                }
                size_t out_index = (size_t)c * (opacks_per_ch / 2) + p;
                out[out_index] = packed;
            }
        }
        return;
    }

    /* All synthesized non-QKVG callers have 32-row-aligned channel stripes:
     * output/down use 64 and gate/up uses 352.  QKVG has its own streaming
     * store, while the only unaligned shape (LM-head 1000) returned above.
     * Keeping the former generic scalar fallback synthesized a large dynamic
     * read/modify/write cone that production can never reach. */
}

/* Head-streamed QKVG producer. Normal GEMVs retain the routed URAM
 * reorder/store path. In QKVG mode, two pack-major collector rounds contain a
 * complete head. Convolve that head and emit three bounded streams to the
 * independent recurrent actor while the GEMV produces later heads. */
static void gemv32_store_or_qkvg_conv_stream(
    hls::stream<Beat512> &result,
    hls::stream<Beat512> &q_stream,
    hls::stream<Beat512> &k_stream,
    hls::stream<Beat512> &v_stream,
    Beat512 *out,
    hls::stream<Beat512> &logits_stream,
    uint32_t rows_per_ch,
    uint32_t opacks_per_ch,
    uint32_t total_opacks,
    bool qkvg_recurrent_mode,
    Beat512 *gate_out,
    const Beat512 conv_weights[3][(GDN_HIDDEN * GDN_CONV) / 16],
    Beat512 conv_tails[3][((GDN_CONV - 1) * GDN_HIDDEN) / 32]
) {
#pragma HLS inline off
#pragma HLS aggregate variable=conv_weights compact=bit
#pragma HLS aggregate variable=conv_tails compact=bit
    if (!qkvg_recurrent_mode) {
        gemv32_store(result, out, logits_stream, rows_per_ch, opacks_per_ch,
                     total_opacks);
        return;
    }

    Beat512 head_fp32[4][GDN_HEAD_DIM / 16];
    Beat512 head_value[4][GDN_HEAD_DIM / 32];
    Beat512 convolved_head[GDN_HEAD_DIM / 32];
#pragma HLS array_partition variable=head_fp32 complete dim=1
#pragma HLS array_partition variable=head_value complete dim=1
qkvg_stream_head: for (uint32_t head = 0; head < GDN_HEADS; ++head) {
#pragma HLS loop_tripcount min=8 max=8
    qkvg_stream_half: for (uint32_t half = 0; half < 2; ++half) {
        qkvg_stream_channel: for (uint32_t channel = 0;
                                  channel < GEMV_CHANNELS; ++channel) {
#pragma HLS loop_tripcount min=32 max=32
#pragma HLS pipeline II=1
                uint32_t kind = channel / (GEMV_CHANNELS / 4);
                uint32_t segment = channel % (GEMV_CHANNELS / 4);
                head_fp32[kind][segment * 2 + half] = result.read();
            }
        }

    qkvg_stream_pack_kind: for (uint32_t kind = 0; kind < 4; ++kind) {
        qkvg_stream_pack: for (uint32_t p = 0;
                               p < GDN_HEAD_DIM / 32; ++p) {
#pragma HLS loop_tripcount min=8 max=8
#pragma HLS pipeline II=1
                const Beat512 lo = head_fp32[kind][2 * p];
                const Beat512 hi = head_fp32[kind][2 * p + 1];
                Beat512 packed = 0;
            qkvg_stream_pack_lane: for (uint32_t lane = 0; lane < 16; ++lane) {
#pragma HLS unroll
                    set_bf16_lane(packed, lane,
                        fp32_to_bf16_rne(get_fp32_lane(lo, lane)));
                    set_bf16_lane(packed, lane + 16,
                        fp32_to_bf16_rne(get_fp32_lane(hi, lane)));
                }
                head_value[kind][p] = packed;
            }
        }

    qkvg_stream_gate_store: for (uint32_t p = 0;
                                     p < GDN_HEAD_DIM / 32; ++p) {
#pragma HLS loop_tripcount min=8 max=8
#pragma HLS pipeline II=1
            gate_out[head * (GDN_HEAD_DIM / 32) + p] = head_value[3][p];
        }

    qkvg_stream_conv_kind: for (uint32_t kind = 0; kind < 3; ++kind) {
#pragma HLS loop_tripcount min=3 max=3
            gdn_depthwise_conv_silu_head_kind(
                convolved_head, head_value,
                conv_weights, conv_tails, head, kind);
        qkvg_stream_conv_emit: for (uint32_t p = 0;
                                     p < GDN_HEAD_DIM / 32; ++p) {
#pragma HLS loop_tripcount min=8 max=8
#pragma HLS pipeline II=1
                Beat512 value = convolved_head[p];
                if (kind == 0)
                    q_stream.write(value);
                else if (kind == 1)
                    k_stream.write(value);
                else
                    v_stream.write(value);
            }
        }

        /* All three actors have consumed this head's old three-row context.
         * Reuse tail row 0 for the new raw row; the final packed store emits
         * old rows 1/2 followed by this row without another BRAM buffer. */
    qkvg_stream_capture_new_tail: for (uint32_t p = 0;
                                           p < GDN_HEAD_DIM / 32; ++p) {
#pragma HLS loop_tripcount min=8 max=8
#pragma HLS pipeline II=1
            uint32_t destination = head * (GDN_HEAD_DIM / 32) + p;
            conv_tails[0][destination] = head_value[0][p];
            conv_tails[1][destination] = head_value[1][p];
            conv_tails[2][destination] = head_value[2][p];
        }
    }
}

/* Decode GEMV with 32 compact weight shards on independent HBM masters. Sixteen
 * two-port clusters consume private activation copies and feed hierarchical
 * collectors. The store stage restores natural output-row order; it also
 * handles the lm_head's partial final pack (1000 rows per channel). */
static void gdn_gemv(
    Beat512 *out, hls::stream<Beat512> &logits_stream, const Beat512 *in,
    const Beat512 *w0, const Beat512 *w1, const Beat512 *w2, const Beat512 *w3,
    const Beat512 *w4, const Beat512 *w5, const Beat512 *w6, const Beat512 *w7,
    const Beat512 *w8, const Beat512 *w9, const Beat512 *w10, const Beat512 *w11,
    const Beat512 *w12, const Beat512 *w13, const Beat512 *w14, const Beat512 *w15,
    const Beat512 *w16, const Beat512 *w17, const Beat512 *w18, const Beat512 *w19,
    const Beat512 *w20, const Beat512 *w21, const Beat512 *w22, const Beat512 *w23,
    const Beat512 *w24, const Beat512 *w25, const Beat512 *w26, const Beat512 *w27,
    Beat512 *w28, Beat512 *w29, Beat512 *w30, Beat512 *w31,
    uint32_t shard_off,
    uint32_t in_dim, uint32_t out_dim,
    bool qkvg_recurrent_mode,
    Beat512 *attn_out, Beat512 *gate_out,
    const Beat512 conv_weights[3][(GDN_HIDDEN * GDN_CONV) / 16],
    Beat512 conv_tails[3][((GDN_CONV - 1) * GDN_HIDDEN) / 32],
    const float a[GDN_HEADS],
    const float b[GDN_HEADS],
    const float layer_a_log[GDN_HEADS],
    const float layer_dt_bias[GDN_HEADS],
    uint32_t layer_index
) {
    #pragma HLS inline off

    const Beat512 *sh0 = w0;
    const Beat512 *sh1 = w1;
    const Beat512 *sh2 = w2;
    const Beat512 *sh3 = w3;
    const Beat512 *sh4 = w4;
    const Beat512 *sh5 = w5;
    const Beat512 *sh6 = w6;
    const Beat512 *sh7 = w7;
    const Beat512 *sh8 = w8;
    const Beat512 *sh9 = w9;
    const Beat512 *sh10 = w10;
    const Beat512 *sh11 = w11;
    const Beat512 *sh12 = w12;
    const Beat512 *sh13 = w13;
    const Beat512 *sh14 = w14;
    const Beat512 *sh15 = w15;
    const Beat512 *sh16 = w16;
    const Beat512 *sh17 = w17;
    const Beat512 *sh18 = w18;
    const Beat512 *sh19 = w19;
    const Beat512 *sh20 = w20;
    const Beat512 *sh21 = w21;
    const Beat512 *sh22 = w22;
    const Beat512 *sh23 = w23;
    const Beat512 *sh24 = w24;
    const Beat512 *sh25 = w25;
    const Beat512 *sh26 = w26;
    const Beat512 *sh27 = w27;
    const Beat512 *sh28 = w28;
    const Beat512 *sh29 = w29;
    const Beat512 *sh30 = w30;
    const Beat512 *sh31 = w31;
    const Beat512 *state_in28 = w28 + GDN_COMPILED_WEIGHT_SHARD_BEATS;
    const Beat512 *state_in29 = w29 + GDN_COMPILED_WEIGHT_SHARD_BEATS;
    const Beat512 *state_in30 = w30 + GDN_COMPILED_WEIGHT_SHARD_BEATS;
    const Beat512 *state_in31 = w31 + GDN_COMPILED_WEIGHT_SHARD_BEATS;
    Beat512 *state_out28 = w28 + GDN_COMPILED_WEIGHT_SHARD_BEATS;
    Beat512 *state_out29 = w29 + GDN_COMPILED_WEIGHT_SHARD_BEATS;
    Beat512 *state_out30 = w30 + GDN_COMPILED_WEIGHT_SHARD_BEATS;
    Beat512 *state_out31 = w31 + GDN_COMPILED_WEIGHT_SHARD_BEATS;

    /* BF16 activations: one 32-lane beat per weight beat, so the activation
     * stream is in_dim/32 beats -- half what the FP32 pair needed. */
    uint32_t k_packs      = in_dim / 32;
    uint32_t rows_per_ch  = out_dim / GEMV_CHANNELS;
    uint32_t opacks_per_ch = (rows_per_ch + 15) >> 4;
    uint32_t n_packs      = rows_per_ch * k_packs;
    uint32_t total_opacks = opacks_per_ch * GEMV_CHANNELS;

    hls::stream<Beat512> ws[GEMV_CHANNELS];
    hls::stream<Beat512> xr[GEMV_CLUSTERS + 1];
    hls::stream<Beat512> ys[GEMV_CLUSTERS];
    hls::stream<Beat512> slr0_result, slr1_result, slr2_result;
    hls::stream<Beat512> slr0_boundary, slr1_boundary, slr2_boundary, result;
    hls::stream<Beat512> q_stream, k_stream, v_stream;
    hls::stream<Beat512> state_stream0, state_stream1;
    hls::stream<Beat512> state_stream2, state_stream3;
    #pragma HLS array_partition variable=ws complete
    #pragma HLS array_partition variable=xr complete
    #pragma HLS array_partition variable=ys complete
    /* iter16: these 69 512-bit FIFOs were impl=lutram at depth 16/4, costing
     * 60,063 LUT + 111,729 FF of CLB resources -- and CLB in SLR0 is the ONLY
     * congested resource (SLR0 99.7%, and every congested window in iter15's
     * router log lies inside SLR0). BRAM is at 35% (637 of 1,816), so this
     * trades the scarce resource for the abundant one.
     *
     * The routed 32-port microbenchmark (microbench/gemv_tile) does exactly
     * this -- depth 64, impl=bram -- and is the only 32-port GEMV on this
     * device that routes with zero errors; it carries 96.88% BRAM / 90.47% CLB
     * in SLR1. Ours is the same engine with an aux path bolted on.
     *
     * Depth 64 (not 16/4) because BRAM cost is width-dominated: a 512-bit FIFO
     * needs ~8 RAMB36 for width whatever its depth, so depth is nearly free
     * here. That is the opposite of LUTRAM, where depth-4 and depth-16 measured
     * 1,547 vs 1,552 FF -- 0.3% apart. Expect ~550 extra BRAM tiles. */
    #pragma HLS stream variable=ws depth=64
    #pragma HLS stream variable=xr depth=64
    #pragma HLS stream variable=ys depth=64
    #pragma HLS stream variable=slr0_result depth=64
    #pragma HLS stream variable=slr1_result depth=64
    #pragma HLS stream variable=slr2_result depth=64
    /* One complete local collector burst must fit so the final collector's
     * fixed 4/6/6 drain order cannot backpressure another branch. HLS measured
     * a 14-entry requirement on the first branch; round all three tiny LUTRAM
     * queues to the natural 16-word burst boundary. */
    #pragma HLS stream variable=slr0_boundary depth=16
    #pragma HLS stream variable=slr1_boundary depth=16
    #pragma HLS stream variable=slr2_boundary depth=16
    #pragma HLS stream variable=result depth=64
    #pragma HLS stream variable=q_stream depth=32
    #pragma HLS stream variable=k_stream depth=32
    #pragma HLS stream variable=v_stream depth=32
    /* Packed weights shorten each head's weight phase from 4,096 to 2,048
     * Beats and BF16 state consumes 512 Beats/head.
     * A two-head queue can therefore fill and block the state-owning MM2S
     * before it delivers the later weights needed to create the Q/K/V that
     * would drain the queue.  Hold the complete eight-head state window so
     * state backpressure can never starve a weight stream.  URAM is deliberate:
     * four 512 x 4,096 FIFOs fit comfortably in the unused UltraRAM budget,
     * whereas the equivalent BRAM growth exceeds the routed design's margin. */
    #pragma HLS stream variable=state_stream0 depth=4096
    #pragma HLS stream variable=state_stream1 depth=4096
    #pragma HLS stream variable=state_stream2 depth=4096
    #pragma HLS stream variable=state_stream3 depth=4096
    #pragma HLS bind_storage variable=ws type=fifo impl=bram
    #pragma HLS bind_storage variable=xr type=fifo impl=bram
    #pragma HLS bind_storage variable=ys type=fifo impl=bram
    #pragma HLS bind_storage variable=slr0_result type=fifo impl=bram
    #pragma HLS bind_storage variable=slr1_result type=fifo impl=bram
    #pragma HLS bind_storage variable=slr2_result type=fifo impl=bram
    #pragma HLS bind_storage variable=slr0_boundary type=fifo impl=lutram
    #pragma HLS bind_storage variable=slr1_boundary type=fifo impl=lutram
    #pragma HLS bind_storage variable=slr2_boundary type=fifo impl=lutram
    #pragma HLS bind_storage variable=result type=fifo impl=bram
    #pragma HLS bind_storage variable=q_stream type=fifo impl=bram
    #pragma HLS bind_storage variable=k_stream type=fifo impl=bram
    #pragma HLS bind_storage variable=v_stream type=fifo impl=bram
    #pragma HLS bind_storage variable=state_stream0 type=fifo impl=uram
    #pragma HLS bind_storage variable=state_stream1 type=fifo impl=uram
    #pragma HLS bind_storage variable=state_stream2 type=fifo impl=uram
    #pragma HLS bind_storage variable=state_stream3 type=fifo impl=uram

    #pragma HLS dataflow disable_start_propagation
    gemv32_load_x_and_w0(in, sh0, shard_off, xr[0], ws[0],
                         k_packs, n_packs);
    gemv32_mm2s<1>(sh1, shard_off, ws[1], n_packs);
    gemv32_mm2s<2>(sh2, shard_off, ws[2], n_packs);
    gemv32_mm2s<3>(sh3, shard_off, ws[3], n_packs);
    gemv32_mm2s<4>(sh4, shard_off, ws[4], n_packs);
    gemv32_mm2s<5>(sh5, shard_off, ws[5], n_packs);
    gemv32_mm2s<6>(sh6, shard_off, ws[6], n_packs);
    gemv32_mm2s<7>(sh7, shard_off, ws[7], n_packs);
    gemv32_mm2s<8>(sh8, shard_off, ws[8], n_packs);
    gemv32_mm2s<9>(sh9, shard_off, ws[9], n_packs);
    gemv32_mm2s<10>(sh10, shard_off, ws[10], n_packs);
    gemv32_mm2s<11>(sh11, shard_off, ws[11], n_packs);
    gemv32_mm2s<12>(sh12, shard_off, ws[12], n_packs);
    gemv32_mm2s<13>(sh13, shard_off, ws[13], n_packs);
    gemv32_mm2s<14>(sh14, shard_off, ws[14], n_packs);
    gemv32_mm2s<15>(sh15, shard_off, ws[15], n_packs);
    gemv32_mm2s<16>(sh16, shard_off, ws[16], n_packs);
    gemv32_mm2s<17>(sh17, shard_off, ws[17], n_packs);
    gemv32_mm2s<18>(sh18, shard_off, ws[18], n_packs);
    gemv32_mm2s<19>(sh19, shard_off, ws[19], n_packs);
    gemv32_mm2s<20>(sh20, shard_off, ws[20], n_packs);
    gemv32_mm2s<21>(sh21, shard_off, ws[21], n_packs);
    gemv32_mm2s<22>(sh22, shard_off, ws[22], n_packs);
    gemv32_mm2s<23>(sh23, shard_off, ws[23], n_packs);
    gemv32_mm2s<24>(sh24, shard_off, ws[24], n_packs);
    gemv32_mm2s<25>(sh25, shard_off, ws[25], n_packs);
    gemv32_mm2s<26>(sh26, shard_off, ws[26], n_packs);
    gemv32_mm2s<27>(sh27, shard_off, ws[27], n_packs);
    gemv32_mm2s_with_state<28>(
        sh28, state_in28, shard_off, ws[28], state_stream0,
        n_packs, layer_index, qkvg_recurrent_mode);
    gemv32_mm2s_with_state<29>(
        sh29, state_in29, shard_off, ws[29], state_stream1,
        n_packs, layer_index, qkvg_recurrent_mode);
    gemv32_mm2s_with_state<30>(
        sh30, state_in30, shard_off, ws[30], state_stream2,
        n_packs, layer_index, qkvg_recurrent_mode);
    gemv32_mm2s_with_state<31>(
        sh31, state_in31, shard_off, ws[31], state_stream3,
        n_packs, layer_index, qkvg_recurrent_mode);

    gemv32_cluster2(ws[0],  ws[1],  xr[0],  xr[1],  ys[0],  k_packs, rows_per_ch);
    gemv32_cluster2(ws[2],  ws[3],  xr[1],  xr[2],  ys[1],  k_packs, rows_per_ch);
    gemv32_cluster2(ws[4],  ws[5],  xr[2],  xr[3],  ys[2],  k_packs, rows_per_ch);
    gemv32_cluster2(ws[6],  ws[7],  xr[3],  xr[4],  ys[3],  k_packs, rows_per_ch);
    gemv32_cluster2(ws[8],  ws[9],  xr[4],  xr[5],  ys[4],  k_packs, rows_per_ch);
    gemv32_cluster2(ws[10], ws[11], xr[5],  xr[6],  ys[5],  k_packs, rows_per_ch);
    gemv32_cluster2(ws[12], ws[13], xr[6],  xr[7],  ys[6],  k_packs, rows_per_ch);
    gemv32_cluster2(ws[14], ws[15], xr[7],  xr[8],  ys[7],  k_packs, rows_per_ch);
    gemv32_cluster2(ws[16], ws[17], xr[8],  xr[9],  ys[8],  k_packs, rows_per_ch);
    gemv32_cluster2(ws[18], ws[19], xr[9],  xr[10], ys[9],  k_packs, rows_per_ch);
    gemv32_cluster2(ws[20], ws[21], xr[10], xr[11], ys[10], k_packs, rows_per_ch);
    gemv32_cluster2(ws[22], ws[23], xr[11], xr[12], ys[11], k_packs, rows_per_ch);
    gemv32_cluster2(ws[24], ws[25], xr[12], xr[13], ys[12], k_packs, rows_per_ch);
    gemv32_cluster2(ws[26], ws[27], xr[13], xr[14], ys[13], k_packs, rows_per_ch);
    gemv32_cluster2(ws[28], ws[29], xr[14], xr[15], ys[14], k_packs, rows_per_ch);
    gemv32_cluster2(ws[30], ws[31], xr[15], xr[16], ys[15], k_packs, rows_per_ch);
    gemv32_drain_x(xr[16], k_packs);
    /* Restore the routed 4/6/6 collector cut. Only the three small relay
     * actors are physically constrained; clusters, local collectors and
     * almost all FIFO endpoints remain free for SSI spreading. */
    gemv32_collect4(ys[0], ys[1], ys[2], ys[3],
                    slr0_result, opacks_per_ch);
    gemv32_collect6(ys[4], ys[5], ys[6], ys[7], ys[8], ys[9],
                    slr1_result, opacks_per_ch);
    gemv32_collect6(ys[10], ys[11], ys[12], ys[13], ys[14], ys[15],
                    slr2_result, opacks_per_ch);
    gemv32_boundary_relay<0>(slr0_result, slr0_boundary,
                             8 * opacks_per_ch);
    gemv32_boundary_relay<1>(slr1_result, slr1_boundary,
                             12 * opacks_per_ch);
    gemv32_boundary_relay<2>(slr2_result, slr2_boundary,
                             12 * opacks_per_ch);
    gemv32_collect_final(slr0_boundary, slr1_boundary, slr2_boundary,
                         result, opacks_per_ch);
    gemv32_store_or_qkvg_conv_stream(
        result, q_stream, k_stream, v_stream,
        out, logits_stream, rows_per_ch, opacks_per_ch, total_opacks,
        qkvg_recurrent_mode, gate_out,
        conv_weights, conv_tails);
    gdn_recurrent_attention_islands(
        q_stream, k_stream, v_stream,
        state_stream0, state_stream1, state_stream2, state_stream3,
        attn_out,
        state_out28, state_out29, state_out30, state_out31,
        a, b, layer_a_log, layer_dt_bias,
        layer_index, qkvg_recurrent_mode);
}
