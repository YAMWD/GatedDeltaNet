#include "gdn_model.h"

#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

union TestFloatBits {
    float value;
    uint32_t bits;
};

static float bits_to_float(uint32_t bits) {
    TestFloatBits converted;
    converted.bits = bits;
    return converted.value;
}

static uint32_t float_to_bits(float value) {
    TestFloatBits converted;
    converted.value = value;
    return converted.bits;
}

static uint32_t mixed_reference(uint16_t weight, uint32_t activation) {
    const uint32_t weight_exponent = (weight >> 7) & 0xffu;
    const uint32_t weight_fraction = weight & 0x7fu;
    const uint32_t activation_exponent = (activation >> 23) & 0xffu;
    const uint32_t activation_fraction = activation & 0x7fffffu;
    const uint32_t sign = ((weight >> 15) ^ (activation >> 31)) << 31;
    const bool weight_nan = weight_exponent == 0xffu && weight_fraction != 0;
    const bool activation_nan = activation_exponent == 0xffu &&
                                activation_fraction != 0;
    const bool weight_inf = weight_exponent == 0xffu && weight_fraction == 0;
    const bool activation_inf = activation_exponent == 0xffu &&
                                activation_fraction == 0;
    const bool weight_zero_or_subnormal = weight_exponent == 0;
    const bool activation_zero_or_subnormal = activation_exponent == 0;

    if (weight_nan || activation_nan ||
        ((weight_inf || activation_inf) &&
         (weight_zero_or_subnormal || activation_zero_or_subnormal)))
        return 0x7fc00000u;
    if (weight_inf || activation_inf)
        return sign | 0x7f800000u;
    if (weight_zero_or_subnormal || activation_zero_or_subnormal)
        return sign;

    volatile float product = bits_to_float((uint32_t)weight << 16) *
                             bits_to_float(activation);
    uint32_t result = float_to_bits(product);
    const uint32_t result_exponent = (result >> 23) & 0xffu;
    const uint32_t result_fraction = result & 0x7fffffu;
    if (result_exponent == 0)
        return sign; /* production FTZ */
    if (result_exponent == 0xffu && result_fraction != 0)
        return 0x7fc00000u;
    return result;
}

static uint16_t bf16_rne_reference(uint32_t bits) {
    const uint32_t magnitude = bits & 0x7fffffffu;
    if (magnitude >= 0x7f800000u) {
        if (magnitude == 0x7f800000u)
            return (uint16_t)(bits >> 16);
        return (uint16_t)((bits & 0x80000000u) ? 0xffc0u : 0x7fc0u);
    }
    return (uint16_t)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

static uint32_t xorshift32(uint32_t &state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

static void fail(const char *test, uint64_t index,
                 uint32_t expected, uint32_t actual) {
    std::fprintf(stderr,
                 "%s failed at %llu: expected=0x%08x actual=0x%08x\n",
                 test, (unsigned long long)index, expected, actual);
    std::exit(1);
}

/* ---------------------------------------------------------------------------
 * Retired multipliers under test, moved here from gdn_model.cpp when the
 * production kernel switched to gdn_native_bf16_mul_to_fp32 (Iter66). This
 * suite still pins their measured semantics; nothing in the kernel calls them.
 * ------------------------------------------------------------------------- */

/* Iter63d fulldsp mixed multiplier: DAZ inputs, FTZ outputs, exact
 * BF16 x FP32 product otherwise. */
static float gdn_mixed_mul_bf16_fp32(Bf16Bits weight, float activation) {
    const uint16_t weight_bits = (uint16_t)weight;
    const uint32_t activation_bits = float_to_bits(activation);
    const uint32_t weight_exponent = (weight_bits >> 7) & 0xffu;
    const uint32_t weight_fraction = weight_bits & 0x7fu;
    const uint32_t activation_exponent = (activation_bits >> 23) & 0xffu;
    const uint32_t activation_fraction = activation_bits & 0x7fffffu;
    const uint32_t sign =
        ((uint32_t)((weight_bits >> 15) ^ (activation_bits >> 31))) << 31;
    const bool weight_nan = weight_exponent == 0xffu && weight_fraction != 0;
    const bool activation_nan = activation_exponent == 0xffu &&
                                activation_fraction != 0;
    const bool weight_inf = weight_exponent == 0xffu && weight_fraction == 0;
    const bool activation_inf = activation_exponent == 0xffu &&
                                activation_fraction == 0;
    const bool weight_zero_or_subnormal = weight_exponent == 0;
    const bool activation_zero_or_subnormal = activation_exponent == 0;
    if (weight_nan || activation_nan || weight_inf || activation_inf ||
        weight_zero_or_subnormal || activation_zero_or_subnormal) {
        if (weight_nan || activation_nan ||
            ((weight_inf || activation_inf) &&
             (weight_zero_or_subnormal || activation_zero_or_subnormal)))
            return bits_to_float(0x7fc00000u);
        if (weight_inf || activation_inf)
            return bits_to_float(sign | 0x7f800000u);
        return bits_to_float(sign);
    }
    volatile float host_product =
        bits_to_float((uint32_t)weight_bits << 16) * activation;
    uint32_t result = float_to_bits(host_product);
    if (((result >> 23) & 0xffu) == 0)                  /* FTZ */
        result = sign;
    return bits_to_float(result);
}

/* Iter63 packed 24x8 pair multiplier (REJECTED for production; the packing
 * claim is still exhaustively proven below). */
static float gdn_bf16_finish(uint32_t wbits, uint32_t xbits, uint32_t p) {
    const uint32_t we = (wbits >> 7) & 0xffu;
    const uint32_t xe = (xbits >> 7) & 0xffu;
    const bool w_nan  = (we == 255u) && ((wbits & 0x7fu) != 0u);
    const bool x_nan  = (xe == 255u) && ((xbits & 0x7fu) != 0u);
    const bool w_inf  = (we == 255u) && ((wbits & 0x7fu) == 0u);
    const bool x_inf  = (xe == 255u) && ((xbits & 0x7fu) == 0u);
    const bool w_zero = (we == 0u);          /* DAZ: subnormal reads as zero */
    const bool x_zero = (xe == 0u);
    const uint32_t sign = ((wbits ^ xbits) & 0x8000u) << 16;

    if (w_nan || x_nan || (w_inf && x_zero) || (x_inf && w_zero))
        return bits_to_float(0x7fc00000u);              /* canonical qNaN */
    if (w_inf || x_inf)
        return bits_to_float(sign | 0x7f800000u);
    if (w_zero || x_zero)
        return bits_to_float(sign);                     /* signed zero */

    /* Both significands are in [128,255], so p is in [2^14, 2^16). */
    const uint32_t norm = (p >> 15) & 1u;
    const int32_t  e    = (int32_t)we + (int32_t)xe - 127 + (int32_t)norm;
    if (e <= 0)                                         /* FTZ */
        return bits_to_float(sign);
    if (e >= 255)
        return bits_to_float(sign | 0x7f800000u);
    const uint32_t frac = norm ? ((p & 0x7fffu) << 8) : ((p & 0x3fffu) << 9);
    return bits_to_float(sign | ((uint32_t)e << 23) | frac);
}

static void gdn_test_bf16_mul_pair(uint16_t w0, uint16_t w1, uint16_t x,
                                   float *r0, float *r1) {
    const uint32_t w0b = w0;
    const uint32_t w1b = w1;
    const uint32_t xb  = x;
    const uint32_t w0s = (((w0b >> 7) & 0xffu) == 0u) ? 0u : (0x80u | (w0b & 0x7fu));
    const uint32_t w1s = (((w1b >> 7) & 0xffu) == 0u) ? 0u : (0x80u | (w1b & 0x7fu));
    const uint32_t xs  = (((xb  >> 7) & 0xffu) == 0u) ? 0u : (0x80u | (xb  & 0x7fu));
    /* (w0s | w1s<<16) is 24 bits, xs is 8: the packed product fits uint32. */
    const uint32_t packed_p = (w0s | (w1s << 16)) * xs;
    *r0 = gdn_bf16_finish(w0b, xb, packed_p & 0xffffu);
    *r1 = gdn_bf16_finish(w1b, xb, packed_p >> 16);
}

int main() {
    if (std::fesetround(FE_TONEAREST) != 0) {
        std::fprintf(stderr, "cannot select round-to-nearest-even\n");
        return 1;
    }
    if (sizeof(Beat512) != 64)
        fail("Beat512 size", 0, 64, sizeof(Beat512));

    uint32_t random = 0x62b16f32u;
    for (uint32_t trial = 0; trial < 100000; ++trial) {
        Beat512 fp32_beat = 0;
        uint32_t expected_fp32[16];
        for (uint32_t lane = 0; lane < 16; ++lane) {
            expected_fp32[lane] = xorshift32(random);
            gdn_test_set_fp32_lane_bits(
                &fp32_beat, lane, expected_fp32[lane]);
        }
        for (uint32_t lane = 0; lane < 16; ++lane) {
            const uint32_t actual =
                gdn_test_get_fp32_lane_bits(&fp32_beat, lane);
            if (actual != expected_fp32[lane])
                fail("FP32 Beat512 round trip",
                     (uint64_t)trial * 16 + lane,
                     expected_fp32[lane], actual);
        }

        Beat512 bf16_beat = 0;
        uint16_t expected_bf16[32];
        for (uint32_t lane = 0; lane < 32; ++lane) {
            expected_bf16[lane] = (uint16_t)xorshift32(random);
            gdn_test_set_bf16_lane_bits(
                &bf16_beat, lane, expected_bf16[lane]);
        }
        for (uint32_t lane = 0; lane < 32; ++lane) {
            const uint16_t actual =
                gdn_test_get_bf16_lane_bits(&bf16_beat, lane);
            if (actual != expected_bf16[lane])
                fail("BF16 Beat512 round trip",
                     (uint64_t)trial * 32 + lane,
                     expected_bf16[lane], actual);
        }
    }

    const uint32_t rne_cases[] = {
        0x00000000u, 0x80000000u, 0x3f800000u, 0xbf800000u,
        0x3f808000u, 0x3f818000u, 0xbf808000u, 0xbf818000u,
        0x3fffffffU, 0x7f7fffffu, 0xff7fffffu, 0x00800000u,
        0x00000001u, 0x007fffffu, 0x7f800000u, 0xff800000u,
        0x7f800001u, 0xff800001u, 0x7fc12345u, 0xffc12345u,
    };
    for (uint32_t i = 0; i < sizeof(rne_cases) / sizeof(rne_cases[0]); ++i) {
        const uint16_t expected = bf16_rne_reference(rne_cases[i]);
        const uint16_t actual =
            gdn_test_fp32_to_bf16_rne_bits(rne_cases[i]);
        if (actual != expected)
            fail("BF16 RNE boundary", i, expected, actual);
    }
    for (uint32_t i = 0; i < 1000000; ++i) {
        const uint32_t bits = xorshift32(random);
        const uint16_t expected = bf16_rne_reference(bits);
        const uint16_t actual = gdn_test_fp32_to_bf16_rne_bits(bits);
        if (actual != expected)
            fail("BF16 RNE random", i, expected, actual);
    }
    for (uint32_t bits = 0; bits <= 0xffffu; ++bits) {
        const uint32_t expected = bits << 16;
        const uint32_t actual =
            gdn_test_bf16_to_fp32_bits((uint16_t)bits);
        if (actual != expected)
            fail("BF16 widen", bits, expected, actual);
    }

    const uint32_t strategic_activation[] = {
        0x00000000u, 0x80000000u, 0x00000001u, 0x007fffffu,
        0x00800000u, 0x00800001u, 0x3effffffu, 0x3f000000u,
        0x3f7fffffu, 0x3f800000u, 0x3f800001u, 0x3fffffffu,
        0x40000000u, 0x7f7fffffu, 0xff7fffffu, 0x7f800000u,
        0xff800000u, 0x7f800001u, 0x7fc00000u, 0xffc12345u,
    };
    for (uint32_t weight = 0; weight <= 0xffffu; ++weight) {
        for (uint32_t value = 0;
             value < sizeof(strategic_activation) /
                     sizeof(strategic_activation[0]); ++value) {
            const uint32_t activation = strategic_activation[value];
            const uint32_t expected =
                mixed_reference((uint16_t)weight, activation);
            const uint32_t actual = float_to_bits(
                gdn_mixed_mul_bf16_fp32(
                    Bf16Bits((uint16_t)weight), bits_to_float(activation)));
            if (actual != expected)
                fail("mixed exhaustive BF16", (uint64_t)weight * 20 + value,
                     expected, actual);
        }
    }

    for (uint32_t i = 0; i < 1000000; ++i) {
        uint16_t weight = (uint16_t)xorshift32(random);
        uint32_t activation = xorshift32(random);
        weight = (uint16_t)((weight & 0x807fu) |
                 ((1u + (xorshift32(random) % 254u)) << 7));
        activation = (activation & 0x807fffffu) |
                     ((1u + (xorshift32(random) % 254u)) << 23);
        const uint32_t expected = mixed_reference(weight, activation);
        const uint32_t actual = float_to_bits(
            gdn_mixed_mul_bf16_fp32(
                Bf16Bits(weight), bits_to_float(activation)));
        if (actual != expected)
            fail("mixed randomized finite", i, expected, actual);
    }

    std::printf(
        "PASS: raw Beat512 lanes, BF16 RNE/widen, 65,536-pattern mixed "
        "boundaries, and 1,000,000 randomized finite products\n");

    /* ------------------------------------------------------------------
     * Iter63: the paired BF16xBF16 multiplier.
     *
     * Exhaustive over all 128^3 significand triples -- 2,097,152 cases, which
     * is the complete space for the packing claim, since the exponent path is
     * a separate adder and is swept below. Each product must equal the exact
     * FP32 product of the two BF16 operands (an 8x8 significand product is 16
     * bits and fits FP32's 24, so no rounding is involved), and neither half of
     * the packed 32-bit result may disturb the other.
     * ------------------------------------------------------------------ */
    {
        uint64_t checked = 0;
        for (uint32_t fw0 = 0; fw0 < 128; ++fw0) {
            for (uint32_t fw1 = 0; fw1 < 128; ++fw1) {
                for (uint32_t fx = 0; fx < 128; ++fx) {
                    /* exponent 127 == 1.0 scale keeps the product in range;
                     * exponents are swept separately below */
                    const uint16_t w0 = (uint16_t)((127u << 7) | fw0);
                    const uint16_t w1 = (uint16_t)((127u << 7) | fw1);
                    const uint16_t x  = (uint16_t)((127u << 7) | fx);
                    float r0 = 0.0f, r1 = 0.0f;
                    gdn_test_bf16_mul_pair(w0, w1, x, &r0, &r1);
                    const float e0 = bits_to_float(gdn_test_bf16_to_fp32_bits(w0)) *
                                     bits_to_float(gdn_test_bf16_to_fp32_bits(x));
                    const float e1 = bits_to_float(gdn_test_bf16_to_fp32_bits(w1)) *
                                     bits_to_float(gdn_test_bf16_to_fp32_bits(x));
                    if (std::memcmp(&r0, &e0, sizeof(float)) != 0)
                        fail("pair mul low significand", checked,
                             float_to_bits(e0), float_to_bits(r0));
                    if (std::memcmp(&r1, &e1, sizeof(float)) != 0)
                        fail("pair mul high significand", checked,
                             float_to_bits(e1), float_to_bits(r1));
                    ++checked;
                }
            }
        }
        std::printf("pair multiplier: %llu exhaustive significand triples OK\n",
                    (unsigned long long)checked);
    }

    /* Exponent and sign sweep, both ports, including the boundaries where the
     * result overflows to infinity or flushes to zero. */
    {
        uint64_t checked = 0, flushed = 0, overflowed = 0;
        for (uint32_t ew = 1; ew < 255; ++ew) {
            for (uint32_t ex = 1; ex < 255; ++ex) {
                for (uint32_t signs = 0; signs < 4; ++signs) {
                    const uint16_t w0 = (uint16_t)(((signs & 1u) << 15) | (ew << 7) | 0x2a);
                    const uint16_t w1 = (uint16_t)(((signs & 1u) << 15) | (ew << 7) | 0x55);
                    const uint16_t x  = (uint16_t)(((signs >> 1) << 15) | (ex << 7) | 0x13);
                    float r0 = 0.0f, r1 = 0.0f;
                    gdn_test_bf16_mul_pair(w0, w1, x, &r0, &r1);
                    const float e0 = bits_to_float(gdn_test_bf16_to_fp32_bits(w0)) *
                                     bits_to_float(gdn_test_bf16_to_fp32_bits(x));
                    const float e1 = bits_to_float(gdn_test_bf16_to_fp32_bits(w1)) *
                                     bits_to_float(gdn_test_bf16_to_fp32_bits(x));
                    /* the kernel flushes subnormal results to zero (FTZ), so
                     * compare against the flushed reference */
                    const float f0 = (std::fpclassify(e0) == FP_SUBNORMAL)
                                   ? std::copysign(0.0f, e0) : e0;
                    const float f1 = (std::fpclassify(e1) == FP_SUBNORMAL)
                                   ? std::copysign(0.0f, e1) : e1;
                    if (f0 != e0 || f1 != e1) ++flushed;
                    if (std::isinf(f0) || std::isinf(f1)) ++overflowed;
                    if (std::memcmp(&r0, &f0, sizeof(float)) != 0)
                        fail("pair mul exponent low", checked,
                             float_to_bits(f0), float_to_bits(r0));
                    if (std::memcmp(&r1, &f1, sizeof(float)) != 0)
                        fail("pair mul exponent high", checked,
                             float_to_bits(f1), float_to_bits(r1));
                    ++checked;
                }
            }
        }
        std::printf("pair multiplier: %llu exponent/sign cases OK "
                    "(%llu flushed, %llu overflowed)\n",
                    (unsigned long long)checked,
                    (unsigned long long)flushed,
                    (unsigned long long)overflowed);
    }

    /* Special classes, and the cross-contamination case that the packed
     * multiply could plausibly get wrong: one port special, the other normal. */
    {
        const uint16_t specials[] = {
            0x0000, 0x8000,             /* +/- zero */
            0x0001, 0x8001,             /* subnormal -> DAZ */
            0x7f80, 0xff80,             /* +/- infinity */
            0x7fc0, 0xffc0, 0x7fa5,     /* NaNs */
            0x3f80, 0xbf80,             /* +/- 1.0 */
        };
        const uint32_t ns = sizeof(specials) / sizeof(specials[0]);
        uint64_t checked = 0;
        for (uint32_t a = 0; a < ns; ++a) {
            for (uint32_t b = 0; b < ns; ++b) {
                for (uint32_t c = 0; c < ns; ++c) {
                    float r0 = 0.0f, r1 = 0.0f;
                    gdn_test_bf16_mul_pair(specials[a], specials[b],
                                           specials[c], &r0, &r1);
                    /* each port must match a single-port evaluation of itself */
                    float s0a = 0.0f, s0b = 0.0f, s1a = 0.0f, s1b = 0.0f;
                    gdn_test_bf16_mul_pair(specials[a], specials[a],
                                           specials[c], &s0a, &s0b);
                    gdn_test_bf16_mul_pair(specials[b], specials[b],
                                           specials[c], &s1a, &s1b);
                    if (std::memcmp(&r0, &s0a, sizeof(float)) != 0)
                        fail("pair mul contamination low", checked,
                             float_to_bits(s0a), float_to_bits(r0));
                    if (std::memcmp(&r1, &s1a, sizeof(float)) != 0)
                        fail("pair mul contamination high", checked,
                             float_to_bits(s1a), float_to_bits(r1));
                    ++checked;
                }
            }
        }
        std::printf("pair multiplier: %llu special-class triples OK "
                    "(no cross-port contamination)\n",
                    (unsigned long long)checked);
    }

    return 0;
}
