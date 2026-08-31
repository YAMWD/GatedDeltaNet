#include "gdn_model.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

union LayoutFloatBits {
    float value;
    uint32_t bits;
};

static float bf16_value(uint16_t bits) {
    LayoutFloatBits value;
    value.bits = (uint32_t)bits << 16;
    return value.value;
}

static void fail(const char *message, size_t index,
                 uint32_t expected, uint32_t actual) {
    std::fprintf(stderr,
                 "%s at %zu: expected=0x%08x actual=0x%08x\n",
                 message, index, expected, actual);
    std::exit(1);
}

int main() {
    const size_t rows = 3;
    std::vector<float> recurrent(rows * 256u);
    std::vector<Beat512> stripes[GDN_RECURRENT_STATE_PORTS];
    Beat512 *stripe_ptrs[GDN_RECURRENT_STATE_PORTS];
    for (uint32_t port = 0; port < GDN_RECURRENT_STATE_PORTS; ++port) {
        stripes[port].resize(rows * 2u);
        stripe_ptrs[port] = stripes[port].data();
    }
    for (size_t i = 0; i < recurrent.size(); ++i)
        recurrent[i] = bf16_value((uint16_t)(0x3f00u | (i & 0x7fu)));

    if (gdn_scatter_recurrent_state(stripe_ptrs, recurrent.data(),
                                    recurrent.size()) != 0)
        fail("state scatter rejected BF16-exact input", 0, 0, 1);

    for (size_t row = 0; row < rows; ++row) {
        for (uint32_t port = 0; port < GDN_RECURRENT_STATE_PORTS; ++port) {
            const uint32_t island = port & 1u;
            const uint32_t high_half = port >> 1;
            for (uint32_t pair = 0; pair < 2; ++pair) {
                const Beat512 &beat = stripes[port][row * 2u + pair];
                for (uint32_t subhalf = 0; subhalf < 2; ++subhalf) {
                    for (uint32_t lane = 0; lane < 16; ++lane) {
                        const uint32_t global_v = high_half * 128u
                                                + pair * 64u
                                                + subhalf * 32u
                                                + island * 16u + lane;
                        const size_t source = row * 256u + global_v;
                        const uint16_t expected =
                            (uint16_t)(0x3f00u | (source & 0x7fu));
                        const uint16_t actual = gdn_test_get_bf16_lane_bits(
                            &beat, subhalf * 16u + lane);
                        if (actual != expected)
                            fail("state scatter mapping", source,
                                 expected, actual);
                    }
                }
            }
        }
    }

    std::vector<float> conv(GDN_WSF_HEADBUF);
    std::vector<Beat512> packed_conv(GDN_WSF_HEADBUF / 16u);
    for (size_t i = 0; i < conv.size(); ++i)
        conv[i] = bf16_value((uint16_t)(0x3e80u | (i & 0x7fu)));
    if (gdn_pack_conv_tails_bf16(packed_conv.data(), conv.data(),
                                 conv.size()) != 0)
        fail("conv pack rejected BF16-exact input", 0, 0, 1);

    for (size_t stripe = 0; stripe < GDN_CONV_TAIL_STRIPES; ++stripe) {
        const size_t source_base = stripe * GDN_CONV_TAIL_FLOATS_PER_STRIPE;
        const size_t destination_base =
            stripe * GDN_CONV_TAIL_RESERVED_BEATS_PER_STRIPE;
        for (uint32_t beat = 0;
             beat < GDN_CONV_TAIL_BF16_BEATS_PER_STRIPE; ++beat) {
            for (uint32_t lane = 0; lane < 32; ++lane) {
                const size_t source = source_base + beat * 32u + lane;
                const uint16_t expected =
                    (uint16_t)(0x3e80u | (source & 0x7fu));
                const uint16_t actual = gdn_test_get_bf16_lane_bits(
                    &packed_conv[destination_base + beat], lane);
                if (actual != expected)
                    fail("conv tail mapping", source, expected, actual);
            }
        }
        for (uint32_t beat = GDN_CONV_TAIL_BF16_BEATS_PER_STRIPE;
             beat < GDN_CONV_TAIL_RESERVED_BEATS_PER_STRIPE; ++beat) {
            for (uint32_t lane = 0; lane < 32; ++lane) {
                const uint16_t actual = gdn_test_get_bf16_lane_bits(
                    &packed_conv[destination_base + beat], lane);
                if (actual != 0)
                    fail("conv reserved padding", destination_base + beat,
                         0, actual);
            }
        }
    }

    std::puts("PASS: BF16 recurrent scatter and reserved-ABI convolution-tail packing");
    return 0;
}
