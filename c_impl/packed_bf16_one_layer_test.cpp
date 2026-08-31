#include "gdn_model.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

namespace {

const size_t kAuxFloats = 63760;
const size_t kWorkspaceBeats = 37650;
const size_t kWeightBeats = 118272;
const size_t kStateBeatsPerPort = 4096;

static_assert(GDN_COMPILED_WEIGHT_SHARD_BEATS == kWeightBeats,
              "generated one-layer shard size drift");
static_assert(GDN_WSF_STATE / GDN_RECURRENT_STATE_PORTS / 32 ==
                  kStateBeatsPerPort,
              "generated one-layer state stripe size drift");
static_assert(GDN_WS_FLOATS / 16 == kWorkspaceBeats,
              "generated one-layer workspace size drift");

template <typename T>
T *allocate_zeroed(size_t count) {
    T *values = new T[count];
    for (size_t index = 0; index < count; ++index)
        values[index] = 0;
    return values;
}

Beat512 bf16_pattern(uint16_t value) {
    Beat512 beat = 0;
    for (uint32_t lane = 0; lane < 32; ++lane)
        beat.range(16 * lane + 15, 16 * lane) = value + (lane & 1u);
    return beat;
}

Beat512 fp32_pattern(uint32_t value) {
    Beat512 beat = 0;
    for (uint32_t lane = 0; lane < 16; ++lane)
        beat.range(32 * lane + 31, 32 * lane) = value;
    return beat;
}

uint64_t checksum_beats(const Beat512 *values, size_t count) {
    uint64_t checksum = 0xcbf29ce484222325ULL;
    for (size_t beat = 0; beat < count; ++beat) {
        for (uint32_t lane = 0; lane < 16; ++lane) {
            const uint32_t bits =
                (uint32_t)values[beat].range(32 * lane + 31, 32 * lane);
            checksum ^= bits;
            checksum *= 0x100000001b3ULL;
        }
    }
    return checksum;
}

}  // namespace

int main() {
    float *aux_weights = allocate_zeroed<float>(kAuxFloats);
    Beat512 *workspace = allocate_zeroed<Beat512>(kWorkspaceBeats);
    Beat512 *weights[GEMV_CHANNELS];

    for (size_t index = 0; index < kAuxFloats; ++index)
        aux_weights[index] = 0.00390625f;
    for (size_t beat = 0; beat < 2048 / 16; ++beat)
        workspace[GDN_WS_OFF_X / 16 + beat] = fp32_pattern(0x3e000000u);

    for (int channel = 0; channel < GEMV_CHANNELS; ++channel) {
        const size_t beats = kWeightBeats +
            (channel >= GDN_RECURRENT_STATE_FIRST_PORT
                ? kStateBeatsPerPort : 0);
        weights[channel] = allocate_zeroed<Beat512>(beats);
        const Beat512 weight_value =
            bf16_pattern((uint16_t)(0x3a80u + (channel & 7)));
        for (size_t beat = 0; beat < kWeightBeats; ++beat)
            weights[channel][beat] = weight_value;
        if (channel >= GDN_RECURRENT_STATE_FIRST_PORT) {
            const Beat512 state_value = bf16_pattern(
                (uint16_t)(0x3c80u + channel - GDN_RECURRENT_STATE_FIRST_PORT));
            for (size_t beat = 0; beat < kStateBeatsPerPort; ++beat)
                weights[channel][kWeightBeats + beat] = state_value;
        }
    }

    uint64_t state_before = 0;
    for (int channel = GDN_RECURRENT_STATE_FIRST_PORT;
         channel < GEMV_CHANNELS; ++channel)
        state_before ^= checksum_beats(weights[channel] + kWeightBeats,
                                       kStateBeatsPerPort);

    const int status = gdn_forward(
        aux_weights, workspace,
        weights[0], weights[1], weights[2], weights[3],
        weights[4], weights[5], weights[6], weights[7],
        weights[8], weights[9], weights[10], weights[11],
        weights[12], weights[13], weights[14], weights[15],
        weights[16], weights[17], weights[18], weights[19],
        weights[20], weights[21], weights[22], weights[23],
        weights[24], weights[25], weights[26], weights[27],
        weights[28], weights[29], weights[30], weights[31]);

    int result = status;
    uint64_t state_after = 0;
    for (int channel = GDN_RECURRENT_STATE_FIRST_PORT;
         channel < GEMV_CHANNELS; ++channel)
        state_after ^= checksum_beats(weights[channel] + kWeightBeats,
                                      kStateBeatsPerPort);

    uint64_t logits_checksum = 0xcbf29ce484222325ULL;
    uint32_t nonzero_logits = 0;
    uint32_t nonfinite_logits = 0;
    const Beat512 *logits = workspace + GDN_WS_OFF_LOGITS / 16;
    for (uint32_t beat = 0; beat < GDN_WSF_LOGITS / 16; ++beat) {
        for (uint32_t lane = 0; lane < 16; ++lane) {
            const uint32_t bits =
                (uint32_t)logits[beat].range(32 * lane + 31, 32 * lane);
            logits_checksum ^= bits;
            logits_checksum *= 0x100000001b3ULL;
            if ((bits & 0x7fffffffu) != 0)
                ++nonzero_logits;
            if ((bits & 0x7f800000u) == 0x7f800000u)
                ++nonfinite_logits;
        }
    }

    if (status != 0 || state_after == state_before ||
        nonzero_logits == 0 || nonfinite_logits != 0) {
        fprintf(stderr,
                "one-layer all-BF16 cosim: status=%d state_changed=%d "
                "nonzero_logits=%u nonfinite_logits=%u checksum=0x%016llx\n",
                status, state_after != state_before, nonzero_logits,
                nonfinite_logits, (unsigned long long)logits_checksum);
        result = 1;
    } else {
        fprintf(stdout,
                "one-layer all-BF16 cosim PASS: state checksum 0x%016llx -> "
                "0x%016llx, logits checksum=0x%016llx nonzero=%u\n",
                (unsigned long long)state_before,
                (unsigned long long)state_after,
                (unsigned long long)logits_checksum, nonzero_logits);
    }

    for (int channel = 0; channel < GEMV_CHANNELS; ++channel)
        delete[] weights[channel];
    delete[] workspace;
    delete[] aux_weights;
    return result;
}
