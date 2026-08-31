#ifndef GDN_MODEL_H
#define GDN_MODEL_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
#include "ap_int.h"

using Beat512 = ap_uint<512>;
using Fp32Bits = ap_uint<32>;
using Bf16Bits = ap_uint<16>;

static_assert(sizeof(Beat512) == 64, "Beat512 must occupy one 512-bit beat");
#endif

/* Parallel HBM weight readers for the decode GEMV (output-stripe split). Each
 * gemv projection's output rows split into GEMV_CHANNELS disjoint shards, read by
 * GEMV_CHANNELS m_axi masters in parallel — the Stage-2 scaling lever. It must
 * equal the count of weight_data_mm* kernel args, the host shard BOs, and the
 * hw.cfg weight_data_mm* channel groups. Defined here so the kernel, the host
 * shard builder, and the run-state all agree on one value. */
#define GEMV_CHANNELS 32
#define GEMV_CLUSTERS 16

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char magic[8];
    uint32_t version;
    uint32_t vocab_size;
    uint32_t hidden_size;
    uint32_t num_layers;
    uint32_t num_heads;
    uint32_t num_v_heads;
    uint32_t head_dim;
    uint32_t intermediate_size;
    uint32_t conv_size;
    uint32_t max_seq_len;
    int32_t bos_token_id;
    int32_t eos_token_id;
    float norm_eps;
} GDNWeightHeader;

/* step 4 Stage B: the 15 activation/state buffers packed into one HBM[0]
 * `workspace` pointer, replacing 15 separate kernel args (and their control_s_axi
 * base-address registers). Offsets are in FLOATS and each is 16-float (512-bit)
 * aligned so max_widen_bitwidth=512 still applies (heads=8 padded to 16). Both
 * the kernel (gdn_model.cpp) and both hosts (gdn_run_state_init csim +
 * host.cpp on-card) derive from this ONE layout, so it cannot drift; gdn_model.cpp
 * static_asserts the sizes against the GDN_* dim macros. */
#define GDN_WSF_HID     2048u        /* hidden buffers: x x_norm q k v gate attn tmp */
#define GDN_WSF_HEAD    16u          /* a,b: 8 heads padded to a 512-bit line */
#define GDN_WSF_MLP     5632u        /* mlp_gate, mlp_up */
#define GDN_WSF_STATE   12582912u    /* recurrent_state: 24*8*256*256 */
#define GDN_WSF_HEADBUF 442368u      /* head_buffer: 24*3*3*2048 */
#define GDN_WSF_LOGITS  32000u       /* logits */
#define GDN_WS_OFF_X          ((size_t)0)
#define GDN_WS_OFF_X_NORM     (GDN_WS_OFF_X         + GDN_WSF_HID)
#define GDN_WS_OFF_Q          (GDN_WS_OFF_X_NORM    + GDN_WSF_HID)
#define GDN_WS_OFF_K          (GDN_WS_OFF_Q         + GDN_WSF_HID)
#define GDN_WS_OFF_V          (GDN_WS_OFF_K         + GDN_WSF_HID)
#define GDN_WS_OFF_A          (GDN_WS_OFF_V         + GDN_WSF_HID)
#define GDN_WS_OFF_B          (GDN_WS_OFF_A         + GDN_WSF_HEAD)
#define GDN_WS_OFF_GATE       (GDN_WS_OFF_B         + GDN_WSF_HEAD)
#define GDN_WS_OFF_ATTN       (GDN_WS_OFF_GATE      + GDN_WSF_HID)
#define GDN_WS_OFF_TMP_HIDDEN (GDN_WS_OFF_ATTN      + GDN_WSF_HID)
#define GDN_WS_OFF_MLP_GATE   (GDN_WS_OFF_TMP_HIDDEN + GDN_WSF_HID)
#define GDN_WS_OFF_MLP_UP     (GDN_WS_OFF_MLP_GATE  + GDN_WSF_MLP)
#define GDN_WS_OFF_REC_STATE  (GDN_WS_OFF_MLP_UP    + GDN_WSF_MLP)
#define GDN_WS_OFF_HEAD_BUF   (GDN_WS_OFF_REC_STATE + GDN_WSF_STATE)
#define GDN_WS_OFF_LOGITS     (GDN_WS_OFF_HEAD_BUF  + GDN_WSF_HEADBUF)
#define GDN_WS_FLOATS         (GDN_WS_OFF_LOGITS    + GDN_WSF_LOGITS)

/* Recurrent-state device layout. The external .gdnstate file remains one
 * contiguous [layer][head][K][V] FP32-word tensor whose values are required
 * to be BF16-exact. At upload time those values are packed 32 per Beat512 and
 * striped over the tails of weight shards 28..31.
 * The four corresponding AXI masters are idle while recurrent attention runs,
 * so the kernel gains independent state bandwidth without adding an m_axi
 * interface or changing the public state-file format. */
#define GDN_RECURRENT_STATE_PORTS       4
#define GDN_RECURRENT_STATE_FIRST_PORT 28
#define GDN_RECURRENT_STATE_STRIPE_FLOATS \
    (GDN_WSF_STATE / GDN_RECURRENT_STATE_PORTS)
#define GDN_RECURRENT_STATE_STRIPE_BF16_BEATS \
    (GDN_RECURRENT_STATE_STRIPE_FLOATS / 32u)
#define GDN_COMPILED_WEIGHT_SHARD_BEATS 1366528u
#define GDN_COMPILED_WEIGHT_SHARD_BYTES \
    ((size_t)GDN_COMPILED_WEIGHT_SHARD_BEATS * sizeof(Beat512))

/* The workspace ABI reserves the original FP32-sized convolution-tail region.
 * Each layer/kind stripe now stores its BF16 payload in the first half of that
 * reservation; the second half remains unused so every GDN_WS_OFF_* value is
 * unchanged. */
#define GDN_CONV_TAIL_STRIPES (24u * 3u)
#define GDN_CONV_TAIL_FLOATS_PER_STRIPE \
    (GDN_WSF_HEADBUF / GDN_CONV_TAIL_STRIPES)
#define GDN_CONV_TAIL_RESERVED_BEATS_PER_STRIPE \
    (GDN_CONV_TAIL_FLOATS_PER_STRIPE / 16u)
#define GDN_CONV_TAIL_BF16_BEATS_PER_STRIPE \
    (GDN_CONV_TAIL_FLOATS_PER_STRIPE / 32u)

typedef struct {
    const float *attn_norm;
    const float *a_log;
    const float *dt_bias;
    const float *q_proj;
    const float *k_proj;
    const float *v_proj;
    const float *a_proj;
    const float *b_proj;
    const float *q_conv;
    const float *k_conv;
    const float *v_conv;
    const float *g_proj;
    const float *o_norm;
    const float *o_proj;
    const float *mlp_norm;
    const float *mlp_gate_proj;
    const float *mlp_up_proj;
    const float *mlp_down_proj;
} GDNLayerWeights;

typedef struct {
    GDNWeightHeader config;
    float *weight_data;
    GDNLayerWeights *layers;
    const float *embeddings;
    const float *final_norm;
    const float *lm_head;
} GDNModel;

typedef struct {
    Beat512 *workspace; /* one 64-byte-aligned raw-beat allocation */
    float *x;
    float *recurrent_state;
    float *head_buffer;
    Beat512 *weight_shards[GEMV_CHANNELS]; /* packed-BF16 GEMV shards */
    float *aux_weights;                   /* compact per-layer non-GEMV weights */
} GDNRunState;

typedef struct {
    uint64_t calls;
    uint64_t special_inputs;
    uint64_t flushed_inputs;
    uint64_t flushed_outputs;
    uint64_t overflows;
} GDNMixedMulStats;

/* Build the GEMV_CHANNELS compact packed-BF16 shards from the flat FP32-word
 * blob. Every source weight must already be BF16-exact. Host-only. */
size_t gdn_weight_shard_beats(const GDNWeightHeader *config);
size_t gdn_weight_shard_bytes(const GDNWeightHeader *config);
int gdn_validate_bf16_exact_weights(const float *weight_data,
                                    const GDNWeightHeader *config);
void gdn_build_weight_shards(const float *weight_data, const GDNWeightHeader *config,
                             Beat512 *const shards[]);
int gdn_validate_weight_shards(const float *weight_data,
                               const GDNWeightHeader *config,
                               Beat512 *const shards[]);
size_t gdn_aux_weight_floats(const GDNWeightHeader *config);
void gdn_build_aux_weights(const float *weight_data, const GDNWeightHeader *config,
                           float *aux_weights);
int gdn_scatter_recurrent_state(Beat512 *const state_stripes[],
                                const float *recurrent_state,
                                size_t recurrent_state_floats);
int gdn_pack_conv_tails_bf16(Beat512 *workspace_head_buffer,
                             const float *conv_tails,
                             size_t conv_tail_floats);

/* Iter66 native-BF16 contract: multiply two BF16 operands, RNE-round the
 * product to BF16 with AMD FPO DAZ/FTZ semantics, then widen by wiring for the
 * existing FP32 accumulation tree. */
float gdn_native_bf16_mul_to_fp32(Bf16Bits weight, Bf16Bits activation);

#ifndef __SYNTHESIS__
void gdn_reset_mixed_mul_stats(void);
GDNMixedMulStats gdn_get_mixed_mul_stats(void);
uint16_t gdn_test_fp32_to_bf16_rne_bits(uint32_t bits);
uint32_t gdn_test_bf16_to_fp32_bits(uint16_t bits);
void gdn_test_set_fp32_lane_bits(Beat512 *beat, uint32_t lane, uint32_t bits);
uint32_t gdn_test_get_fp32_lane_bits(const Beat512 *beat, uint32_t lane);
void gdn_test_set_bf16_lane_bits(Beat512 *beat, uint32_t lane, uint16_t bits);
uint16_t gdn_test_get_bf16_lane_bits(const Beat512 *beat, uint32_t lane);
#endif

int gdn_model_load(GDNModel *model, const char *path);
void gdn_model_free(GDNModel *model);

int gdn_run_state_init(GDNRunState *state, const GDNModel *model, uint32_t max_tokens);
void gdn_run_state_free(GDNRunState *state);

/* Decode-only forward (the kernel top). Forwards exactly one token (num_tokens
 * must be 1) through the GEMV datapath against the persistent per-layer recurrent
 * + conv state in recurrent_state / head_buffer (loaded from the GPU .gdnstate
 * export). The state is restored at each layer's start and saved at its end —
 * there is no prefill / no GEMM / no mode flag. */
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
);

/* Single-token decode step (the only host entry): gdn_forward with num_tokens=1
 * against the persistent per-layer recurrent/conv state in the run-state buffers
 * (loaded from the GPU .gdnstate export). */
int gdn_decode_step_host(const GDNModel *model, GDNRunState *state, const int32_t *token);
void gdn_compute_logits(const GDNModel *model, const float *hidden, float *logits_out);

/* Native-only visibility into the final normalized hidden vector and the
 * complete LM-head result immediately before the synthesized argmax.  HLS
 * removes both the setter and all associated stores under __SYNTHESIS__, so
 * this does not add an AXI port or alter the production kernel ABI. */
#ifndef __SYNTHESIS__
void gdn_set_native_debug_buffers(float *final_hidden, float *logits);
#endif

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif
