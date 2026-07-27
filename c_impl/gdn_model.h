#ifndef GDN_MODEL_H
#define GDN_MODEL_H

#include <stdint.h>
#include <stddef.h>

/* Parallel HBM weight readers for the decode GEMV (output-stripe split). Each
 * gemv projection's output rows split into GEMV_CHANNELS disjoint shards, read by
 * GEMV_CHANNELS m_axi masters in parallel — the Stage-2 scaling lever. It must
 * equal the count of weight_data_mm* kernel args, the host shard BOs, and the
 * hw.cfg weight_data_mm* channel groups. Defined here so the kernel, the host
 * shard builder, and the run-state all agree on one value. */
#define GEMV_CHANNELS 32
#define GEMV_CLUSTERS 16
#define GEMV_CHANNELS_PER_CLUSTER 2

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
    uint32_t max_tokens;
    float *workspace;   /* step 4 Stage B: one HBM[0] alloc (GDN_WS_FLOATS); the
                         * 15 pointers below are views into it at GDN_WS_OFF_*. */
    float *x;
    float *x_norm;
    float *q;
    float *k;
    float *v;
    float *a;
    float *b;
    float *gate;
    float *attn;
    float *tmp_hidden;
    float *mlp_gate;
    float *mlp_up;
    float *recurrent_state;
    float *head_buffer;
    float *weight_shards[GEMV_CHANNELS];  /* compact gemv weight shards (built from weight_data) */
    float *aux_weights;                   /* compact per-layer non-GEMV weights */
    float *logits;                        /* [vocab] lm_head gemv scratch (decode argmax → x_norm[0]) */
} GDNRunState;

/* Build the GEMV_CHANNELS compact weight shards from the flat weight blob; each
 * of the GEMV_CHANNELS shard buffers (shards[0..GEMV_CHANNELS-1]) is
 * gdn_weight_shard_floats(config) floats. Host-only. */
size_t gdn_weight_shard_floats(const GDNWeightHeader *config);
void gdn_build_weight_shards(const float *weight_data, const GDNWeightHeader *config,
                             float *const shards[]);
size_t gdn_aux_weight_floats(const GDNWeightHeader *config);
void gdn_build_aux_weights(const float *weight_data, const GDNWeightHeader *config,
                           float *aux_weights);

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
    float *workspace,   /* step 4 Stage B: 15 activation/state buffers at GDN_WS_OFF_* */
    const float *weight_data_mm0,
    const float *weight_data_mm1,
    const float *weight_data_mm2,
    const float *weight_data_mm3,
    const float *weight_data_mm4,
    const float *weight_data_mm5,
    const float *weight_data_mm6,
    const float *weight_data_mm7,
    const float *weight_data_mm8,
    const float *weight_data_mm9,
    const float *weight_data_mm10,
    const float *weight_data_mm11,
    const float *weight_data_mm12,
    const float *weight_data_mm13,
    const float *weight_data_mm14,
    const float *weight_data_mm15,
    const float *weight_data_mm16,
    const float *weight_data_mm17,
    const float *weight_data_mm18,
    const float *weight_data_mm19,
    const float *weight_data_mm20,
    const float *weight_data_mm21,
    const float *weight_data_mm22,
    const float *weight_data_mm23,
    const float *weight_data_mm24,
    const float *weight_data_mm25,
    const float *weight_data_mm26,
    const float *weight_data_mm27,
    const float *weight_data_mm28,
    const float *weight_data_mm29,
    const float *weight_data_mm30,
    const float *weight_data_mm31
);

/* Single-token decode step (the only host entry): gdn_forward with num_tokens=1
 * against the persistent per-layer recurrent/conv state in the run-state buffers
 * (loaded from the GPU .gdnstate export). */
int gdn_decode_step_host(const GDNModel *model, GDNRunState *state, const int32_t *token);
void gdn_compute_logits(const GDNModel *model, const float *hidden, float *logits_out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif
