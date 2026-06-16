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
#define GEMV_CHANNELS 8

/* ---- Per-op profiling selector for gdn_forward (Option A: per-op CU launches) --
 * gdn_forward normally runs the full fused forward (prof_op == GDN_OP_ALL). For
 * on-card per-op latency profiling, the host launches the kernel once per
 * (prof_layer, prof_op): the kernel runs exactly ONE sub-op — reading inputs
 * from / writing outputs to the persistent HBM activation BOs — and returns, so
 * XRT times each op individually. Codes 0..17 are per-layer ops (need prof_layer
 * in [0,num_layers)); GDN_OP_EMBED/FINALNORM/LMHEAD/ARGMAX are global (prof_layer
 * ignored); GDN_OP_NOP returns immediately (launch-overhead baseline the host
 * subtracts). Codes 0..17 MUST match the call order in gdn_forward's body. */
#define GDN_OP_ALL        (-1)
#define GDN_OP_RMSNORM1     0   /* attn rmsnorm */
#define GDN_OP_GEMV_Q       1
#define GDN_OP_GEMV_K       2
#define GDN_OP_GEMV_V       3
#define GDN_OP_MATMUL_A     4   /* a-gate (gdn_matmul_tiled) */
#define GDN_OP_MATMUL_B     5   /* b-gate (gdn_matmul_tiled) */
#define GDN_OP_GEMV_GATE    6
#define GDN_OP_CONV_Q       7   /* depthwise conv+silu + pack16_copy */
#define GDN_OP_CONV_K       8
#define GDN_OP_CONV_V       9
#define GDN_OP_RECURRENT   10
#define GDN_OP_ONORM       11   /* output norm + gate */
#define GDN_OP_GEMV_O      12   /* o-proj gemv + residual add */
#define GDN_OP_RMSNORM2    13   /* mlp rmsnorm */
#define GDN_OP_GEMV_MLPG   14
#define GDN_OP_GEMV_MLPU   15
#define GDN_OP_SWIGLU      16
#define GDN_OP_GEMV_MLPD   17   /* mlp_down gemv + residual add */
#define GDN_OP_PERLAYER_COUNT 18
#define GDN_OP_EMBED      100
#define GDN_OP_FINALNORM  101
#define GDN_OP_LMHEAD     102
#define GDN_OP_ARGMAX     103
#define GDN_OP_NOP        104

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
    float *logits;                        /* [vocab] lm_head gemv scratch (decode argmax → x_norm[0]) */
} GDNRunState;

/* Build the GEMV_CHANNELS compact weight shards from the flat weight blob; each
 * of the GEMV_CHANNELS shard buffers (shards[0..GEMV_CHANNELS-1]) is
 * gdn_weight_shard_floats(config) floats. Host-only. */
size_t gdn_weight_shard_floats(const GDNWeightHeader *config);
void gdn_build_weight_shards(const float *weight_data, const GDNWeightHeader *config,
                             float *const shards[]);

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
    const GDNWeightHeader *config,
    const float *weight_data,
    uint32_t max_tokens,
    float *x,
    float *x_norm,
    float *q,
    float *k,
    float *v,
    float *a,
    float *b,
    float *gate,
    float *attn,
    float *tmp_hidden,
    float *mlp_gate,
    float *mlp_up,
    float *recurrent_state,
    float *head_buffer,
    const int32_t *tokens,
    uint32_t num_tokens,
    const float *weight_data_mm,   /* gemv weight shard 0 (dedicated 512-bit master) */
    const float *weight_data_mm2,  /* shard 1 (Stage 2: parallel readers) */
    const float *weight_data_mm3,  /* shard 2 (Stage 2b: N=4) */
    const float *weight_data_mm4,  /* shard 3 */
    const float *weight_data_mm5,  /* shard 4 (Stage 2c: N=8) */
    const float *weight_data_mm6,  /* shard 5 */
    const float *weight_data_mm7,  /* shard 6 */
    const float *weight_data_mm8,  /* shard 7 */
    float *logits,                 /* [vocab] lm_head gemv scratch; argmax → x_norm[0] */
    int32_t prof_layer,            /* per-op profiling: target layer for a per-layer op (ignored unless prof_op is 0..17) */
    int32_t prof_op                /* GDN_OP_ALL(-1) = full fused forward; else run exactly one GDN_OP_* sub-op */
);

/* Single-token decode step (the only host entry): gdn_forward with num_tokens=1
 * against the persistent per-layer recurrent/conv state in the run-state buffers
 * (loaded from the GPU .gdnstate export). */
int gdn_decode_step_host(const GDNModel *model, GDNRunState *state, const int32_t *token);
/* Per-op profiling variant: run exactly one (prof_layer, prof_op) sub-op of the
 * decode step against the persistent run-state (Option A). prof_op==GDN_OP_ALL is
 * the full fused step (== gdn_decode_step_host). */
int gdn_decode_step_op_host(const GDNModel *model, GDNRunState *state,
                            const int32_t *token, int32_t prof_layer, int32_t prof_op);

/* Host-only op schedule for the Option-A per-op profiler: the 18 per-layer
 * sub-ops in gdn_forward execution order (op code + label + coarse type for the
 * breakdown). The embed (pre) and final_norm/lm_head/argmax (post) globals are
 * driven separately by the profiler. */
typedef struct { int32_t op; const char *name; const char *type; } GDNProfOpInfo;
const GDNProfOpInfo *gdn_profile_perlayer_ops(int *count);

void gdn_compute_logits(const GDNModel *model, const float *hidden, float *logits_out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif
