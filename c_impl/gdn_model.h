#ifndef GDN_MODEL_H
#define GDN_MODEL_H

#include <stdint.h>
#include <stddef.h>

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
} GDNRunState;

int gdn_model_load(GDNModel *model, const char *path);
void gdn_model_free(GDNModel *model);

int gdn_run_state_init(GDNRunState *state, const GDNModel *model, uint32_t max_tokens);
void gdn_run_state_free(GDNRunState *state);

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
    uint32_t num_tokens
);
int gdn_forward_host(const GDNModel *model, GDNRunState *state, const int32_t *tokens, uint32_t num_tokens);
void gdn_compute_logits(const GDNModel *model, const float *hidden, float *logits_out);

int gdn_attn_forward(
    const GDNWeightHeader *config,
    const float *weight_data,
    uint32_t layer_index,
    const float *input,
    float *output,
    float *q,
    float *k,
    float *v,
    float *a,
    float *b,
    float *gate,
    float *attn,
    float *tmp_hidden,
    float *recurrent_state,
    float *head_buffer,
    uint32_t num_tokens
);

int gdn_attn_forward_layer(
    const GDNModel *model,
    GDNRunState *state,
    uint32_t layer_index,
    const float *input,
    float *output,
    uint32_t num_tokens
);

/* HLS top wrapper for the tiled matmul: out = in * weights^T.
 *   in       is num_rows x in_dim   (row-major)
 *   weights  is out_dim x in_dim    (row-major)
 *   out      is num_rows x out_dim  (row-major)
 * Each pointer is mapped to its own AXI master bundle so load_in / load_wt /
 * store_out can be parallelised in any future dataflow refactor.
 * Returns 0 on success. */
int gdn_matmul_top(
    float *out,
    const float *in,
    const float *weights,
    uint32_t num_rows,
    uint32_t in_dim,
    uint32_t out_dim
);

#endif
