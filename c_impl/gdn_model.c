#include "gdn_model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GDN_WEIGHT_HEADER_BYTES 60

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

    if (fread(model->weight_data, sizeof(float), total_floats, file) != total_floats) {
        gdn_print_error("failed to read weight payload");
        fclose(file);
        gdn_model_free(model);
        return -1;
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

int gdn_run_state_init(GDNRunState *state, const GDNModel *model, uint32_t max_tokens) {
    size_t hidden_tokens;
    size_t head_tokens;
    size_t hidden;
    size_t num_heads;
    size_t intermediate;
    size_t head_dim;
    size_t value_dim;

    memset(state, 0, sizeof(*state));
    if (max_tokens == 0 || max_tokens > model->config.max_seq_len) {
        gdn_print_error("invalid max_tokens for run state");
        return -1;
    }

    state->max_tokens = max_tokens;
    hidden = model->config.hidden_size;
    num_heads = model->config.num_heads;
    intermediate = model->config.intermediate_size;
    head_dim = model->config.head_dim;
    value_dim = hidden / num_heads;
    hidden_tokens = (size_t)max_tokens * hidden;
    head_tokens = (size_t)max_tokens * num_heads;

    if (gdn_alloc_run_buffer(&state->x, hidden_tokens) != 0) return -1;
    if (gdn_alloc_run_buffer(&state->x_norm, hidden_tokens) != 0) return -1;
    if (gdn_alloc_run_buffer(&state->q, hidden_tokens) != 0) return -1;
    if (gdn_alloc_run_buffer(&state->k, hidden_tokens) != 0) return -1;
    if (gdn_alloc_run_buffer(&state->v, hidden_tokens) != 0) return -1;
    if (gdn_alloc_run_buffer(&state->a, head_tokens) != 0) return -1;
    if (gdn_alloc_run_buffer(&state->b, head_tokens) != 0) return -1;
    if (gdn_alloc_run_buffer(&state->gate, hidden_tokens) != 0) return -1;
    if (gdn_alloc_run_buffer(&state->attn, hidden_tokens) != 0) return -1;
    if (gdn_alloc_run_buffer(&state->tmp_hidden, hidden_tokens) != 0) return -1;
    if (gdn_alloc_run_buffer(&state->mlp_gate, (size_t)max_tokens * intermediate) != 0) return -1;
    if (gdn_alloc_run_buffer(&state->mlp_up, (size_t)max_tokens * intermediate) != 0) return -1;
    if (gdn_alloc_run_buffer(&state->recurrent_state, (size_t)num_heads * head_dim * value_dim) != 0) return -1;
    if (gdn_alloc_run_buffer(&state->head_buffer, ((value_dim + 15) / 16) * 16) != 0) return -1;

    return 0;
}

void gdn_run_state_free(GDNRunState *state) {
    free(state->x);
    free(state->x_norm);
    free(state->q);
    free(state->k);
    free(state->v);
    free(state->a);
    free(state->b);
    free(state->gate);
    free(state->attn);
    free(state->tmp_hidden);
    free(state->mlp_gate);
    free(state->mlp_up);
    free(state->recurrent_state);
    free(state->head_buffer);
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

static float gdn_l2_norm_inv(const float *x, uint32_t length) {
    double sum = 0.0;
    uint32_t index;
    l2_accum: for (index = 0; index < length; ++index) {
    #pragma HLS loop_tripcount min=256 max=256  /* always called with head_dim=256 */
        sum += (double)x[index] * (double)x[index];
    }
    return 1.0f / sqrtf((float)sum + 1e-6f);
}

static void gdn_embed_tokens(
    float *x,
    const float *embeddings,
    const int32_t *tokens,
    uint32_t num_tokens,
    uint32_t hidden,
    uint32_t vocab
) {
    uint32_t token_index;

    embed_loop: for (token_index = 0; token_index < num_tokens; ++token_index) {
    #pragma HLS loop_tripcount min=1 max=2048  /* num_tokens: 1..max_seq_len */
        int32_t token = tokens[token_index];
        if (token < 0 || (uint32_t)token >= vocab) {
            gdn_print_error("token id out of range");
            return;
        }
        {
            uint32_t col;
            embed_copy: for (col = 0; col < hidden; ++col) {
            #pragma HLS loop_tripcount min=2048 max=2048  /* hidden=2048 */
                x[(size_t)token_index * hidden + col] = embeddings[(size_t)token * hidden + col];
            }
        }
    }
}

static void gdn_rmsnorm_rows(
    float *out,
    const float *in,
    const float *weight,
    uint32_t num_rows,
    uint32_t num_cols,
    float eps
) {
    uint32_t row;
    rmsnorm_row: for (row = 0; row < num_rows; ++row) {
    #pragma HLS loop_tripcount min=1 max=2048  /* num_tokens: 1..max_seq_len */
        const float *in_row = in + (size_t)row * num_cols;
        float *out_row = out + (size_t)row * num_cols;
        double sum = 0.0;
        uint32_t col;
        float scale;
        rmsnorm_sq: for (col = 0; col < num_cols; ++col) {
        #pragma HLS loop_tripcount min=2048 max=2048  /* always called with hidden=2048 */
            sum += (double)in_row[col] * (double)in_row[col];
        }
        scale = 1.0f / sqrtf((float)(sum / num_cols) + eps);
        rmsnorm_scale: for (col = 0; col < num_cols; ++col) {
        #pragma HLS loop_tripcount min=2048 max=2048  /* always called with hidden=2048 */
            out_row[col] = in_row[col] * scale * weight[col];
        }
    }
}

/* Tile sizes for tiled matmul — chosen to divide evenly into hidden=2048
   and intermediate=5632 (=16*352).  out_dim=8 (a/b proj) is smaller than
   TILE_C but handled by the boundary guard inside the inner loops. */
#define MM_TILE_R 16   /* rows (num_tokens dimension) */
#define MM_TILE_C 16   /* columns (out_dim dimension) */
#define MM_TILE_K 16   /* reduction (in_dim dimension) */

static void gdn_matmul(
    float *out,
    const float *in,
    const float *weights,
    uint32_t num_rows,
    uint32_t in_dim,
    uint32_t out_dim
) {
    uint32_t tr, tc, tk;   /* tile indices */
    uint32_t r, c, k;      /* intra-tile indices */
    float local_in[MM_TILE_R][MM_TILE_K];
    float local_wt[MM_TILE_K][MM_TILE_C];
    float local_out[MM_TILE_R][MM_TILE_C];
    #pragma HLS array_partition variable=local_in  dim=2 complete
    #pragma HLS array_partition variable=local_wt  dim=1 complete
    #pragma HLS array_partition variable=local_out dim=2 complete

    /* Zero the output buffer — replacement for memset so HLS can estimate latency */
    mm_zero_row: for (r = 0; r < num_rows; ++r) {
    #pragma HLS loop_tripcount min=1 max=2048  /* num_tokens: 1..max_seq_len */
        mm_zero_col: for (c = 0; c < out_dim; ++c) {
    #pragma HLS loop_tripcount min=8 max=5632  /* out_dim: num_heads=8 to intermediate=5632 */
    #pragma HLS pipeline II=1
            out[(size_t)r * out_dim + c] = 0.0f;
        }
    }

    /* Tiled matrix multiply: out[R][C] += in[R][K] * weights[C][K]^T
       Weight layout is row-major with weights[out_index] being one row of
       length in_dim, i.e. weights is (out_dim x in_dim). */
    mm_tile_r: for (tr = 0; tr < num_rows; tr += MM_TILE_R) {
    #pragma HLS loop_tripcount min=1 max=128  /* ceil(2048/16) */
        mm_tile_k: for (tk = 0; tk < in_dim; tk += MM_TILE_K) {
    #pragma HLS loop_tripcount min=128 max=352  /* ceil(2048/16) to ceil(5632/16) */

            /* Load input tile: local_in[r][k] = in[tr+r][tk+k] */
            mm_load_in_r: for (r = 0; r < MM_TILE_R; ++r) {
            #pragma HLS loop_tripcount min=16 max=16
                mm_load_in_k: for (k = 0; k < MM_TILE_K; ++k) {
                #pragma HLS loop_tripcount min=16 max=16
                #pragma HLS pipeline II=1
                    uint32_t gr = tr + r;
                    uint32_t gk = tk + k;
                    local_in[r][k] = (gr < num_rows && gk < in_dim)
                        ? in[(size_t)gr * in_dim + gk] : 0.0f;
                }
            }

            mm_tile_c: for (tc = 0; tc < out_dim; tc += MM_TILE_C) {
            #pragma HLS loop_tripcount min=1 max=352  /* ceil(8/16)=1 to ceil(5632/16)=352 */

                /* Load weight tile: local_wt[k][c] = weights[tc+c][tk+k] */
                mm_load_wt_c: for (c = 0; c < MM_TILE_C; ++c) {
                #pragma HLS loop_tripcount min=16 max=16
                    mm_load_wt_k: for (k = 0; k < MM_TILE_K; ++k) {
                    #pragma HLS loop_tripcount min=16 max=16
                    #pragma HLS pipeline II=1
                        uint32_t gc = tc + c;
                        uint32_t gk = tk + k;
                        local_wt[k][c] = (gc < out_dim && gk < in_dim)
                            ? weights[(size_t)gc * in_dim + gk] : 0.0f;
                    }
                }

                /* Load partial sums (only on the first K-tile) */
                if (tk == 0) {
                    mm_load_out_r: for (r = 0; r < MM_TILE_R; ++r) {
                    #pragma HLS loop_tripcount min=16 max=16
                        mm_load_out_c: for (c = 0; c < MM_TILE_C; ++c) {
                        #pragma HLS loop_tripcount min=16 max=16
                        #pragma HLS pipeline II=1
                            local_out[r][c] = 0.0f;
                        }
                    }
                } else {
                    mm_reload_out_r: for (r = 0; r < MM_TILE_R; ++r) {
                    #pragma HLS loop_tripcount min=16 max=16
                        mm_reload_out_c: for (c = 0; c < MM_TILE_C; ++c) {
                        #pragma HLS loop_tripcount min=16 max=16
                        #pragma HLS pipeline II=1
                            uint32_t gr = tr + r;
                            uint32_t gc = tc + c;
                            local_out[r][c] = (gr < num_rows && gc < out_dim)
                                ? out[(size_t)gr * out_dim + gc] : 0.0f;
                        }
                    }
                }

                /* Compute: local_out[r][c] += sum_k local_in[r][k] * local_wt[k][c] */
                mm_comp_r: for (r = 0; r < MM_TILE_R; ++r) {
                #pragma HLS loop_tripcount min=16 max=16
                    mm_comp_k: for (k = 0; k < MM_TILE_K; ++k) {
                    #pragma HLS loop_tripcount min=16 max=16
                    #pragma HLS pipeline II=1
                        float a_val = local_in[r][k];
                        mm_comp_c: for (c = 0; c < MM_TILE_C; ++c) {
                        #pragma HLS loop_tripcount min=16 max=16
                        #pragma HLS unroll
                            local_out[r][c] += a_val * local_wt[k][c];
                        }
                    }
                }

                /* Store output tile back to DRAM */
                mm_store_r: for (r = 0; r < MM_TILE_R; ++r) {
                #pragma HLS loop_tripcount min=16 max=16
                    mm_store_c: for (c = 0; c < MM_TILE_C; ++c) {
                    #pragma HLS loop_tripcount min=16 max=16
                    #pragma HLS pipeline II=1
                        uint32_t gr = tr + r;
                        uint32_t gc = tc + c;
                        if (gr < num_rows && gc < out_dim) {
                            out[(size_t)gr * out_dim + gc] = local_out[r][c];
                        }
                    }
                }

            } /* mm_tile_c */
        } /* mm_tile_k */
    } /* mm_tile_r */
}

static void gdn_depthwise_conv_silu(
    float *out,
    const float *in,
    const float *weights,
    uint32_t num_rows,
    uint32_t num_cols,
    uint32_t kernel_size
) {
    uint32_t row;
    conv_row: for (row = 0; row < num_rows; ++row) {
    #pragma HLS loop_tripcount min=1 max=2048  /* num_tokens: 1..max_seq_len */
        uint32_t col;
        conv_col: for (col = 0; col < num_cols; ++col) {
        #pragma HLS loop_tripcount min=2048 max=2048  /* always called with hidden=2048 */
            float sum = 0.0f;
            uint32_t kernel_index;
            int32_t start = (int32_t)row - (int32_t)kernel_size + 1;
            conv_kern: for (kernel_index = 0; kernel_index < kernel_size; ++kernel_index) {
            #pragma HLS loop_tripcount min=4 max=4  /* conv_size=4 for all layers */
                int32_t source_row = start + (int32_t)kernel_index;
                if (source_row >= 0) {
                    sum += in[(size_t)source_row * num_cols + col] *
                           weights[(size_t)col * kernel_size + kernel_index];
                }
            }
            out[(size_t)row * num_cols + col] = gdn_silu(sum);
        }
    }
}

static void gdn_recurrent_attention(
    float *attn_out,
    float *recurrent_state,
    float *head_buffer,
    const float *q,
    const float *k,
    const float *v,
    const float *a,
    const float *b,
    const float *layer_a_log,
    const float *layer_dt_bias,
    uint32_t hidden,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t num_tokens
) {
    uint32_t value_dim = hidden / num_heads;
    float q_scale = 1.0f / sqrtf((float)head_dim);
    size_t state_size = (size_t)num_heads * head_dim * value_dim;
    uint32_t token_index;

    {
        size_t index;
        state_clear: for (index = 0; index < state_size; ++index) {
        #pragma HLS loop_tripcount min=524288 max=524288  /* num_heads*head_dim*value_dim = 8*256*256 */
            recurrent_state[index] = 0.0f;
        }
    }

    recur_token: for (token_index = 0; token_index < num_tokens; ++token_index) {
    #pragma HLS loop_tripcount min=1 max=2048  /* num_tokens: 1..max_seq_len */
        uint32_t head_index;
        recur_head: for (head_index = 0; head_index < num_heads; ++head_index) {
        #pragma HLS loop_tripcount min=8 max=8  /* num_heads=8 */
            const float *q_head = q + (size_t)token_index * hidden + (size_t)head_index * head_dim;
            const float *k_head = k + (size_t)token_index * hidden + (size_t)head_index * head_dim;
            const float *v_head = v + (size_t)token_index * hidden + (size_t)head_index * value_dim;
            float *state_head = recurrent_state + (size_t)head_index * head_dim * value_dim;
            float *out_head = attn_out + (size_t)token_index * hidden + (size_t)head_index * value_dim;
            float q_inv = gdn_l2_norm_inv(q_head, head_dim);
            float k_inv = gdn_l2_norm_inv(k_head, head_dim);
            float beta = gdn_sigmoid(b[(size_t)token_index * num_heads + head_index]);
            float decay_input = a[(size_t)token_index * num_heads + head_index] + layer_dt_bias[head_index];
            float decay_value = -expf(layer_a_log[head_index]) * gdn_softplus(decay_input);
            float decay = expf(decay_value);
            uint32_t index;
            uint32_t value_index;

            state_decay: for (index = 0; index < head_dim * value_dim; ++index) {
            #pragma HLS loop_tripcount min=65536 max=65536  /* head_dim*value_dim = 256*256 */
                state_head[index] *= decay;
            }

            delta_proj: for (value_index = 0; value_index < value_dim; ++value_index) {
            #pragma HLS loop_tripcount min=256 max=256  /* value_dim = hidden/num_heads = 2048/8 = 256 */
                float projection = 0.0f;
                uint32_t key_index;
                delta_proj_dot: for (key_index = 0; key_index < head_dim; ++key_index) {
                #pragma HLS loop_tripcount min=256 max=256  /* head_dim=256 */
                    projection += state_head[(size_t)key_index * value_dim + value_index] *
                                  (k_head[key_index] * k_inv);
                }
                head_buffer[value_index] = beta * (v_head[value_index] - projection);
            }

            state_update_k: for (index = 0; index < head_dim; ++index) {
            #pragma HLS loop_tripcount min=256 max=256  /* head_dim=256 */
                float normalized_k = k_head[index] * k_inv;
                float *state_row = state_head + (size_t)index * value_dim;
                state_update_v: for (value_index = 0; value_index < value_dim; ++value_index) {
                #pragma HLS loop_tripcount min=256 max=256  /* value_dim = hidden/num_heads = 256 */
                    state_row[value_index] += normalized_k * head_buffer[value_index];
                }
            }

            query_out: for (value_index = 0; value_index < value_dim; ++value_index) {
            #pragma HLS loop_tripcount min=256 max=256  /* value_dim = hidden/num_heads = 256 */
                float sum = 0.0f;
                uint32_t key_index;
                query_out_dot: for (key_index = 0; key_index < head_dim; ++key_index) {
                #pragma HLS loop_tripcount min=256 max=256  /* head_dim=256 */
                    sum += state_head[(size_t)key_index * value_dim + value_index] *
                           (q_head[key_index] * q_inv * q_scale);
                }
                out_head[value_index] = sum;
            }
        }
    }
}

static void gdn_output_norm_and_gate(
    float *attn,
    const float *gate,
    const float *weight,
    uint32_t num_tokens,
    uint32_t num_heads,
    uint32_t head_dim,
    float eps
) {
    uint32_t token_index;
    onorm_token: for (token_index = 0; token_index < num_tokens; ++token_index) {
    #pragma HLS loop_tripcount min=1 max=2048  /* num_tokens: 1..max_seq_len */
        uint32_t head_index;
        onorm_head: for (head_index = 0; head_index < num_heads; ++head_index) {
        #pragma HLS loop_tripcount min=8 max=8  /* num_heads=8 */
            float *attn_head = attn + (size_t)token_index * num_heads * head_dim + (size_t)head_index * head_dim;
            const float *gate_head = gate + (size_t)token_index * num_heads * head_dim + (size_t)head_index * head_dim;
            double sum = 0.0;
            uint32_t index;
            float scale;
            onorm_sq: for (index = 0; index < head_dim; ++index) {
            #pragma HLS loop_tripcount min=256 max=256  /* head_dim=256 */
                sum += (double)attn_head[index] * (double)attn_head[index];
            }
            scale = 1.0f / sqrtf((float)(sum / head_dim) + eps);
            onorm_gate: for (index = 0; index < head_dim; ++index) {
            #pragma HLS loop_tripcount min=256 max=256  /* head_dim=256 */
                float normalized = attn_head[index] * scale * weight[index];
                float gate_value = gate_head[index];
                attn_head[index] = normalized * gate_value * gdn_sigmoid(gate_value);
            }
        }
    }
}

static void gdn_swiglu_inplace(float *gate, const float *up, size_t count) {
    size_t index;
    swiglu_loop: for (index = 0; index < count; ++index) {
    #pragma HLS loop_tripcount min=5632 max=11534336  /* count = num_tokens*intermediate: 1*5632 to 2048*5632 */
        gate[index] = gdn_silu(gate[index]) * up[index];
    }
}

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
) {
    /* Depths match gdn-1.3b-f32.gdnw: hidden=2048 heads=8 head_dim=256
    intermediate=5632 layers=24 conv=4 max_seq_len=2048 vocab=32000 */
    #pragma HLS interface m_axi port=config depth=1 offset=slave
    #pragma HLS interface m_axi port=weight_data depth=1466343808 offset=slave
    #pragma HLS interface m_axi port=x depth=4194304 offset=slave
    #pragma HLS interface m_axi port=x_norm depth=4194304 offset=slave
    #pragma HLS interface m_axi port=q depth=4194304 offset=slave
    #pragma HLS interface m_axi port=k depth=4194304 offset=slave
    #pragma HLS interface m_axi port=v depth=4194304 offset=slave
    #pragma HLS interface m_axi port=a depth=16384 offset=slave
    #pragma HLS interface m_axi port=b depth=16384 offset=slave
    #pragma HLS interface m_axi port=gate depth=4194304 offset=slave
    #pragma HLS interface m_axi port=attn depth=4194304 offset=slave
    #pragma HLS interface m_axi port=tmp_hidden depth=4194304 offset=slave
    #pragma HLS interface m_axi port=mlp_gate depth=11534336 offset=slave
    #pragma HLS interface m_axi port=mlp_up depth=11534336 offset=slave
    #pragma HLS interface m_axi port=recurrent_state depth=524288 offset=slave
    #pragma HLS interface m_axi port=head_buffer depth=256 offset=slave
    #pragma HLS interface m_axi port=tokens depth=2048 offset=slave
    #pragma HLS interface s_axilite port=max_tokens
    #pragma HLS interface s_axilite port=num_tokens
    #pragma HLS interface s_axilite port=return

    uint32_t hidden = config->hidden_size;
    uint32_t num_heads = config->num_heads;
    uint32_t head_dim = config->head_dim;
    uint32_t intermediate = config->intermediate_size;
    uint32_t layer_index;
    size_t hidden_count;
    size_t mlp_count;
    const float *embeddings = weight_data;
    const float *final_norm = weight_data + gdn_final_norm_offset(config);

    if (num_tokens == 0 || num_tokens > max_tokens) {
        gdn_print_error("invalid token count for forward pass");
        return -1;
    }

    hidden_count = (size_t)num_tokens * hidden;
    mlp_count = (size_t)num_tokens * intermediate;

    gdn_embed_tokens(x, embeddings, tokens, num_tokens, hidden, config->vocab_size);
    layer_loop: for (layer_index = 0; layer_index < config->num_layers; ++layer_index) {
    #pragma HLS loop_tripcount min=24 max=24  /* num_layers=24 */
        size_t layer_offset = gdn_layer_weight_offset(config, layer_index);
        const float *layer_attn_norm = weight_data + layer_offset;
        const float *layer_a_log;
        const float *layer_dt_bias;
        const float *layer_q_proj;
        const float *layer_k_proj;
        const float *layer_v_proj;
        const float *layer_a_proj;
        const float *layer_b_proj;
        const float *layer_q_conv;
        const float *layer_k_conv;
        const float *layer_v_conv;
        const float *layer_g_proj;
        const float *layer_o_norm;
        const float *layer_o_proj;
        const float *layer_mlp_norm;
        const float *layer_mlp_gate_proj;
        const float *layer_mlp_up_proj;
        const float *layer_mlp_down_proj;
        size_t index;

        layer_offset += hidden;
        layer_a_log = weight_data + layer_offset;
        layer_offset += num_heads;
        layer_dt_bias = weight_data + layer_offset;
        layer_offset += num_heads;
        layer_q_proj = weight_data + layer_offset;
        layer_offset += (size_t)hidden * hidden;
        layer_k_proj = weight_data + layer_offset;
        layer_offset += (size_t)hidden * hidden;
        layer_v_proj = weight_data + layer_offset;
        layer_offset += (size_t)hidden * hidden;
        layer_a_proj = weight_data + layer_offset;
        layer_offset += (size_t)num_heads * hidden;
        layer_b_proj = weight_data + layer_offset;
        layer_offset += (size_t)num_heads * hidden;
        layer_q_conv = weight_data + layer_offset;
        layer_offset += (size_t)hidden * config->conv_size;
        layer_k_conv = weight_data + layer_offset;
        layer_offset += (size_t)hidden * config->conv_size;
        layer_v_conv = weight_data + layer_offset;
        layer_offset += (size_t)hidden * config->conv_size;
        layer_g_proj = weight_data + layer_offset;
        layer_offset += (size_t)hidden * hidden;
        layer_o_norm = weight_data + layer_offset;
        layer_offset += head_dim;
        layer_o_proj = weight_data + layer_offset;
        layer_offset += (size_t)hidden * hidden;
        layer_mlp_norm = weight_data + layer_offset;
        layer_offset += hidden;
        layer_mlp_gate_proj = weight_data + layer_offset;
        layer_offset += (size_t)intermediate * hidden;
        layer_mlp_up_proj = weight_data + layer_offset;
        layer_offset += (size_t)intermediate * hidden;
        layer_mlp_down_proj = weight_data + layer_offset;

        gdn_rmsnorm_rows(x_norm, x, layer_attn_norm, num_tokens, hidden, config->norm_eps);
        gdn_matmul(q, x_norm, layer_q_proj, num_tokens, hidden, hidden);
        gdn_matmul(k, x_norm, layer_k_proj, num_tokens, hidden, hidden);
        gdn_matmul(v, x_norm, layer_v_proj, num_tokens, hidden, hidden);
        gdn_matmul(a, x_norm, layer_a_proj, num_tokens, hidden, num_heads);
        gdn_matmul(b, x_norm, layer_b_proj, num_tokens, hidden, num_heads);
        gdn_matmul(gate, x_norm, layer_g_proj, num_tokens, hidden, hidden);

        gdn_depthwise_conv_silu(tmp_hidden, q, layer_q_conv, num_tokens, hidden, config->conv_size);
        conv_copy_q: for (index = 0; index < hidden_count; ++index) {
        #pragma HLS loop_tripcount min=2048 max=4194304  /* hidden_count = num_tokens*hidden: 1*2048 to 2048*2048 */
            q[index] = tmp_hidden[index];
        }
        gdn_depthwise_conv_silu(tmp_hidden, k, layer_k_conv, num_tokens, hidden, config->conv_size);
        conv_copy_k: for (index = 0; index < hidden_count; ++index) {
        #pragma HLS loop_tripcount min=2048 max=4194304  /* hidden_count = num_tokens*hidden: 1*2048 to 2048*2048 */
            k[index] = tmp_hidden[index];
        }
        gdn_depthwise_conv_silu(tmp_hidden, v, layer_v_conv, num_tokens, hidden, config->conv_size);
        conv_copy_v: for (index = 0; index < hidden_count; ++index) {
        #pragma HLS loop_tripcount min=2048 max=4194304  /* hidden_count = num_tokens*hidden: 1*2048 to 2048*2048 */
            v[index] = tmp_hidden[index];
        }

        gdn_recurrent_attention(
            attn,
            recurrent_state,
            head_buffer,
            q,
            k,
            v,
            a,
            b,
            layer_a_log,
            layer_dt_bias,
            hidden,
            num_heads,
            head_dim,
            num_tokens
        );
        gdn_output_norm_and_gate(attn, gate, layer_o_norm, num_tokens, num_heads, head_dim, config->norm_eps);
        gdn_matmul(tmp_hidden, attn, layer_o_proj, num_tokens, hidden, hidden);
        attn_residual: for (index = 0; index < hidden_count; ++index) {
        #pragma HLS loop_tripcount min=2048 max=4194304  /* hidden_count = num_tokens*hidden: 1*2048 to 2048*2048 */
            x[index] += tmp_hidden[index];
        }

        gdn_rmsnorm_rows(x_norm, x, layer_mlp_norm, num_tokens, hidden, config->norm_eps);
        gdn_matmul(mlp_gate, x_norm, layer_mlp_gate_proj, num_tokens, hidden, intermediate);
        gdn_matmul(mlp_up, x_norm, layer_mlp_up_proj, num_tokens, hidden, intermediate);
        gdn_swiglu_inplace(mlp_gate, mlp_up, mlp_count);
        gdn_matmul(tmp_hidden, mlp_gate, layer_mlp_down_proj, num_tokens, intermediate, hidden);
        mlp_residual: for (index = 0; index < hidden_count; ++index) {
        #pragma HLS loop_tripcount min=2048 max=4194304  /* hidden_count = num_tokens*hidden: 1*2048 to 2048*2048 */
            x[index] += tmp_hidden[index];
        }
    }

    gdn_rmsnorm_rows(x_norm, x, final_norm, num_tokens, hidden, config->norm_eps);
    return 0;
}

int gdn_forward_host(const GDNModel *model, GDNRunState *state, const int32_t *tokens, uint32_t num_tokens) {
    return gdn_forward(
        &model->config,
        model->weight_data,
        state->max_tokens,
        state->x,
        state->x_norm,
        state->q,
        state->k,
        state->v,
        state->a,
        state->b,
        state->gate,
        state->attn,
        state->tmp_hidden,
        state->mlp_gate,
        state->mlp_up,
        state->recurrent_state,
        state->head_buffer,
        tokens,
        num_tokens
    );
}

void gdn_compute_logits(const GDNModel *model, const float *hidden, float *logits_out) {
    uint32_t vocab = model->config.vocab_size;
    uint32_t hidden_size = model->config.hidden_size;
    uint32_t vocab_index;

    for (vocab_index = 0; vocab_index < vocab; ++vocab_index) {
        const float *weight_row = model->lm_head + (size_t)vocab_index * hidden_size;
        float sum = 0.0f;
        uint32_t hidden_index;
        for (hidden_index = 0; hidden_index < hidden_size; ++hidden_index) {
            sum += hidden[hidden_index] * weight_row[hidden_index];
        }
        logits_out[vocab_index] = sum;
    }
}

int gdn_attn_forward_layer(
    const GDNModel *model,
    GDNRunState *state,
    uint32_t layer_index,
    const float *input,
    float *output,
    uint32_t num_tokens
) {
    uint32_t hidden = model->config.hidden_size;
    uint32_t num_heads = model->config.num_heads;
    uint32_t head_dim = model->config.head_dim;
    uint32_t conv_size = model->config.conv_size;
    size_t hidden_count = (size_t)num_tokens * hidden;
    const GDNLayerWeights *lw;
    size_t index;

    if (layer_index >= model->config.num_layers) {
        gdn_print_error("layer index out of range");
        return -1;
    }
    if (num_tokens == 0 || num_tokens > state->max_tokens) {
        gdn_print_error("invalid token count for attn forward");
        return -1;
    }

    lw = &model->layers[layer_index];

    /* Projections: input -> q, k, v, a, b, gate */
    gdn_matmul(state->q, input, lw->q_proj, num_tokens, hidden, hidden);
    gdn_matmul(state->k, input, lw->k_proj, num_tokens, hidden, hidden);
    gdn_matmul(state->v, input, lw->v_proj, num_tokens, hidden, hidden);
    gdn_matmul(state->a, input, lw->a_proj, num_tokens, hidden, num_heads);
    gdn_matmul(state->b, input, lw->b_proj, num_tokens, hidden, num_heads);
    gdn_matmul(state->gate, input, lw->g_proj, num_tokens, hidden, hidden);

    /* Depthwise conv1d + SiLU on q, k, v */
    gdn_depthwise_conv_silu(state->tmp_hidden, state->q, lw->q_conv, num_tokens, hidden, conv_size);
    for (index = 0; index < hidden_count; ++index) state->q[index] = state->tmp_hidden[index];

    gdn_depthwise_conv_silu(state->tmp_hidden, state->k, lw->k_conv, num_tokens, hidden, conv_size);
    for (index = 0; index < hidden_count; ++index) state->k[index] = state->tmp_hidden[index];

    gdn_depthwise_conv_silu(state->tmp_hidden, state->v, lw->v_conv, num_tokens, hidden, conv_size);
    for (index = 0; index < hidden_count; ++index) state->v[index] = state->tmp_hidden[index];

    /* Recurrent attention */
    gdn_recurrent_attention(
        state->attn, state->recurrent_state, state->head_buffer,
        state->q, state->k, state->v,
        state->a, state->b,
        lw->a_log, lw->dt_bias,
        hidden, num_heads, head_dim, num_tokens
    );

    /* Output norm + gate */
    gdn_output_norm_and_gate(
        state->attn, state->gate, lw->o_norm,
        num_tokens, num_heads, head_dim, model->config.norm_eps
    );

    /* Output projection -> output */
    gdn_matmul(output, state->attn, lw->o_proj, num_tokens, hidden, hidden);

    return 0;
}
