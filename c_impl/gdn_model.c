#include "gdn_model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GDN_WEIGHT_HEADER_BYTES 60

static void gdn_print_error(const char *message) {
    fprintf(stderr, "%s\n", message);
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
    if (gdn_alloc_run_buffer(&state->head_buffer, value_dim) != 0) return -1;

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
    for (index = 0; index < length; ++index) {
        sum += (double)x[index] * (double)x[index];
    }
    return 1.0f / sqrtf((float)sum + 1e-6f);
}

static void gdn_embed_tokens(float *x, const GDNModel *model, const int32_t *tokens, uint32_t num_tokens) {
    uint32_t token_index;
    uint32_t hidden = model->config.hidden_size;
    uint32_t vocab = model->config.vocab_size;

    for (token_index = 0; token_index < num_tokens; ++token_index) {
        int32_t token = tokens[token_index];
        if (token < 0 || (uint32_t)token >= vocab) {
            gdn_print_error("token id out of range");
            return;
        }
        memcpy(
            x + (size_t)token_index * hidden,
            model->embeddings + (size_t)token * hidden,
            (size_t)hidden * sizeof(float)
        );
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
    for (row = 0; row < num_rows; ++row) {
        const float *in_row = in + (size_t)row * num_cols;
        float *out_row = out + (size_t)row * num_cols;
        double sum = 0.0;
        uint32_t col;
        float scale;
        for (col = 0; col < num_cols; ++col) {
            sum += (double)in_row[col] * (double)in_row[col];
        }
        scale = 1.0f / sqrtf((float)(sum / num_cols) + eps);
        for (col = 0; col < num_cols; ++col) {
            out_row[col] = in_row[col] * scale * weight[col];
        }
    }
}

static void gdn_matmul(
    float *out,
    const float *in,
    const float *weights,
    uint32_t num_rows,
    uint32_t in_dim,
    uint32_t out_dim
) {
    uint32_t row;
    for (row = 0; row < num_rows; ++row) {
        const float *in_row = in + (size_t)row * in_dim;
        float *out_row = out + (size_t)row * out_dim;
        uint32_t out_index;
        for (out_index = 0; out_index < out_dim; ++out_index) {
            const float *weight_row = weights + (size_t)out_index * in_dim;
            float sum = 0.0f;
            uint32_t in_index;
            for (in_index = 0; in_index < in_dim; ++in_index) {
                sum += in_row[in_index] * weight_row[in_index];
            }
            out_row[out_index] = sum;
        }
    }
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
    for (row = 0; row < num_rows; ++row) {
        uint32_t col;
        for (col = 0; col < num_cols; ++col) {
            float sum = 0.0f;
            uint32_t kernel_index;
            int32_t start = (int32_t)row - (int32_t)kernel_size + 1;
            for (kernel_index = 0; kernel_index < kernel_size; ++kernel_index) {
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
    GDNRunState *state,
    const GDNModel *model,
    const GDNLayerWeights *layer,
    uint32_t num_tokens
) {
    uint32_t hidden = model->config.hidden_size;
    uint32_t num_heads = model->config.num_heads;
    uint32_t head_dim = model->config.head_dim;
    uint32_t value_dim = hidden / num_heads;
    float q_scale = 1.0f / sqrtf((float)head_dim);
    size_t state_size = (size_t)num_heads * head_dim * value_dim;
    uint32_t token_index;

    memset(state->recurrent_state, 0, state_size * sizeof(float));

    for (token_index = 0; token_index < num_tokens; ++token_index) {
        uint32_t head_index;
        for (head_index = 0; head_index < num_heads; ++head_index) {
            const float *q_head = state->q + (size_t)token_index * hidden + (size_t)head_index * head_dim;
            const float *k_head = state->k + (size_t)token_index * hidden + (size_t)head_index * head_dim;
            const float *v_head = state->v + (size_t)token_index * hidden + (size_t)head_index * value_dim;
            float *state_head = state->recurrent_state + (size_t)head_index * head_dim * value_dim;
            float *out_head = attn_out + (size_t)token_index * hidden + (size_t)head_index * value_dim;
            float q_inv = gdn_l2_norm_inv(q_head, head_dim);
            float k_inv = gdn_l2_norm_inv(k_head, head_dim);
            float beta = gdn_sigmoid(state->b[(size_t)token_index * num_heads + head_index]);
            float decay_input = state->a[(size_t)token_index * num_heads + head_index] + layer->dt_bias[head_index];
            float decay_value = -expf(layer->a_log[head_index]) * gdn_softplus(decay_input);
            float decay = expf(decay_value);
            uint32_t index;
            uint32_t value_index;

            for (index = 0; index < head_dim * value_dim; ++index) {
                state_head[index] *= decay;
            }

            for (value_index = 0; value_index < value_dim; ++value_index) {
                float projection = 0.0f;
                uint32_t key_index;
                for (key_index = 0; key_index < head_dim; ++key_index) {
                    projection += state_head[(size_t)key_index * value_dim + value_index] *
                                  (k_head[key_index] * k_inv);
                }
                state->head_buffer[value_index] = beta * (v_head[value_index] - projection);
            }

            for (index = 0; index < head_dim; ++index) {
                float normalized_k = k_head[index] * k_inv;
                float *state_row = state_head + (size_t)index * value_dim;
                for (value_index = 0; value_index < value_dim; ++value_index) {
                    state_row[value_index] += normalized_k * state->head_buffer[value_index];
                }
            }

            for (value_index = 0; value_index < value_dim; ++value_index) {
                float sum = 0.0f;
                uint32_t key_index;
                for (key_index = 0; key_index < head_dim; ++key_index) {
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
    for (token_index = 0; token_index < num_tokens; ++token_index) {
        uint32_t head_index;
        for (head_index = 0; head_index < num_heads; ++head_index) {
            float *attn_head = attn + (size_t)token_index * num_heads * head_dim + (size_t)head_index * head_dim;
            const float *gate_head = gate + (size_t)token_index * num_heads * head_dim + (size_t)head_index * head_dim;
            double sum = 0.0;
            uint32_t index;
            float scale;
            for (index = 0; index < head_dim; ++index) {
                sum += (double)attn_head[index] * (double)attn_head[index];
            }
            scale = 1.0f / sqrtf((float)(sum / head_dim) + eps);
            for (index = 0; index < head_dim; ++index) {
                float normalized = attn_head[index] * scale * weight[index];
                float gate_value = gate_head[index];
                attn_head[index] = normalized * gate_value * gdn_sigmoid(gate_value);
            }
        }
    }
}

static void gdn_swiglu_inplace(float *gate, const float *up, size_t count) {
    size_t index;
    for (index = 0; index < count; ++index) {
        gate[index] = gdn_silu(gate[index]) * up[index];
    }
}

int gdn_forward(const GDNModel *model, GDNRunState *state, const int32_t *tokens, uint32_t num_tokens) {
    uint32_t hidden = model->config.hidden_size;
    uint32_t num_heads = model->config.num_heads;
    uint32_t head_dim = model->config.head_dim;
    uint32_t intermediate = model->config.intermediate_size;
    uint32_t layer_index;
    size_t hidden_count;
    size_t mlp_count;

    if (num_tokens == 0 || num_tokens > state->max_tokens) {
        gdn_print_error("invalid token count for forward pass");
        return -1;
    }

    hidden_count = (size_t)num_tokens * hidden;
    mlp_count = (size_t)num_tokens * intermediate;

    gdn_embed_tokens(state->x, model, tokens, num_tokens);
    for (layer_index = 0; layer_index < model->config.num_layers; ++layer_index) {
        const GDNLayerWeights *layer = &model->layers[layer_index];
        size_t index;

        gdn_rmsnorm_rows(state->x_norm, state->x, layer->attn_norm, num_tokens, hidden, model->config.norm_eps);
        gdn_matmul(state->q, state->x_norm, layer->q_proj, num_tokens, hidden, hidden);
        gdn_matmul(state->k, state->x_norm, layer->k_proj, num_tokens, hidden, hidden);
        gdn_matmul(state->v, state->x_norm, layer->v_proj, num_tokens, hidden, hidden);
        gdn_matmul(state->a, state->x_norm, layer->a_proj, num_tokens, hidden, num_heads);
        gdn_matmul(state->b, state->x_norm, layer->b_proj, num_tokens, hidden, num_heads);
        gdn_matmul(state->gate, state->x_norm, layer->g_proj, num_tokens, hidden, hidden);

        gdn_depthwise_conv_silu(state->tmp_hidden, state->q, layer->q_conv, num_tokens, hidden, model->config.conv_size);
        memcpy(state->q, state->tmp_hidden, hidden_count * sizeof(float));
        gdn_depthwise_conv_silu(state->tmp_hidden, state->k, layer->k_conv, num_tokens, hidden, model->config.conv_size);
        memcpy(state->k, state->tmp_hidden, hidden_count * sizeof(float));
        gdn_depthwise_conv_silu(state->tmp_hidden, state->v, layer->v_conv, num_tokens, hidden, model->config.conv_size);
        memcpy(state->v, state->tmp_hidden, hidden_count * sizeof(float));

        gdn_recurrent_attention(state->attn, state, model, layer, num_tokens);
        gdn_output_norm_and_gate(state->attn, state->gate, layer->o_norm, num_tokens, num_heads, head_dim, model->config.norm_eps);
        gdn_matmul(state->tmp_hidden, state->attn, layer->o_proj, num_tokens, hidden, hidden);
        for (index = 0; index < hidden_count; ++index) {
            state->x[index] += state->tmp_hidden[index];
        }

        gdn_rmsnorm_rows(state->x_norm, state->x, layer->mlp_norm, num_tokens, hidden, model->config.norm_eps);
        gdn_matmul(state->mlp_gate, state->x_norm, layer->mlp_gate_proj, num_tokens, hidden, intermediate);
        gdn_matmul(state->mlp_up, state->x_norm, layer->mlp_up_proj, num_tokens, hidden, intermediate);
        gdn_swiglu_inplace(state->mlp_gate, state->mlp_up, mlp_count);
        gdn_matmul(state->tmp_hidden, state->mlp_gate, layer->mlp_down_proj, num_tokens, intermediate, hidden);
        for (index = 0; index < hidden_count; ++index) {
            state->x[index] += state->tmp_hidden[index];
        }
    }

    gdn_rmsnorm_rows(state->x_norm, state->x, model->final_norm, num_tokens, hidden, model->config.norm_eps);
    return 0;
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
