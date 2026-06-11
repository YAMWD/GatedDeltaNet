#include "gdn_model.h"

#include "hls_stream.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GDN_WEIGHT_HEADER_BYTES 60

/* Compile-time constants for GDN-1.3B (used by on-chip state and parallelism) */
#define GDN_HEADS   8
#define GDN_DK    256   /* head_dim = query/key dimension */
#define GDN_DV    256   /* value_dim = hidden/num_heads   */
#define GDN_PK     16   /* column parallelism factor      */

/* Pack16 = 16 FP32 values = 64 bytes = 512 bits. Used both by the systolic
 * matmul (as the stream word) and by the element-wise Pack16 helpers
 * (gdn_pack16_copy / gdn_pack16_add_inplace) further below so the 512-bit
 * m_axi adapter can carry one beat per pipelined iteration. */
struct Pack16 {
    float data[16];
};

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
    /* Decode persistence: recurrent_state holds ALL layers (24 x 2 MB = 48 MB)
     * and head_buffer is repurposed as the conv tail store: per layer, 3 convs
     * (q/k/v) x (conv_size-1) rows x hidden floats (~1.7 MB). Prefill ignores
     * both unless GDN_DECODE_* flags are set. */
    if (gdn_alloc_run_buffer(&state->recurrent_state,
            (size_t)model->config.num_layers * num_heads * head_dim * value_dim) != 0) return -1;
    if (gdn_alloc_run_buffer(&state->head_buffer,
            (size_t)model->config.num_layers * 3 * (model->config.conv_size - 1) * hidden) != 0) return -1;

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

/* 256-input fully unrolled balanced fadd tree (8 levels, depth log2(256)=8).
 * HLS's auto-balance on `sum += arr[i]` produced a 256-deep serial chain
 * instead of a tree, which makes the reduction the bottleneck of any II=1
 * pipelined loop that feeds it. Calling this helper from the reduction site
 * forces an explicit paired-sum tree shape. The function is `inline` so it
 * lives in the caller's pipeline scope. */
static float gdn_tree_reduce_256(const float arr[256]) {
#pragma HLS inline
    float l128[128];
    float l64[64];
    float l32[32];
    float l16[16];
    float l8[8];
    float l4[4];
    float l2[2];
    #pragma HLS array_partition variable=l128 complete
    #pragma HLS array_partition variable=l64  complete
    #pragma HLS array_partition variable=l32  complete
    #pragma HLS array_partition variable=l16  complete
    #pragma HLS array_partition variable=l8   complete
    #pragma HLS array_partition variable=l4   complete
    #pragma HLS array_partition variable=l2   complete

    uint32_t i;
    L128: for (i = 0; i < 128; ++i) { _Pragma("HLS unroll") l128[i] = arr[2*i]   + arr[2*i+1];   }
    L64:  for (i = 0; i < 64;  ++i) { _Pragma("HLS unroll") l64[i]  = l128[2*i]  + l128[2*i+1];  }
    L32:  for (i = 0; i < 32;  ++i) { _Pragma("HLS unroll") l32[i]  = l64[2*i]   + l64[2*i+1];   }
    L16:  for (i = 0; i < 16;  ++i) { _Pragma("HLS unroll") l16[i]  = l32[2*i]   + l32[2*i+1];   }
    L8:   for (i = 0; i < 8;   ++i) { _Pragma("HLS unroll") l8[i]   = l16[2*i]   + l16[2*i+1];   }
    L4:   for (i = 0; i < 4;   ++i) { _Pragma("HLS unroll") l4[i]   = l8[2*i]    + l8[2*i+1];    }
    L2:   for (i = 0; i < 2;   ++i) { _Pragma("HLS unroll") l2[i]   = l4[2*i]    + l4[2*i+1];    }
    return l2[0] + l2[1];
}

/* ============================================================
 * Element-wise helpers vectorised over Pack16 (16 FP32 lanes per beat).
 *
 * These wrap the common load/op/store patterns that used to live as
 * inline scalar loops in gdn_forward / gdn_attn_forward. Going through
 * Pack16 lets HLS use the 512-bit m_axi adapter for one wide read + one
 * wide write per pipeline iteration, instead of one narrow access per
 * float. Profiling on the prior bitstream showed those scalar loops
 * accounting for ~58% of per-layer cycles even though they're trivial
 * arithmetic; this rewrite is the actual fix.
 *
 * Callers pass element counts that are always divisible by 16:
 *   hidden_count = num_tokens × hidden (hidden=2048 ⇒ /16 OK)
 *   mlp_count    = num_tokens × intermediate (intermediate=5632 ⇒ /16 OK)
 * Both source and destination XRT buffers are page-aligned (≥ 4 KiB)
 * by xrt::bo allocation, so Pack16 alignment is satisfied.
 * ============================================================ */

/* dst[i] = src[i] for count floats. count must be multiple of 16. */
static void gdn_pack16_copy(float *dst, const float *src, size_t count) {
    Pack16 *d = reinterpret_cast<Pack16 *>(dst);
    const Pack16 *s = reinterpret_cast<const Pack16 *>(src);
    const size_t count16 = count >> 4;
    pack16_copy: for (size_t i = 0; i < count16; ++i) {
    #pragma HLS loop_tripcount min=128 max=262144  /* 2048/16 .. 4M/16 */
    #pragma HLS pipeline II=1
        d[i] = s[i];
    }
}

/* dst[i] += src[i] for count floats. count must be multiple of 16. */
static void gdn_pack16_add_inplace(float *dst, const float *src, size_t count) {
    Pack16 *d = reinterpret_cast<Pack16 *>(dst);
    const Pack16 *s = reinterpret_cast<const Pack16 *>(src);
    const size_t count16 = count >> 4;
    pack16_add: for (size_t i = 0; i < count16; ++i) {
    #pragma HLS loop_tripcount min=128 max=262144  /* 2048/16 .. 4M/16 */
    #pragma HLS pipeline II=1
        Pack16 ld = d[i];
        Pack16 ls = s[i];
        pack16_add_lane: for (int j = 0; j < 16; ++j) {
        #pragma HLS unroll
            ld.data[j] += ls.data[j];
        }
        d[i] = ld;
    }
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
    /* Pack16-widened activation I/O: read/write 16 cols (512-bit) per beat by
     * indexing the Pack16 *base* by an integer (row*col_packs + cp) — a
     * pre-offset float pointer (in + row*num_cols) would leave alignment
     * unprovable and HLS would demote the access to 32-bit. num_cols is always
     * hidden=2048 (16 | num_cols). */
    const Pack16 *in_p  = reinterpret_cast<const Pack16 *>(in);
    Pack16       *out_p = reinterpret_cast<Pack16 *>(out);
    uint32_t col_packs = num_cols / 16;

    /* Buffer the per-channel norm weight once (it is otherwise re-read every
     * row); cyclic/16 so the scale pass reads 16 lanes in parallel. */
    float w_loc[2048];
    #pragma HLS array_partition variable=w_loc cyclic factor=16
    rms_load_w: for (uint32_t c = 0; c < num_cols; ++c) {
    #pragma HLS loop_tripcount min=2048 max=2048
    #pragma HLS pipeline II=1
        w_loc[c] = weight[c];
    }

    rmsnorm_row: for (uint32_t row = 0; row < num_rows; ++row) {
    #pragma HLS loop_tripcount min=1 max=2048  /* num_tokens: 1..max_seq_len */
        /* sum of squares — 16 squares/beat reduced into a double accumulator
         * (double preserves the precision of the original serial reduction). */
        double sum = 0.0;
        rmsnorm_sq: for (uint32_t cp = 0; cp < col_packs; ++cp) {
        #pragma HLS loop_tripcount min=128 max=128
        #pragma HLS pipeline II=1
            Pack16 v = in_p[(size_t)row * col_packs + cp];
            float s = 0.0f;
            sq_lane: for (int kk = 0; kk < 16; ++kk) {
            #pragma HLS unroll
                s += v.data[kk] * v.data[kk];
            }
            sum += (double)s;
        }
        float scale = 1.0f / sqrtf((float)(sum / num_cols) + eps);
        rmsnorm_scale: for (uint32_t cp = 0; cp < col_packs; ++cp) {
        #pragma HLS loop_tripcount min=128 max=128
        #pragma HLS pipeline II=1
            Pack16 v = in_p[(size_t)row * col_packs + cp];
            Pack16 o;
            scl_lane: for (int kk = 0; kk < 16; ++kk) {
            #pragma HLS unroll
                o.data[kk] = v.data[kk] * scale * w_loc[cp * 16 + kk];
            }
            out_p[(size_t)row * col_packs + cp] = o;
        }
    }
}


/* Tile sizes for tiled matmul — chosen to divide evenly into hidden=2048
   and intermediate=5632 (=16*352).  out_dim=8 (a/b proj) is smaller than
   TILE_C but handled by the boundary guard inside the inner loops. */
#define MM_TILE_R 16   /* rows (num_tokens dimension) */
#define MM_TILE_C 16   /* columns (out_dim dimension) */
#define MM_TILE_K 16   /* reduction (in_dim dimension) */

static void gdn_matmul_tiled(
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
    #pragma HLS loop_tripcount min=128 max=128  /* a/b_proj only: in_dim=2048, ceil(2048/16)=128 */

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
            #pragma HLS loop_tripcount min=1 max=1  /* a/b_proj only: out_dim=8 < MM_TILE_C=16, single iter */

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

                /* Compute: local_out[r][c] += sum_k local_in[r][k] * local_wt[k][c]
                 *
                 * Manually flattened (R x C) loop: HLS won't auto-flatten the
                 * mm_comp_r/mm_comp_c nest (the warning "outer loop is not a
                 * perfect loop" appears because the surrounding tile loop has
                 * statements before/after this pair). Manual flatten avoids
                 * 16x pipeline-fill overhead (each fill costs ~30 cycles).
                 *
                 * Within the loop body:
                 *   - mm_comp_k is fully unrolled into 16 parallel fmuls.
                 *   - The 16 partial products are reduced through a balanced
                 *     4-level fadd tree (paired sums), so the critical path is
                 *     log2(16) fadd stages instead of a 16-deep serial chain
                 *     (which is what HLS auto-generated from `dot += ...` and
                 *     stretched the iteration latency to 55 cycles).
                 *   - dependence false on local_out: the carried dep is
                 *     genuinely false because each iteration touches a unique
                 *     (r, c) pair, and dim 2 complete partition makes each c
                 *     a distinct register bank.
                 */
                mm_comp_rc: for (uint32_t rc = 0; rc < MM_TILE_R * MM_TILE_C; ++rc) {
                #pragma HLS loop_tripcount min=256 max=256
                #pragma HLS pipeline II=1
                #pragma HLS dependence variable=local_out type=inter direction=RAW false
                    uint32_t r = rc / MM_TILE_C;
                    uint32_t c = rc % MM_TILE_C;

                    float p[MM_TILE_K];
                    mm_comp_k: for (k = 0; k < MM_TILE_K; ++k) {
                    #pragma HLS loop_tripcount min=16 max=16
                    #pragma HLS unroll
                        p[k] = local_in[r][k] * local_wt[k][c];
                    }

                    /* Balanced 4-level adder tree: 16 -> 8 -> 4 -> 2 -> 1 */
                    float s2_0 = p[0]  + p[1],  s2_1 = p[2]  + p[3];
                    float s2_2 = p[4]  + p[5],  s2_3 = p[6]  + p[7];
                    float s2_4 = p[8]  + p[9],  s2_5 = p[10] + p[11];
                    float s2_6 = p[12] + p[13], s2_7 = p[14] + p[15];

                    float s4_0 = s2_0 + s2_1, s4_1 = s2_2 + s2_3;
                    float s4_2 = s2_4 + s2_5, s4_3 = s2_6 + s2_7;

                    float s8_0 = s4_0 + s4_1, s8_1 = s4_2 + s4_3;

                    float dot  = s8_0 + s8_1;

                    local_out[r][c] += dot;
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

/* No dispatch wrapper — gdn_forward calls gdn_gemv directly for the large
 * decode projections (defined below) and gdn_matmul_tiled for the small
 * a/b_proj shapes (out_dim=8). Direct calls let HLS allocate and report only
 * the path actually used. */

/* Compile-time bounds for the conv buffers. The model always calls this with
 * hidden=2048 and conv_size=4; smaller calls still fit. */
#define GDN_CONV_COLS_MAX 2048
#define GDN_CONV_K_MAX    4

static void gdn_depthwise_conv_silu(
    float *out,
    const float *in,
    const float *weights,
    float *conv_tail,        /* decode: last (kernel_size-1) input rows for this (layer, conv) */
    uint32_t num_rows,
    uint32_t num_cols,
    uint32_t kernel_size
) {
    /* Buffered weights (per-channel, num_cols x kernel_size) and a 4-row
     * sliding window over the input. Together these eliminate the AXI-port
     * contention that prevented conv_col from pipelining when the kernel was
     * unrolled with raw m_axi loads. */
    float w_loc[GDN_CONV_COLS_MAX][GDN_CONV_K_MAX];
    #pragma HLS array_partition variable=w_loc dim=2 complete
    #pragma HLS array_partition variable=w_loc dim=1 cyclic factor=16  /* 16 channels/beat */

    float in_window[GDN_CONV_K_MAX][GDN_CONV_COLS_MAX];
    #pragma HLS array_partition variable=in_window dim=1 complete
    #pragma HLS array_partition variable=in_window dim=2 cyclic factor=16

    /* Pack16-widened activation I/O: 16 channels (512-bit) per beat. conv is
     * depthwise, so channels are independent and contiguous — index the Pack16
     * base by an integer (row*col_packs + cp). num_cols=hidden=2048 (16|cols). */
    const Pack16 *in_p  = reinterpret_cast<const Pack16 *>(in);
    Pack16       *out_p = reinterpret_cast<Pack16 *>(out);
    uint32_t col_packs = num_cols / 16;
    uint32_t col, row, k;

    /* Load all weights once: weights[col * kernel_size + k] for col=0..num_cols-1, k=0..kernel_size-1. */
    conv_load_w_col: for (col = 0; col < num_cols; ++col) {
    #pragma HLS loop_tripcount min=2048 max=2048
        conv_load_w_k: for (k = 0; k < kernel_size; ++k) {
        #pragma HLS loop_tripcount min=4 max=4
        #pragma HLS pipeline II=1
            w_loc[col][k] = weights[(size_t)col * kernel_size + k];
        }
    }

    /* Zero the sliding window so the first (kernel_size-1) rows automatically
     * skip out-of-bounds source rows (no conditional needed). */
    conv_init_win_k: for (k = 0; k < kernel_size; ++k) {
    #pragma HLS loop_tripcount min=4 max=4
    #pragma HLS unroll
        conv_init_win_col: for (col = 0; col < num_cols; ++col) {
        #pragma HLS loop_tripcount min=2048 max=2048
        #pragma HLS pipeline II=1
            in_window[k][col] = 0.0f;
        }
    }

    {
        /* Decode (the only mode): pre-load the last (kernel_size-1) prefix rows
         * into window slots 1..kernel_size-1; the row-0 load shifts them into
         * 0..k-2 so the first new token sees {tail0, tail1, tail2, row0}. */
        const Pack16 *tail_p = reinterpret_cast<const Pack16 *>(conv_tail);
        conv_rst_k: for (k = 0; k + 1 < kernel_size; ++k) {
        #pragma HLS loop_tripcount min=3 max=3
            conv_rst_cp: for (uint32_t cp = 0; cp < num_cols / 16; ++cp) {
            #pragma HLS loop_tripcount min=128 max=128
            #pragma HLS pipeline II=1
                Pack16 t = tail_p[(size_t)k * (num_cols / 16) + cp];
                conv_rst_lane: for (int kk = 0; kk < 16; ++kk) {
                #pragma HLS unroll
                    in_window[k + 1][cp * 16 + kk] = t.data[kk];
                }
            }
        }
    }

    /* Streaming conv: per row, do two II=1 phases.
     *   Phase A (load + shift): pull row r from m_axi and shift the window.
     *                           Only the gmem READ channel is touched.
     *   Phase B (compute + write): MAC against w_loc and emit to m_axi.
     *                              Only the gmem WRITE channel is touched.
     *
     * Fusing into one phase forces HLS to schedule a gmem read and write in
     * the same iteration, which it serialises through a single port even
     * though the AR/AW channels are independent — costing II=155 in v2. */
    conv_row: for (row = 0; row < num_rows; ++row) {
    #pragma HLS loop_tripcount min=1 max=2048

        conv_load: for (uint32_t cp = 0; cp < col_packs; ++cp) {
        #pragma HLS loop_tripcount min=128 max=128
        #pragma HLS pipeline II=1
            Pack16 v = in_p[(size_t)row * col_packs + cp];
            conv_load_lane: for (int kk = 0; kk < 16; ++kk) {
            #pragma HLS unroll
                uint32_t c = cp * 16 + kk;
                in_window[0][c] = in_window[1][c];
                in_window[1][c] = in_window[2][c];
                in_window[2][c] = in_window[3][c];
                in_window[3][c] = v.data[kk];
            }
        }

        conv_compute: for (uint32_t cp = 0; cp < col_packs; ++cp) {
        #pragma HLS loop_tripcount min=128 max=128
        #pragma HLS pipeline II=1
            Pack16 o;
            conv_comp_lane: for (int kk = 0; kk < 16; ++kk) {
            #pragma HLS unroll
                uint32_t c = cp * 16 + kk;
                /* in_window[k] holds source row (row - kernel_size + 1 + k). */
                float sum = in_window[0][c] * w_loc[c][0]
                          + in_window[1][c] * w_loc[c][1]
                          + in_window[2][c] * w_loc[c][2]
                          + in_window[3][c] * w_loc[c][3];
                o.data[kk] = gdn_silu(sum);
            }
            out_p[(size_t)row * col_packs + cp] = o;
        }
    }

    {
        /* Decode (the only mode): persist the last (kernel_size-1) input rows;
         * after the final row's shift, window slots 1..kernel_size-1 hold the
         * newest k-1 inputs. */
        Pack16 *tail_p = reinterpret_cast<Pack16 *>(conv_tail);
        conv_sav_k: for (k = 0; k + 1 < kernel_size; ++k) {
        #pragma HLS loop_tripcount min=3 max=3
            conv_sav_cp: for (uint32_t cp = 0; cp < col_packs; ++cp) {
            #pragma HLS loop_tripcount min=128 max=128
            #pragma HLS pipeline II=1
                Pack16 t;
                conv_sav_lane: for (int kk = 0; kk < 16; ++kk) {
                #pragma HLS unroll
                    t.data[kk] = in_window[k + 1][cp * 16 + kk];
                }
                tail_p[(size_t)k * col_packs + cp] = t;
            }
        }
    }
}

/* -----------------------------------------------------------------------
 * Optimized recurrent attention with:
 *   1. Persistent on-chip state in BRAM (eliminates external memory traffic)
 *   2. Fused two-pass pipeline (1 read + 1 read-modify-write vs 4 passes)
 *   3. Column parallelism P_K=16 (16 MACs per cycle on state accesses)
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
static void gdn_recurrent_attention(
    float *attn_out,
    float *recurrent_state,  /* decode: per-layer state at layer_index*8*256*256; prefill: unused */
    float *head_buffer,      /* unused: local buffers used instead */
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
    uint32_t num_tokens,
    uint32_t layer_index
) {
    /* On-chip persistent recurrent state: 8 heads × 256 × 256 FP32 = 2 MB */
    static float state[GDN_HEADS][GDN_DK][GDN_DV];
#pragma HLS bind_storage variable=state type=RAM_2P impl=BRAM
#pragma HLS array_partition variable=state dim=3 cyclic factor=16

    float q_scale = 1.0f / sqrtf((float)GDN_DK);
    uint32_t h, j, i;
    uint32_t token_index;
    /* Layer slice of the HBM-resident decode state (48 MB across 24 layers). */
    size_t st_base = (size_t)layer_index * GDN_HEADS * GDN_DK * GDN_DV;

    {
        /* Decode (the only mode): restore this layer's state from HBM
         * (sequential 512-bit bursts). State was loaded from the GPU export. */
        state_rst_h: for (h = 0; h < GDN_HEADS; ++h) {
        #pragma HLS loop_tripcount min=8 max=8
            state_rst_j: for (j = 0; j < GDN_DK; ++j) {
            #pragma HLS loop_tripcount min=256 max=256
                state_rst_i: for (i = 0; i < GDN_DV; i += GDN_PK) {
                #pragma HLS loop_tripcount min=16 max=16
                #pragma HLS pipeline II=1
                    uint32_t pp;
                    for (pp = 0; pp < GDN_PK; ++pp) {
                    #pragma HLS unroll
                        state[h][j][i + pp] = recurrent_state[
                            st_base + ((size_t)h * GDN_DK + j) * GDN_DV + i + pp];
                    }
                }
            }
        }
    }

    recur_token: for (token_index = 0; token_index < num_tokens; ++token_index) {
    #pragma HLS loop_tripcount min=1 max=2048
        uint32_t head_index;
        recur_head: for (head_index = 0; head_index < GDN_HEADS; ++head_index) {
        #pragma HLS loop_tripcount min=8 max=8

            const float *q_head = q + (size_t)token_index * hidden
                                    + (size_t)head_index * GDN_DK;
            const float *k_head = k + (size_t)token_index * hidden
                                    + (size_t)head_index * GDN_DK;
            const float *v_head = v + (size_t)token_index * hidden
                                    + (size_t)head_index * GDN_DV;
            float *out_head = attn_out + (size_t)token_index * hidden
                                       + (size_t)head_index * GDN_DV;

            /* ---- Local per-token buffers ---- */
            float q_loc[GDN_DK];
            float k_loc[GDN_DK];
            float v_loc[GDN_DV];
            float r_buf[GDN_DV];   /* retrieval result           */
            float o_buf[GDN_DV];   /* partial output              */
            float dv[GDN_DV];      /* delta correction            */
#pragma HLS array_partition variable=r_buf cyclic factor=16
#pragma HLS array_partition variable=o_buf cyclic factor=16
#pragma HLS array_partition variable=dv    cyclic factor=16
#pragma HLS array_partition variable=v_loc cyclic factor=16

            /* ---- Load q, k from DRAM, square into per-element scratch ----
             * Pipelined load loop has no carried dep (each iteration writes a
             * different qsq[j]/ksq[j]). The L2 sums are produced by a fully-
             * unrolled tree reduction in a separate phase. The earlier 8-lane
             * partial accumulator was muxed by HLS into a single register and
             * tracked as a carried dep, holding load_qk at II=2.
             *
             * load_qk itself is still bound by 2 m_axi reads per iter on the
             * shared gmem port (HLS schedules them in 2 cycles), so II=2 is
             * fundamental here without splitting q/k onto separate bundles.
             */
            float qsq_arr[GDN_DK], ksq_arr[GDN_DK];
            #pragma HLS array_partition variable=qsq_arr complete
            #pragma HLS array_partition variable=ksq_arr complete

            load_qk: for (j = 0; j < GDN_DK; ++j) {
            #pragma HLS loop_tripcount min=256 max=256
            #pragma HLS pipeline II=1
                float qj = q_head[j];
                float kj = k_head[j];
                q_loc[j] = qj;
                k_loc[j] = kj;
                qsq_arr[j] = qj * qj;
                ksq_arr[j] = kj * kj;
            }

            float q_sq = gdn_tree_reduce_256(qsq_arr);
            float k_sq = gdn_tree_reduce_256(ksq_arr);

            float q_inv = 1.0f / sqrtf(q_sq + 1e-6f);
            float k_inv = 1.0f / sqrtf(k_sq + 1e-6f);

            /* Normalise q and k in local buffers */
            norm_qk: for (j = 0; j < GDN_DK; ++j) {
            #pragma HLS loop_tripcount min=256 max=256
            #pragma HLS pipeline II=1
                q_loc[j] *= q_inv;
                k_loc[j] *= k_inv;
            }

            /* Load v from DRAM */
            load_v: for (i = 0; i < GDN_DV; ++i) {
            #pragma HLS loop_tripcount min=256 max=256
            #pragma HLS pipeline II=1
                v_loc[i] = v_head[i];
            }

            /* Scalar gates */
            float beta = gdn_sigmoid(
                b[(size_t)token_index * num_heads + head_index]);
            float decay_in = a[(size_t)token_index * num_heads + head_index]
                           + layer_dt_bias[head_index];
            float decay_val = -expf(layer_a_log[head_index])
                            * gdn_softplus(decay_in);
            float g = expf(decay_val);   /* decay factor */

            /* ---- Phase 1: α = q_norm^T · k_norm (scalar) ----
             * Same store-products + tree-reduce pattern as load_qk above.
             * No carried dep -> dot_alpha pipelines at II=1.
             */
            float alpha_prod[GDN_DK];
            #pragma HLS array_partition variable=alpha_prod complete

            dot_alpha: for (j = 0; j < GDN_DK; ++j) {
            #pragma HLS loop_tripcount min=256 max=256
            #pragma HLS pipeline II=1
                alpha_prod[j] = q_loc[j] * k_loc[j];
            }

            float alpha = gdn_tree_reduce_256(alpha_prod);

            /* ---- Phase 2: FUSED READ PASS ----
             * For each column i, accumulate across rows j:
             *   r_buf[i] = Σ_j S[j][i] * k_norm[j]   (retrieval)
             *   o_buf[i] = Σ_j S[j][i] * q_norm[j]   (partial output)
             * With P_K=16 column parallelism at II=1.
             */
            init_ro: for (i = 0; i < GDN_DV; ++i) {
            #pragma HLS loop_tripcount min=256 max=256
            #pragma HLS pipeline II=1
            #pragma HLS unroll factor=16
                r_buf[i] = 0.0f;
                o_buf[i] = 0.0f;
            }

            fused_rd_j: for (j = 0; j < GDN_DK; ++j) {
            #pragma HLS loop_tripcount min=256 max=256
                float kj = k_loc[j];
                float qj = q_loc[j];
                fused_rd_i: for (i = 0; i < GDN_DV; i += GDN_PK) {
                #pragma HLS loop_tripcount min=16 max=16
                #pragma HLS pipeline II=1
                    uint32_t pp;
                    for (pp = 0; pp < GDN_PK; ++pp) {
                    #pragma HLS unroll
                        float s = state[head_index][j][i + pp];
                        r_buf[i + pp] += s * kj;
                        o_buf[i + pp] += s * qj;
                    }
                }
            }

            /* ---- Phase 3: Delta correction + output correction ----
             * Δv[i] = β * (v[i] - r[i])
             * o[i]  = q_scale * (g * ô[i] + α * Δv[i])
             *
             * Compute into on-chip out_loc with P_K=16 column parallelism, then
             * drain to the AXI port in a separate II=1 loop. Without the split,
             * delta_out's 16 simultaneous m_axi stores serialised onto a single
             * gmem port and HLS reported II=16 ("limited memory ports").
             */
            float out_loc[GDN_DV];
            #pragma HLS array_partition variable=out_loc cyclic factor=16

            delta_out: for (i = 0; i < GDN_DV; ++i) {
            #pragma HLS loop_tripcount min=256 max=256
            #pragma HLS pipeline II=1
            #pragma HLS unroll factor=16
                float d = beta * (v_loc[i] - g * r_buf[i]);
                dv[i] = d;
                out_loc[i] = q_scale * (g * o_buf[i] + alpha * d);
            }

            delta_drain: for (i = 0; i < GDN_DV; ++i) {
            #pragma HLS loop_tripcount min=256 max=256
            #pragma HLS pipeline II=1
                out_head[i] = out_loc[i];
            }

            /* ---- Phase 4: FUSED WRITE PASS (state update + decay) ----
             * S[j][i] = g * S[j][i] + k_norm[j] * Δv[i]
             */
            fused_wr_j: for (j = 0; j < GDN_DK; ++j) {
            #pragma HLS loop_tripcount min=256 max=256
                float kj = k_loc[j];
                fused_wr_i: for (i = 0; i < GDN_DV; i += GDN_PK) {
                #pragma HLS loop_tripcount min=16 max=16
                #pragma HLS pipeline II=1
                    uint32_t pp;
                    for (pp = 0; pp < GDN_PK; ++pp) {
                    #pragma HLS unroll
                        state[head_index][j][i + pp] =
                            g * state[head_index][j][i + pp]
                            + kj * dv[i + pp];
                    }
                }
            }
        } /* recur_head */
    } /* recur_token */

    {
        /* Decode (the only mode): persist this layer's updated state to HBM
         * (sequential 512-bit bursts) for the next token's restore. */
        state_sav_h: for (h = 0; h < GDN_HEADS; ++h) {
        #pragma HLS loop_tripcount min=8 max=8
            state_sav_j: for (j = 0; j < GDN_DK; ++j) {
            #pragma HLS loop_tripcount min=256 max=256
                state_sav_i: for (i = 0; i < GDN_DV; i += GDN_PK) {
                #pragma HLS loop_tripcount min=16 max=16
                #pragma HLS pipeline II=1
                    uint32_t pp;
                    for (pp = 0; pp < GDN_PK; ++pp) {
                    #pragma HLS unroll
                        recurrent_state[
                            st_base + ((size_t)h * GDN_DK + j) * GDN_DV + i + pp] =
                                state[h][j][i + pp];
                    }
                }
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
    /* Pack16-widened: attn/gate are read/written 16 lanes (512-bit) per beat by
     * indexing the Pack16 *base* with an integer pack offset. head_dim=256 is a
     * multiple of 16, and (token*num_heads + head)*head_dim is too, so a whole
     * head spans hd_packs=16 aligned Pack16 words. */
    const Pack16 *attn_in  = reinterpret_cast<const Pack16 *>(attn);
    Pack16       *attn_out = reinterpret_cast<Pack16 *>(attn);
    const Pack16 *gate_p   = reinterpret_cast<const Pack16 *>(gate);
    uint32_t hd_packs = head_dim / 16;

    /* Pre-load the per-head norm weight once and reuse for every (token, head). */
    float weight_loc[GDN_DV];
    #pragma HLS array_partition variable=weight_loc cyclic factor=16
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
            #pragma HLS array_partition variable=attn_loc cyclic factor=16
            #pragma HLS array_partition variable=gate_loc cyclic factor=16

            /* Phase 1: load attn (Pack16) into local + accumulate sum of squares */
            double sum = 0.0;
            onorm_sq: for (uint32_t ip = 0; ip < hd_packs; ++ip) {
            #pragma HLS loop_tripcount min=16 max=16
            #pragma HLS pipeline II=1
                Pack16 v = attn_in[base + ip];
                float s = 0.0f;
                onorm_sq_lane: for (int kk = 0; kk < 16; ++kk) {
                #pragma HLS unroll
                    float a = v.data[kk];
                    attn_loc[ip * 16 + kk] = a;
                    s += a * a;
                }
                sum += (double)s;
            }

            /* Phase 2: load gate (Pack16) into local buffer */
            onorm_load_g: for (uint32_t ip = 0; ip < hd_packs; ++ip) {
            #pragma HLS loop_tripcount min=16 max=16
            #pragma HLS pipeline II=1
                Pack16 g = gate_p[base + ip];
                onorm_g_lane: for (int kk = 0; kk < 16; ++kk) {
                #pragma HLS unroll
                    gate_loc[ip * 16 + kk] = g.data[kk];
                }
            }

            float scale = 1.0f / sqrtf((float)(sum / (double)head_dim) + eps);

            /* Phase 3: combine and write back (Pack16) */
            onorm_gate: for (uint32_t ip = 0; ip < hd_packs; ++ip) {
            #pragma HLS loop_tripcount min=16 max=16
            #pragma HLS pipeline II=1
                Pack16 o;
                onorm_gate_lane: for (int kk = 0; kk < 16; ++kk) {
                #pragma HLS unroll
                    uint32_t index = ip * 16 + kk;
                    float normalized = attn_loc[index] * scale * weight_loc[index];
                    float gate_value = gate_loc[index];
                    o.data[kk] = normalized * gate_value * gdn_sigmoid(gate_value);
                }
                attn_out[base + ip] = o;
            }
        }
    }
}

/* SwiGLU in place — `gate[i] = silu(gate[i]) * up[i]`. Vectorised over
 * Pack16 (16 FP32 lanes / 64 bytes) so HLS uses the 512-bit m_axi adapter
 * for one wide read + one wide read + one wide write per iter instead of
 * three narrow accesses per element. count is always a multiple of 16
 * (count = num_tokens × intermediate, intermediate=5632 ⇒ 16 | count). */
static void gdn_swiglu_inplace(float *gate, const float *up, size_t count) {
    Pack16 *gate16 = reinterpret_cast<Pack16 *>(gate);
    const Pack16 *up16 = reinterpret_cast<const Pack16 *>(up);
    const size_t count16 = count >> 4;  /* count / 16 */
    swiglu_loop: for (size_t i = 0; i < count16; ++i) {
    #pragma HLS loop_tripcount min=352 max=720896  /* count16: 5632/16 .. 2048*5632/16 */
    #pragma HLS pipeline II=1
        Pack16 g = gate16[i];
        Pack16 u = up16[i];
        swiglu_lane: for (int j = 0; j < 16; ++j) {
        #pragma HLS unroll
            g.data[j] = gdn_silu(g.data[j]) * u.data[j];
        }
        gate16[i] = g;
    }
}


/* Forward decl: the decode-only GEMV engine (num_rows==1). Same call signature
 * as gdn_matmul_2d so the projection call sites swap 1:1. Defined below next to
 * the IN_DIM_MAX / Pack16 machinery it shares. */
static void gdn_gemv(
    float *out, const float *in, const float *weights, uint32_t w_pack_off,
    uint32_t num_rows, uint32_t in_dim, uint32_t out_dim);

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
    const float *weight_data_mm
) {
    /* Depths match gdn-1.3b-f32.gdnw: hidden=2048 heads=8 head_dim=256
    intermediate=5632 layers=24 conv=4 max_seq_len=2048 vocab=32000 */
    #pragma HLS interface m_axi port=config depth=1 offset=slave
    /* weight_data on its own bundle (same reason as in gdn_attn_forward) —
     * systolic ReadB reads weights, ReadA reads x_norm/mlp_gate/attn;
     * HLS dataflow requires distinct bundles per task.
     *
     * max_widen_bitwidth=512 on every large float* port forces a 512-bit
     * (=16-float) m_axi adapter, so each Pack16 transfer is one wide beat
     * instead of 16 narrow ones. Lifts ReadB from II=16 → II=1 on the
     * weight side and similarly drops the per-element II of swiglu / the
     * residual adds / matmul output stores from ~150 to ~10. */
    #pragma HLS interface m_axi port=weight_data depth=1466343808 offset=slave bundle=mem_weights max_widen_bitwidth=512 max_read_burst_length=64
    /* weight_data_mm aliases the same HBM weight blob but on a DEDICATED bundle
     * read only by the matmul (gdn_matmul_2d, all Pack16). Splitting it off the
     * scalar weight readers (rmsnorm/conv/onorm/embed, which share mem_weights)
     * lets HLS widen the matmul weight reads to 512-bit instead of 32-bit — the
     * scalar co-readers were demoting the shared bundle, capping the weight
     * port at ~388 MB/s (32-bit) on hardware. The host binds the same weight
     * buffer to both ports (read-only alias); hw.cfg maps both to HBM[10:31]. */
    #pragma HLS interface m_axi port=weight_data_mm depth=1466343808 offset=slave bundle=mem_weights_mm max_widen_bitwidth=512 max_read_burst_length=64
    /* Phase B: activations split across distinct AXI bundles -> distinct HBM
     * channels (hw.cfg), so each stage's input-read master and output-write
     * master run concurrently instead of contending on one gmem port (HBM[0]).
     *   gmem_x   = residual stream + norm out + matmul-output staging
     *   gmem_qkv = attention activations (matmul outputs / conv I/O)
     *   gmem_mlp = MLP intermediates
     * Matmul in/out pairs land on different bundles (x_norm->q, attn->tmp_hidden,
     * x_norm->mlp_*, mlp_gate->tmp_hidden), enabling concurrent load/store. */
    #pragma HLS interface m_axi port=x depth=4194304 offset=slave max_widen_bitwidth=512 bundle=gmem_x
    #pragma HLS interface m_axi port=x_norm depth=4194304 offset=slave max_widen_bitwidth=512 bundle=gmem_x
    #pragma HLS interface m_axi port=q depth=4194304 offset=slave max_widen_bitwidth=512 bundle=gmem_qkv
    #pragma HLS interface m_axi port=k depth=4194304 offset=slave max_widen_bitwidth=512 bundle=gmem_qkv
    #pragma HLS interface m_axi port=v depth=4194304 offset=slave max_widen_bitwidth=512 bundle=gmem_qkv
    #pragma HLS interface m_axi port=a depth=16384 offset=slave
    #pragma HLS interface m_axi port=b depth=16384 offset=slave
    #pragma HLS interface m_axi port=gate depth=4194304 offset=slave max_widen_bitwidth=512 bundle=gmem_qkv
    #pragma HLS interface m_axi port=attn depth=4194304 offset=slave max_widen_bitwidth=512 bundle=gmem_qkv
    #pragma HLS interface m_axi port=tmp_hidden depth=4194304 offset=slave max_widen_bitwidth=512 bundle=gmem_x
    #pragma HLS interface m_axi port=mlp_gate depth=11534336 offset=slave max_widen_bitwidth=512 bundle=gmem_mlp
    #pragma HLS interface m_axi port=mlp_up depth=11534336 offset=slave max_widen_bitwidth=512 bundle=gmem_mlp
    /* Decode persistence: recurrent_state holds num_layers × 2 MB = 48 MB of
     * per-layer state (RESTORE/SAVE); head_buffer is repurposed as the per-layer
     * conv tail (24 layers × 3 convs × (conv-1) rows × hidden = ~1.7 MB). */
    #pragma HLS interface m_axi port=recurrent_state depth=12582912 offset=slave max_widen_bitwidth=512
    #pragma HLS interface m_axi port=head_buffer depth=442368 offset=slave max_widen_bitwidth=512
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

        layer_offset += hidden;
        layer_a_log = weight_data + layer_offset;
        layer_offset += num_heads;
        layer_dt_bias = weight_data + layer_offset;
        layer_offset += num_heads;
        layer_q_proj = weight_data_mm + layer_offset;
        layer_offset += (size_t)hidden * hidden;
        layer_k_proj = weight_data_mm + layer_offset;
        layer_offset += (size_t)hidden * hidden;
        layer_v_proj = weight_data_mm + layer_offset;
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
        layer_g_proj = weight_data_mm + layer_offset;
        layer_offset += (size_t)hidden * hidden;
        layer_o_norm = weight_data + layer_offset;
        layer_offset += head_dim;
        layer_o_proj = weight_data_mm + layer_offset;
        layer_offset += (size_t)hidden * hidden;
        layer_mlp_norm = weight_data + layer_offset;
        layer_offset += hidden;
        layer_mlp_gate_proj = weight_data_mm + layer_offset;
        layer_offset += (size_t)intermediate * hidden;
        layer_mlp_up_proj = weight_data_mm + layer_offset;
        layer_offset += (size_t)intermediate * hidden;
        layer_mlp_down_proj = weight_data_mm + layer_offset;

        gdn_rmsnorm_rows(x_norm, x, layer_attn_norm, num_tokens, hidden, config->norm_eps);
        gdn_gemv(q, x_norm, weight_data_mm, (uint32_t)((layer_q_proj - weight_data_mm) / 16), num_tokens, hidden, hidden);
        gdn_gemv(k, x_norm, weight_data_mm, (uint32_t)((layer_k_proj - weight_data_mm) / 16), num_tokens, hidden, hidden);
        gdn_gemv(v, x_norm, weight_data_mm, (uint32_t)((layer_v_proj - weight_data_mm) / 16), num_tokens, hidden, hidden);
        gdn_matmul_tiled(a, x_norm, layer_a_proj, num_tokens, hidden, num_heads);
        gdn_matmul_tiled(b, x_norm, layer_b_proj, num_tokens, hidden, num_heads);
        gdn_gemv(gate, x_norm, weight_data_mm, (uint32_t)((layer_g_proj - weight_data_mm) / 16), num_tokens, hidden, hidden);

        /* Per-(layer, conv) slice of the persistent conv tail in head_buffer:
         * 3 convs/layer × (conv_size-1) rows × hidden floats. */
        size_t tail_stride = (size_t)(config->conv_size - 1) * hidden;
        float *q_tail = head_buffer + ((size_t)layer_index * 3 + 0) * tail_stride;
        float *k_tail = head_buffer + ((size_t)layer_index * 3 + 1) * tail_stride;
        float *v_tail = head_buffer + ((size_t)layer_index * 3 + 2) * tail_stride;

        gdn_depthwise_conv_silu(tmp_hidden, q, layer_q_conv, q_tail, num_tokens, hidden, config->conv_size);
        gdn_pack16_copy(q, tmp_hidden, hidden_count);
        gdn_depthwise_conv_silu(tmp_hidden, k, layer_k_conv, k_tail, num_tokens, hidden, config->conv_size);
        gdn_pack16_copy(k, tmp_hidden, hidden_count);
        gdn_depthwise_conv_silu(tmp_hidden, v, layer_v_conv, v_tail, num_tokens, hidden, config->conv_size);
        gdn_pack16_copy(v, tmp_hidden, hidden_count);

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
            num_tokens,
            layer_index
        );
        gdn_output_norm_and_gate(attn, gate, layer_o_norm, num_tokens, num_heads, head_dim, config->norm_eps);
        gdn_gemv(tmp_hidden, attn, weight_data_mm, (uint32_t)((layer_o_proj - weight_data_mm) / 16), num_tokens, hidden, hidden);
        gdn_pack16_add_inplace(x, tmp_hidden, hidden_count);

        gdn_rmsnorm_rows(x_norm, x, layer_mlp_norm, num_tokens, hidden, config->norm_eps);
        gdn_gemv(mlp_gate, x_norm, weight_data_mm, (uint32_t)((layer_mlp_gate_proj - weight_data_mm) / 16), num_tokens, hidden, intermediate);
        gdn_gemv(mlp_up, x_norm, weight_data_mm, (uint32_t)((layer_mlp_up_proj - weight_data_mm) / 16), num_tokens, hidden, intermediate);
        gdn_swiglu_inplace(mlp_gate, mlp_up, mlp_count);
        gdn_gemv(tmp_hidden, mlp_gate, weight_data_mm, (uint32_t)((layer_mlp_down_proj - weight_data_mm) / 16), num_tokens, intermediate, hidden);
        gdn_pack16_add_inplace(x, tmp_hidden, hidden_count);
    }

    gdn_rmsnorm_rows(x_norm, x, final_norm, num_tokens, hidden, config->norm_eps);
    return 0;
}

/* Decode-only host entry: forward exactly one token against the persistent
 * per-layer recurrent + conv state held in the run-state buffers (loaded from
 * the GPU .gdnstate export). gdn_forward is decode-only — it restores each
 * layer's state at the start and saves the update at the end. */
int gdn_decode_step_host(const GDNModel *model, GDNRunState *state, const int32_t *token) {
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
        token,
        1u,
        model->weight_data  /* weight_data_mm: same blob, dedicated 512-bit AXI bundle on HW */
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


/* =======================================================================
 * gdn_gemv — decode-only matrix-vector engine (num_rows == 1).
 *
 * Decode is a GEMV: out[o] = sum_k in[k] * W[o][k], one token, each weight read
 * exactly once. The design follows the SOTA bandwidth-bound FPGA-HBM decode
 * engines (FlightLLM / DFX / "Pushing to the Limit of Memory Bandwidth"):
 *
 *   1. ACTIVATION-STATIONARY: the single activation vector `in` (<= 5632 fp32 =
 *      22 KB) is loaded once into on-chip `a_loc` and reused for every output;
 *      weights are STREAMED from HBM once and never cached. (The opposite of the
 *      prefill weight-stationary GEMM, which reused each weight across 256 rows.)
 *   2. CONTIGUOUS WEIGHT STREAM: weights are row-major [out_dim][in_dim], so one
 *      output's row W[o][:] is contiguous. Scanning ONE output at a time keeps
 *      the 512-bit weight port reading one Pack16 (16 fp32) beat/cycle at II=1 —
 *      a clean burst. (Interleaving 16 rows would scatter the port to II=16.)
 *   3. ADDER-TREE + PARTIAL BANKS: each beat does 16 multiplies reduced by a
 *      combinational tree; the running sum rotates across GEMV_PARTIAL banks to
 *      hide FP32 fadd latency so the k-loop holds II=1.
 *   4. Pack16 OUTPUT: 16 dot-products are buffered and written as one 512-bit
 *      beat, matching the activation layout (out_dim % 16 == 0 for all calls).
 *
 * Per-token cost is then weight_bytes / port_bandwidth (one 512-bit master at
 * 100 MHz = 6.4 GB/s -> ~1 s for the 5.6 GB blob): the single-port GEMV floor,
 * vs the 2.56 s the GEMM datapath spent at num_rows=1 (255/256 of its array
 * idle). Widening to N HBM weight readers scales this toward HBM aggregate.
 * ======================================================================= */
#define GEMV_PARTIAL 8       /* power of two; >= FP32 fadd latency in cycles */
#define IN_DIM_MAX   5632    /* max in_dim (intermediate=5632) — sizes a_loc */

static void gdn_gemv(
    float *out, const float *in, const float *weights, uint32_t w_pack_off,
    uint32_t num_rows, uint32_t in_dim, uint32_t out_dim
) {
    #pragma HLS inline off
    (void)num_rows;  /* decode GEMV: always the single token (row 0) */

    /* Resident activation vector; cyclic/16 so the Pack16 load and the 16-lane
     * MAC both hit distinct banks (II=1). */
    static float a_loc[IN_DIM_MAX];
    #pragma HLS array_partition variable=a_loc cyclic factor=16

    const Pack16 *in_p  = reinterpret_cast<const Pack16 *>(in);
    const Pack16 *w_p   = reinterpret_cast<const Pack16 *>(weights);
    Pack16       *out_p = reinterpret_cast<Pack16 *>(out);
    uint32_t k_packs = in_dim / 16;

    /* ---- Load the activation vector once (512-bit beats). ---- */
    gemv_load_a: for (uint32_t kp = 0; kp < k_packs; ++kp) {
    #pragma HLS loop_tripcount min=128 max=352
    #pragma HLS pipeline II=1
        Pack16 v = in_p[kp];
        gemv_la_un: for (int kk = 0; kk < 16; ++kk) {
        #pragma HLS unroll
            a_loc[kp * 16 + kk] = v.data[kk];
        }
    }

    /* ---- Stream weights: one output row at a time (contiguous burst), MAC
     * against the resident vector, buffer 16 results, write one Pack16. ---- */
    Pack16 out_buf;
    gemv_o: for (uint32_t o = 0; o < out_dim; ++o) {
    #pragma HLS loop_tripcount min=8 max=5632
        float part[GEMV_PARTIAL];
        #pragma HLS array_partition variable=part complete
        gemv_init: for (int p = 0; p < GEMV_PARTIAL; ++p) {
        #pragma HLS unroll
            part[p] = 0.0f;
        }
        size_t row_base = (size_t)w_pack_off + (size_t)o * k_packs;
        gemv_k: for (uint32_t kp = 0; kp < k_packs; ++kp) {
        #pragma HLS loop_tripcount min=128 max=352
        #pragma HLS pipeline II=1
            Pack16 w = w_p[row_base + kp];
            float lane = 0.0f;
            gemv_mac: for (int kk = 0; kk < 16; ++kk) {
            #pragma HLS unroll
                lane += w.data[kk] * a_loc[kp * 16 + kk];
            }
            part[kp & (GEMV_PARTIAL - 1)] += lane;
        }
        /* reduce GEMV_PARTIAL banks (balanced tree) */
        float s0 = part[0] + part[1];
        float s1 = part[2] + part[3];
        float s2 = part[4] + part[5];
        float s3 = part[6] + part[7];
        out_buf.data[o & 15] = (s0 + s1) + (s2 + s3);
        if ((o & 15) == 15) {
            out_p[o >> 4] = out_buf;
        }
    }
}

