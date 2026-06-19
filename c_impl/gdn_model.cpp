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
/* GEMV_CHANNELS lives in gdn_model.h (shared by the kernel and the host shard
 * builder / run-state); do not redefine it here. */

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

/* ---- Compact weight shards for the multi-channel decode GEMV (Stage 2) ----
 * Each gemv projection's output rows are split into GEMV_CHANNELS stripes; shard
 * c holds stripe c of every projection, packed per layer in the order
 * q,k,v,gate,o,mlp_gate,mlp_up,mlp_down — exactly the order gdn_forward threads
 * its compact shard offset (soff). Total across all shards = one copy of the
 * projection weights (no replication), so the parallel 512-bit readers fit the
 * same HBM budget as the old single weight_data_mm copy. Host-only (memcpy). */
size_t gdn_weight_shard_floats(const GDNWeightHeader *config) {
    size_t H = config->hidden_size, I = config->intermediate_size, V = config->vocab_size;
    size_t per_layer = 5 * (H / GEMV_CHANNELS) * H     /* q,k,v,gate,o */
                     + 2 * (I / GEMV_CHANNELS) * H     /* mlp_gate, mlp_up */
                     +     (H / GEMV_CHANNELS) * I;     /* mlp_down */
    /* + lm_head [V,H], appended once after all layers so the decode kernel can
     * emit logits (and argmax) on-chip. V % GEMV_CHANNELS == 0 (32000/8). */
    return (size_t)config->num_layers * per_layer + (V / GEMV_CHANNELS) * H;
}

void gdn_build_weight_shards(const float *wd, const GDNWeightHeader *config,
                             float *const shards[]) {
    size_t H = config->hidden_size, I = config->intermediate_size;
    size_t nh = config->num_heads, hd = config->head_dim, cs = config->conv_size;
    size_t soff = 0;  /* running float offset into each shard */
    uint32_t L;
    for (L = 0; L < config->num_layers; ++L) {
        size_t base = gdn_layer_weight_offset(config, L);
        size_t q  = base + H + 2 * nh;                       /* past attn_norm,a_log,dt_bias */
        size_t k  = q + H * H;
        size_t v  = k + H * H;
        size_t g  = v + H * H + 2 * nh * H + 3 * H * cs;     /* past a/b proj + 3 convs */
        size_t o  = g + H * H + hd;                          /* past g_proj + o_norm */
        size_t mg = o + H * H + H;                           /* past o_proj + mlp_norm */
        size_t mu = mg + I * H;
        size_t md = mu + I * H;
        size_t poff[8] = { q, k, v, g, o, mg, mu, md };
        size_t pout[8] = { H, H, H, H, H, I, I, H };
        size_t pin [8] = { H, H, H, H, H, H, H, I };
        int p;
        for (p = 0; p < 8; ++p) {
            /* output rows split into GEMV_CHANNELS stripes; stripe c = rows
             * [c*out/N,(c+1)*out/N) → floats [poff + c*s, ...) → shards[c]. */
            size_t s = (pout[p] / GEMV_CHANNELS) * pin[p];   /* one stripe (floats) */
            int c;
            for (c = 0; c < GEMV_CHANNELS; ++c)
                memcpy(shards[c] + soff, wd + poff[p] + (size_t)c * s, s * sizeof(float));
            soff += s;
        }
    }
    /* lm_head [V,H] (global): split its rows into GEMV_CHANNELS stripes appended
     * after every layer's projections — the order gdn_forward threads for the
     * final logits gemv. lm_head sits right after final_norm (H floats) in the blob. */
    {
        size_t V = config->vocab_size;
        size_t lmh = gdn_final_norm_offset(config) + H;   /* past final_norm */
        size_t s = (V / GEMV_CHANNELS) * H;               /* one stripe (floats) */
        int c;
        for (c = 0; c < GEMV_CHANNELS; ++c)
            memcpy(shards[c] + soff, wd + lmh + (size_t)c * s, s * sizeof(float));
        soff += s;
    }
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

    /* Stage 2: build the GEMV_CHANNELS compact weight shards (split the gemv
     * projection weights by output stripe) the decode datapath reads in
     * parallel. Same total size as one weight copy — no replication. */
    {
        size_t shard_floats = gdn_weight_shard_floats(&model->config);
        int c;
        for (c = 0; c < GEMV_CHANNELS; ++c)
            if (gdn_alloc_run_buffer(&state->weight_shards[c], shard_floats) != 0) return -1;
        gdn_build_weight_shards(model->weight_data, &model->config, state->weight_shards);
    }
    /* lm_head gemv scratch (decode writes logits here, argmaxes to x_norm[0]). */
    if (gdn_alloc_run_buffer(&state->logits, model->config.vocab_size) != 0) return -1;

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
    {
        int c;
        for (c = 0; c < GEMV_CHANNELS; ++c) free(state->weight_shards[c]);
    }
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


/* Compile-time bounds for gdn_gemv_tiny's on-chip buffers (the a/b gate
 * projections: out_dim = num_heads = 8, in_dim = hidden = 2048). */
#define GDN_GEMV_TINY_OUT_MAX 8
#define GDN_GEMV_TINY_IN_MAX  2048

static void gdn_gemv_tiny(
    float *out,
    const float *in,
    const float *weights,
    uint32_t num_rows,
    uint32_t in_dim,
    uint32_t out_dim
) {
    /* Decode-shape GEMV for the tiny a/b gate projections (num_rows=1,
     * in_dim=hidden=2048, out_dim=num_heads=8). Three II=1 steps:
     *   1. load the single activation row into resident a_loc (read once);
     *   2. preload all out_dim weight rows to BRAM as one CONTIGUOUS burst (a/b are
     *      [out_dim][in_dim] row-major) — avoids the per-(c,kc) strided HBM reads;
     *   3. one k-pass computing all out_dim outputs in PARALLEL, each with its own
     *      accumulator + the SAME balanced-tree-per-16-chunk sequential reduction.
     * Bit-exact to the prior per-output reduction (each acc[c] keeps the chunk
     * order); removes the 8x per-output pipeline restart + redundant activation
     * reads that made the prior form ~0.18 ms/call. */
    const Pack16 *in_p = reinterpret_cast<const Pack16 *>(in);
    const Pack16 *w_p  = reinterpret_cast<const Pack16 *>(weights);
    uint32_t k_packs = in_dim / 16;   /* 128 for in_dim=2048 */
    uint32_t c, kc, i;
    (void)num_rows;  /* decode: always the single token (row 0) */

    /* (1) resident activation — read the token's in[] once, reused by every output */
    float a_loc[GDN_GEMV_TINY_IN_MAX];
    #pragma HLS array_partition variable=a_loc cyclic factor=16
    gvt_la: for (kc = 0; kc < k_packs; ++kc) {
    #pragma HLS loop_tripcount min=128 max=128
    #pragma HLS pipeline II=1
        Pack16 a = in_p[kc];
        gvt_la_i: for (i = 0; i < 16; ++i) {
        #pragma HLS unroll
            a_loc[kc * 16 + i] = a.data[i];
        }
    }

    /* (2) preload weights to BRAM — one contiguous burst over [out_dim][in_dim] */
    float w_loc[GDN_GEMV_TINY_OUT_MAX][GDN_GEMV_TINY_IN_MAX];
    #pragma HLS array_partition variable=w_loc dim=1 complete
    #pragma HLS array_partition variable=w_loc dim=2 cyclic factor=16
    gvt_lw_c: for (c = 0; c < out_dim; ++c) {
    #pragma HLS loop_tripcount min=8 max=8
        gvt_lw_kc: for (kc = 0; kc < k_packs; ++kc) {
        #pragma HLS loop_tripcount min=128 max=128
        #pragma HLS pipeline II=1
            Pack16 w = w_p[(size_t)c * k_packs + kc];
            gvt_lw_i: for (i = 0; i < 16; ++i) {
            #pragma HLS unroll
                w_loc[c][kc * 16 + i] = w.data[i];
            }
        }
    }

    /* (3) one k-pass, all outputs in parallel; per-output sequential reduction */
    float acc[GDN_GEMV_TINY_OUT_MAX];
    #pragma HLS array_partition variable=acc complete
    gvt_init: for (c = 0; c < out_dim; ++c) {
    #pragma HLS unroll
        acc[c] = 0.0f;
    }
    gvt_k: for (kc = 0; kc < k_packs; ++kc) {
    #pragma HLS loop_tripcount min=128 max=128
    #pragma HLS pipeline II=1
        gvt_c: for (c = 0; c < out_dim; ++c) {
        #pragma HLS unroll
            float p[16];
            #pragma HLS array_partition variable=p complete
            gvt_mul: for (i = 0; i < 16; ++i) {
            #pragma HLS unroll
                p[i] = a_loc[kc * 16 + i] * w_loc[c][kc * 16 + i];
            }
            /* Balanced 4-level adder tree (same per-chunk reduction as before). */
            float s2_0 = p[0]  + p[1],  s2_1 = p[2]  + p[3];
            float s2_2 = p[4]  + p[5],  s2_3 = p[6]  + p[7];
            float s2_4 = p[8]  + p[9],  s2_5 = p[10] + p[11];
            float s2_6 = p[12] + p[13], s2_7 = p[14] + p[15];
            float s4_0 = s2_0 + s2_1, s4_1 = s2_2 + s2_3;
            float s4_2 = s2_4 + s2_5, s4_3 = s2_6 + s2_7;
            float s8_0 = s4_0 + s4_1, s8_1 = s4_2 + s4_3;
            acc[c] += s8_0 + s8_1;
        }
    }
    gvt_st: for (c = 0; c < out_dim; ++c) {
    #pragma HLS unroll
        out[c] = acc[c];
    }
}

/* No dispatch wrapper — gdn_forward calls gdn_gemv directly for the large
 * decode projections (defined below) and gdn_gemv_tiny for the small
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

    /* Load all conv weights once. weights[col*ks + k] is contiguous, so read it as
     * 512-bit Pack16 bursts (16 floats/beat) instead of one 32-bit scalar/cycle:
     * the scalar form was HBM-latency-bound at ~1.6 ms/call on-card (65% of conv,
     * measured) — the Pack16 burst is the pattern gemv_tiny uses. For conv_size=4
     * (always, in decode) each beat carries 4 cols x 4 taps; w_loc dim1 is cyclic16
     * and dim2 complete, so the 16 lane writes hit distinct banks at II=1. Bit-exact:
     * identical w_loc. (Scalar fallback kept for kernel_size != 4.) */
    if (kernel_size == 4) {
        const Pack16 *w_src = reinterpret_cast<const Pack16 *>(weights);
        conv_load_w_b: for (uint32_t cb = 0; cb < num_cols / 4; ++cb) {
        #pragma HLS loop_tripcount min=512 max=512
        #pragma HLS pipeline II=1
            Pack16 wp = w_src[cb];
            conv_load_w_j: for (int j = 0; j < 4; ++j) {
            #pragma HLS unroll
                conv_load_w_kk: for (int kk = 0; kk < 4; ++kk) {
                #pragma HLS unroll
                    w_loc[cb * 4 + (uint32_t)j][kk] = wp.data[j * 4 + kk];
                }
            }
        }
    } else {
        conv_load_w_col: for (col = 0; col < num_cols; ++col) {
        #pragma HLS loop_tripcount min=2048 max=2048
            conv_load_w_k: for (k = 0; k < kernel_size; ++k) {
            #pragma HLS loop_tripcount min=4 max=4
            #pragma HLS pipeline II=1
                w_loc[col][k] = weights[(size_t)col * kernel_size + k];
            }
        }
    }

    /* (Decode) The sliding-window zero-init was DROPPED — it is redundant: the
     * tail restore below fills window slots 1..kernel_size-1, and the row-0 load
     * shifts them down (slot0<-slot1, ...), so all kernel_size slots are defined
     * before the compute reads them; the pre-restore values are never read.
     * Removing the 4x2048 zeroing saved ~0.08 ms/call (~6 ms/token). Bit-exact. */

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


/* Forward decl: the decode-only GEMV engine (num_rows==1) with GEMV_CHANNELS
 * parallel HBM weight readers; weights0..weights3 are the compact shards, each on
 * its own m_axi master. Defined below next to the IN_DIM_MAX / Pack16 machinery. */
static void gdn_gemv(
    float *out, const float *in,
    const float *weights0, const float *weights1,
    const float *weights2, const float *weights3,
    const float *weights4, const float *weights5,
    const float *weights6, const float *weights7, uint32_t w_pack_off,
    uint32_t num_rows, uint32_t in_dim, uint32_t out_dim);

/* On-chip greedy argmax over the [vocab] logits (first-max tie-break); writes the
 * token id as a float into out_token_f[0]. Defined below near gdn_gemv. */
static void gdn_argmax(float *out_token_f, const float *logits, uint32_t vocab);

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
    const float *weight_data_mm,   /* gemv shard 0 reader */
    const float *weight_data_mm2,  /* gemv shard 1 reader */
    const float *weight_data_mm3,  /* gemv shard 2 reader */
    const float *weight_data_mm4,  /* gemv shard 3 reader */
    const float *weight_data_mm5,  /* gemv shard 4 reader (Stage 2c: N=8) */
    const float *weight_data_mm6,  /* gemv shard 5 reader */
    const float *weight_data_mm7,  /* gemv shard 6 reader */
    const float *weight_data_mm8,  /* gemv shard 7 reader */
    float *logits                  /* [vocab] scratch: lm_head gemv output (argmax → x_norm[0]) */
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
    /* num_read_outstanding bumped (default 16): the conv weight load (load_w) and
     * the a/b gemv_tiny both read this scalar master and were read-latency-bound;
     * more outstanding reads keep the 64-beat bursts in flight. Interface-only. */
    #pragma HLS interface m_axi port=weight_data depth=1466343808 offset=slave bundle=mem_weights max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=64
    /* weight_data_mm aliases the same HBM weight blob but on a DEDICATED bundle
     * read only by the matmul (gdn_matmul_2d, all Pack16). Splitting it off the
     * scalar weight readers (rmsnorm/conv/onorm/embed, which share mem_weights)
     * lets HLS widen the matmul weight reads to 512-bit instead of 32-bit — the
     * scalar co-readers were demoting the shared bundle, capping the weight
     * port at ~388 MB/s (32-bit) on hardware. The host binds the same weight
     * buffer to both ports (read-only alias); hw.cfg maps both to HBM[10:31]. */
    #pragma HLS interface m_axi port=weight_data_mm depth=1466343808 offset=slave bundle=mem_weights_mm max_widen_bitwidth=512 max_read_burst_length=64
    /* 2nd weight reader on its OWN bundle/master so the HBM crossbar serves it
     * concurrently with mem_weights_mm (Stage 2: ~2× weight read bandwidth).
     * Aliases the same blob; hw.cfg maps mem_weights_mm2 across HBM[10:31]. */
    #pragma HLS interface m_axi port=weight_data_mm2 depth=1466343808 offset=slave bundle=mem_weights_mm2 max_widen_bitwidth=512 max_read_burst_length=64
    /* Stage 2b (N=4): shards 2 and 3 on their OWN bundles/masters so the HBM
     * crossbar serves all four shard streams concurrently (~4× weight read
     * bandwidth). hw.cfg maps mem_weights_mm3/mm4 to DISJOINT HBM bank groups. */
    #pragma HLS interface m_axi port=weight_data_mm3 depth=1466343808 offset=slave bundle=mem_weights_mm3 max_widen_bitwidth=512 max_read_burst_length=64
    #pragma HLS interface m_axi port=weight_data_mm4 depth=1466343808 offset=slave bundle=mem_weights_mm4 max_widen_bitwidth=512 max_read_burst_length=64
    /* Stage 2c (N=8): shards 4-7 on their OWN bundles/masters — eight disjoint
     * weight readers total (~8× weight read bandwidth, sub-linear in practice).
     * hw.cfg maps mem_weights_mm5..mm8 to DISJOINT HBM bank groups. */
    #pragma HLS interface m_axi port=weight_data_mm5 depth=1466343808 offset=slave bundle=mem_weights_mm5 max_widen_bitwidth=512 max_read_burst_length=64
    #pragma HLS interface m_axi port=weight_data_mm6 depth=1466343808 offset=slave bundle=mem_weights_mm6 max_widen_bitwidth=512 max_read_burst_length=64
    #pragma HLS interface m_axi port=weight_data_mm7 depth=1466343808 offset=slave bundle=mem_weights_mm7 max_widen_bitwidth=512 max_read_burst_length=64
    #pragma HLS interface m_axi port=weight_data_mm8 depth=1466343808 offset=slave bundle=mem_weights_mm8 max_widen_bitwidth=512 max_read_burst_length=64
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
    /* lm_head logits scratch shares the gmem_mlp master (written once per token by
     * the final gemv, then read by the on-chip argmax) — no extra HBM master. */
    #pragma HLS interface m_axi port=logits depth=32000 offset=slave max_widen_bitwidth=512 bundle=gmem_mlp
    /* Decode persistence: recurrent_state holds num_layers × 2 MB = 48 MB of
     * per-layer state (RESTORE/SAVE); head_buffer is repurposed as the per-layer
     * conv tail (24 layers × 3 convs × (conv-1) rows × hidden = ~1.7 MB). */
    #pragma HLS interface m_axi port=recurrent_state depth=12582912 offset=slave max_widen_bitwidth=512
    /* head_buffer holds the per-(layer,conv) tail; the conv-tail SAVE writes ~384
     * Pack16 beats/call. With default m_axi write params those writes were
     * write-response-latency bound (~170 cyc/beat, 0.65 ms/call = 69% of conv),
     * while the same-structure tail RESTORE (a read) was 5x faster. Widen the write
     * burst and raise outstanding writes so the B-response latency is hidden (reads
     * bumped too, to speed restore). Interface-only — bit-exact. */
    #pragma HLS interface m_axi port=head_buffer depth=442368 offset=slave max_widen_bitwidth=512 num_write_outstanding=64 num_read_outstanding=64 max_write_burst_length=64 max_read_burst_length=64
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

    /* Compact-shard geometry (Pack16 units): each gemv projection's output stripe
     * (out_dim/GEMV_CHANNELS rows) occupies stripe_packs in every shard, packed
     * per layer in the order q,k,v,gate,o,mlp_gate,mlp_up,mlp_down — matching
     * gdn_build_weight_shards. shard0/shard1 are passed as weight_data_mm/_mm2. */
    size_t shard_hh = (size_t)(hidden / GEMV_CHANNELS) * (hidden / 16);
    size_t shard_ih = (size_t)(intermediate / GEMV_CHANNELS) * (hidden / 16);
    size_t shard_di = (size_t)(hidden / GEMV_CHANNELS) * (intermediate / 16);
    size_t shard_per_layer = 5 * shard_hh + 2 * shard_ih + shard_di;

    gdn_embed_tokens(x, embeddings, tokens, num_tokens, hidden, config->vocab_size);
    layer_loop: for (layer_index = 0; layer_index < config->num_layers; ++layer_index) {
    #pragma HLS loop_tripcount min=24 max=24  /* num_layers=24 */
        size_t layer_offset = gdn_layer_weight_offset(config, layer_index);
        const float *layer_attn_norm = weight_data + layer_offset;
        const float *layer_a_log;
        const float *layer_dt_bias;
        const float *layer_a_proj;
        const float *layer_b_proj;
        const float *layer_q_conv;
        const float *layer_k_conv;
        const float *layer_v_conv;
        const float *layer_o_norm;
        const float *layer_mlp_norm;

        /* Scalar weights stay in weight_data (full blob); the gemv projection
         * weights live in the compact shards. layer_offset still advances past
         * the projections so the following scalar offsets stay correct. */
        layer_offset += hidden;                          /* past attn_norm */
        layer_a_log = weight_data + layer_offset;
        layer_offset += num_heads;
        layer_dt_bias = weight_data + layer_offset;
        layer_offset += num_heads;
        layer_offset += (size_t)hidden * hidden;         /* past q_proj (shards) */
        layer_offset += (size_t)hidden * hidden;         /* past k_proj */
        layer_offset += (size_t)hidden * hidden;         /* past v_proj */
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
        layer_offset += (size_t)hidden * hidden;         /* past g_proj (shards) */
        layer_o_norm = weight_data + layer_offset;
        layer_offset += head_dim;
        layer_offset += (size_t)hidden * hidden;         /* past o_proj (shards) */
        layer_mlp_norm = weight_data + layer_offset;
        /* mlp_gate/up/down projection weights live in the shards. */

        /* Running compact-shard offset (Pack16); order q,k,v,gate,o,mlp_gate,
         * mlp_up,mlp_down — matches gdn_build_weight_shards. */
        size_t soff = (size_t)layer_index * shard_per_layer;

        gdn_rmsnorm_rows(x_norm, x, layer_attn_norm, num_tokens, hidden, config->norm_eps);
        gdn_gemv(q, x_norm, weight_data_mm, weight_data_mm2, weight_data_mm3, weight_data_mm4, weight_data_mm5, weight_data_mm6, weight_data_mm7, weight_data_mm8, (uint32_t)soff, num_tokens, hidden, hidden);    soff += shard_hh;
        gdn_gemv(k, x_norm, weight_data_mm, weight_data_mm2, weight_data_mm3, weight_data_mm4, weight_data_mm5, weight_data_mm6, weight_data_mm7, weight_data_mm8, (uint32_t)soff, num_tokens, hidden, hidden);    soff += shard_hh;
        gdn_gemv(v, x_norm, weight_data_mm, weight_data_mm2, weight_data_mm3, weight_data_mm4, weight_data_mm5, weight_data_mm6, weight_data_mm7, weight_data_mm8, (uint32_t)soff, num_tokens, hidden, hidden);    soff += shard_hh;
        gdn_gemv_tiny(a, x_norm, layer_a_proj, num_tokens, hidden, num_heads);
        gdn_gemv_tiny(b, x_norm, layer_b_proj, num_tokens, hidden, num_heads);
        gdn_gemv(gate, x_norm, weight_data_mm, weight_data_mm2, weight_data_mm3, weight_data_mm4, weight_data_mm5, weight_data_mm6, weight_data_mm7, weight_data_mm8, (uint32_t)soff, num_tokens, hidden, hidden); soff += shard_hh;

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
        gdn_gemv(tmp_hidden, attn, weight_data_mm, weight_data_mm2, weight_data_mm3, weight_data_mm4, weight_data_mm5, weight_data_mm6, weight_data_mm7, weight_data_mm8, (uint32_t)soff, num_tokens, hidden, hidden); soff += shard_hh;
        gdn_pack16_add_inplace(x, tmp_hidden, hidden_count);

        gdn_rmsnorm_rows(x_norm, x, layer_mlp_norm, num_tokens, hidden, config->norm_eps);
        gdn_gemv(mlp_gate, x_norm, weight_data_mm, weight_data_mm2, weight_data_mm3, weight_data_mm4, weight_data_mm5, weight_data_mm6, weight_data_mm7, weight_data_mm8, (uint32_t)soff, num_tokens, hidden, intermediate); soff += shard_ih;
        gdn_gemv(mlp_up, x_norm, weight_data_mm, weight_data_mm2, weight_data_mm3, weight_data_mm4, weight_data_mm5, weight_data_mm6, weight_data_mm7, weight_data_mm8, (uint32_t)soff, num_tokens, hidden, intermediate);   soff += shard_ih;
        gdn_swiglu_inplace(mlp_gate, mlp_up, mlp_count);
        gdn_gemv(tmp_hidden, mlp_gate, weight_data_mm, weight_data_mm2, weight_data_mm3, weight_data_mm4, weight_data_mm5, weight_data_mm6, weight_data_mm7, weight_data_mm8, (uint32_t)soff, num_tokens, intermediate, hidden);
        gdn_pack16_add_inplace(x, tmp_hidden, hidden_count);
    }

    gdn_rmsnorm_rows(x_norm, x, final_norm, num_tokens, hidden, config->norm_eps);
    /* lm_head on-chip: logits[vocab] = x_norm @ lm_head via the same 8-reader
     * sharded gemv (lm_head's stripe is appended after every layer in each shard,
     * so its offset is num_layers*shard_per_layer). Then greedy argmax → next
     * token id, written into x_norm[0] — x_norm is done as an activation, so it
     * doubles as the 1-int output (no extra port). This makes the kernel emit a
     * complete decode step (token in → token out), matching what the GPU times. */
    {
        size_t lm_soff = (size_t)config->num_layers * shard_per_layer;
        gdn_gemv(logits, x_norm,
                 weight_data_mm, weight_data_mm2, weight_data_mm3, weight_data_mm4,
                 weight_data_mm5, weight_data_mm6, weight_data_mm7, weight_data_mm8,
                 (uint32_t)lm_soff, num_tokens, hidden, config->vocab_size);
        gdn_argmax(x_norm, logits, config->vocab_size);
    }
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
        state->weight_shards[0], state->weight_shards[1],  /* gemv shards 0,1 */
        state->weight_shards[2], state->weight_shards[3],  /* gemv shards 2,3 */
        state->weight_shards[4], state->weight_shards[5],  /* gemv shards 4,5 */
        state->weight_shards[6], state->weight_shards[7],  /* gemv shards 6,7 */
        state->logits                                      /* lm_head scratch; token → x_norm[0] */
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
 *   2. DECOUPLED READER -> MAC (HLS dataflow): a dedicated gemv_read process
 *      bursts the whole projection's weights (out_dim*k_packs beats, row-major
 *      back-to-back) into a FIFO; gemv_compute drains it and MACs in parallel.
 *      One contiguous 512-bit burst, never broken between output rows — the fix
 *      for the read+MAC-coupled version that sustained only 45% of the port.
 *      (This is FlightLLM's "streaming" weight transfer.)
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
#define GEMV_PARTIAL  8      /* power of two; >= FP32 fadd latency in cycles */
#define IN_DIM_MAX    5632   /* max in_dim (intermediate=5632) — sizes a_loc */
/* GEMV_CHANNELS lives in gdn_model.h (shared by the kernel and the host). */

/* Producer (one HBM channel): stream this channel's output stripe as ONE
 * contiguous burst (base .. base+n_packs, a single monotonic sweep). The N
 * readers run on distinct m_axi weight masters, so the HBM crossbar serves
 * their bursts concurrently → ~N× the single-port 5.30 GB/s. */
static void gemv_read_ch(const Pack16 *w_p, size_t base, uint32_t n_packs,
                         hls::stream<Pack16> &wf) {
    gemv_rd: for (uint32_t i = 0; i < n_packs; ++i) {
    #pragma HLS loop_tripcount min=131072 max=360448
    #pragma HLS pipeline II=1
        wf.write(w_p[base + i]);
    }
}

/* ---- Per-channel processing elements (Stage-2 routing fix) -------------
 * The monolithic 2-wide consumer (one module doing BOTH channels' 16-wide
 * fp32 reductions) packed a single region densely enough to fail routing
 * twice: partially-conflicted full_dsp fp-adder nets, congestion level 7.
 * The fix is one PE per channel — each gemv_pe_mac is its own dataflow
 * process with exactly the Stage-1 single-port density (which routed), so
 * Vivado places the N PEs in separate regions instead of one hot-spot. A
 * single MAC-free gemv_collect is the only writer to `out` (no two-writers-
 * to-one-port hazard). Bit-exactness holds: same resident a_loc, same beat
 * order, same partial-bank adder-tree reduction as the monolithic version. */

/* Activation fan-out: read the resident activation once (single reader of the
 * `in` m_axi port → clean dataflow) and broadcast each beat to all N PEs, which
 * keep private a_loc copies for parallel random access. */
static void gemv_pe_bcast(const Pack16 *in_p, hls::stream<Pack16> af[GEMV_CHANNELS],
                          uint32_t k_packs) {
    gemv_bc: for (uint32_t kp = 0; kp < k_packs; ++kp) {
    #pragma HLS loop_tripcount min=128 max=352
    #pragma HLS pipeline II=1
        Pack16 v = in_p[kp];
        gemv_bc_un: for (int c = 0; c < GEMV_CHANNELS; ++c) {
        #pragma HLS unroll
            af[c].write(v);
        }
    }
}

/* One PE (one channel): drain the broadcast activation into a private a_loc
 * (cyclic/16 → 16 parallel lanes), then MAC this channel's `stripe` output rows,
 * consuming its weight FIFO at II=1. GEMV_PARTIAL rotating banks + an adder tree
 * hide FP32 fadd latency; each dot product is emitted as a scalar to `of`. One
 * PE == the Stage-1 single-port datapath that routed. */
static void gemv_pe_mac(hls::stream<Pack16> &af, hls::stream<Pack16> &wf,
                        hls::stream<float> &of, uint32_t stripe, uint32_t k_packs) {
    float a_loc[IN_DIM_MAX];
    #pragma HLS array_partition variable=a_loc cyclic factor=16

    gemv_pe_load_a: for (uint32_t kp = 0; kp < k_packs; ++kp) {
    #pragma HLS loop_tripcount min=128 max=352
    #pragma HLS pipeline II=1
        Pack16 v = af.read();
        gemv_pe_la_un: for (int kk = 0; kk < 16; ++kk) {
        #pragma HLS unroll
            a_loc[kp * 16 + kk] = v.data[kk];
        }
    }

    gemv_pe_o: for (uint32_t i = 0; i < stripe; ++i) {
    #pragma HLS loop_tripcount min=512 max=2816
        float part[GEMV_PARTIAL];
        #pragma HLS array_partition variable=part complete
        gemv_pe_init: for (int p = 0; p < GEMV_PARTIAL; ++p) {
        #pragma HLS unroll
            part[p] = 0.0f;
        }
        gemv_pe_k: for (uint32_t kp = 0; kp < k_packs; ++kp) {
        #pragma HLS loop_tripcount min=128 max=352
        #pragma HLS pipeline II=1
            Pack16 w = wf.read();
            float lane = 0.0f;
            /* Reduce in LUT fabric, not DSP: keeps the full_dsp fp-adders out of
             * the (DSP-dense) multiply region — the partially-conflicted-net
             * congestion that failed the monolithic build. */
            #pragma HLS bind_op variable=lane op=fadd impl=fabric
            gemv_pe_lane: for (int kk = 0; kk < 16; ++kk) {
            #pragma HLS unroll
                lane += w.data[kk] * a_loc[kp * 16 + kk];
            }
            part[kp & (GEMV_PARTIAL - 1)] += lane;
        }
        float s0 = part[0] + part[1];
        float s1 = part[2] + part[3];
        float s2 = part[4] + part[5];
        float s3 = part[6] + part[7];
        of.write((s0 + s1) + (s2 + s3));
    }
}

/* Collector: the single writer to `out`. Packs 16 consecutive dot products of
 * each channel into one 512-bit beat written to that channel's output stripe.
 * MAC-free (no DSP) → adds no congestion to the PE regions. */
static void gemv_collect(hls::stream<float> of[GEMV_CHANNELS], Pack16 *out_p,
                         uint32_t stripe, uint32_t stripe_packs) {
    Pack16 buf[GEMV_CHANNELS];
    #pragma HLS array_partition variable=buf complete
    gemv_col: for (uint32_t i = 0; i < stripe; ++i) {
    #pragma HLS loop_tripcount min=512 max=2816
        gemv_col_rd: for (int c = 0; c < GEMV_CHANNELS; ++c) {
        #pragma HLS unroll
            buf[c].data[i & 15] = of[c].read();
        }
        if ((i & 15) == 15) {
            gemv_col_wr: for (int c = 0; c < GEMV_CHANNELS; ++c) {
            #pragma HLS unroll
                out_p[(size_t)c * stripe_packs + (i >> 4)] = buf[c];
            }
        }
    }
}

/* Decode GEMV with GEMV_CHANNELS parallel HBM weight readers, COMPACT-SHARDED.
 * shard c is a distinct buffer on its own m_axi master/HBM channels holding
 * output stripe c (rows [c*stripe,(c+1)*stripe)) of every projection, packed
 * back-to-back. All shards share one layout, so this projection's stripe sits at
 * the SAME `shard_off` (Pack16) in every shard. Reader c streams its stripe from
 * shard c into PE c; the PEs run as independent dataflow processes (placed apart
 * → routable) and gemv_collect writes their outputs. Shards are built by
 * gdn_build_weight_shards (host) in the same projection order gdn_forward threads
 * `shard_off`. out_dim % (16*GEMV_CHANNELS) == 0 for every projection (2048,
 * 5632), so stripe boundaries are Pack16-aligned. */
static void gdn_gemv(
    float *out, const float *in,
    const float *shard0, const float *shard1,
    const float *shard2, const float *shard3,
    const float *shard4, const float *shard5,
    const float *shard6, const float *shard7, uint32_t shard_off,
    uint32_t num_rows, uint32_t in_dim, uint32_t out_dim
) {
    #pragma HLS inline off
    (void)num_rows;  /* decode GEMV: always the single token (row 0) */

    const Pack16 *in_p = reinterpret_cast<const Pack16 *>(in);
    const Pack16 *sh[GEMV_CHANNELS];
    #pragma HLS array_partition variable=sh complete
    sh[0] = reinterpret_cast<const Pack16 *>(shard0);
    sh[1] = reinterpret_cast<const Pack16 *>(shard1);
    sh[2] = reinterpret_cast<const Pack16 *>(shard2);
    sh[3] = reinterpret_cast<const Pack16 *>(shard3);
    sh[4] = reinterpret_cast<const Pack16 *>(shard4);
    sh[5] = reinterpret_cast<const Pack16 *>(shard5);
    sh[6] = reinterpret_cast<const Pack16 *>(shard6);
    sh[7] = reinterpret_cast<const Pack16 *>(shard7);
    Pack16 *out_p = reinterpret_cast<Pack16 *>(out);

    uint32_t k_packs      = in_dim / 16;
    uint32_t stripe       = out_dim / GEMV_CHANNELS;  /* 16-aligned for our shapes */
    uint32_t stripe_packs = stripe >> 4;              /* output packs per channel */
    uint32_t burst_packs  = stripe * k_packs;         /* weight packs per channel */

    hls::stream<Pack16> af[GEMV_CHANNELS];
    hls::stream<Pack16> wf[GEMV_CHANNELS];
    hls::stream<float>  of[GEMV_CHANNELS];
    #pragma HLS array_partition variable=af complete
    #pragma HLS array_partition variable=wf complete
    #pragma HLS array_partition variable=of complete
    #pragma HLS stream variable=af depth=512
    #pragma HLS stream variable=wf depth=128
    #pragma HLS stream variable=of depth=64

    #pragma HLS dataflow
    gemv_pe_bcast(in_p, af, k_packs);
    gemv_pe: for (int c = 0; c < GEMV_CHANNELS; ++c) {
    #pragma HLS unroll
        gemv_read_ch(sh[c], (size_t)shard_off, burst_packs, wf[c]);
        gemv_pe_mac(af[c], wf[c], of[c], stripe, k_packs);
    }
    gemv_collect(of, out_p, stripe, stripe_packs);
}

/* On-chip greedy argmax over the [vocab] logits the lm_head gemv wrote to HBM.
 * First-max tie-break (strict >), matching the host gdn_compute_logits /
 * argmax_logits the decode golden was validated against. Reads Pack16 bursts;
 * writes the token id as a float into out_token_f[0] (= x_norm[0]), the kernel's
 * 1-int decode output. ~vocab cycles — negligible vs the layer gemvs. */
static void gdn_argmax(float *out_token_f, const float *logits, uint32_t vocab) {
    const Pack16 *lp = reinterpret_cast<const Pack16 *>(logits);
    uint32_t n_packs = vocab >> 4;             /* vocab % 16 == 0 (32000) */
    float best = -3.402823466e38f;             /* -FLT_MAX */
    uint32_t best_i = 0;
    argmax_pk: for (uint32_t p = 0; p < n_packs; ++p) {
    #pragma HLS loop_tripcount min=2000 max=2000
        Pack16 v = lp[p];
        argmax_j: for (int j = 0; j < 16; ++j) {
            if (v.data[j] > best) { best = v.data[j]; best_i = (p << 4) + (uint32_t)j; }
        }
    }
    out_token_f[0] = (float)best_i;
}

