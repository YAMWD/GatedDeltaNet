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
#define GDN_RECURRENT_LANES 16 /* recurrent-state column parallelism */
#define GDN_NORM_LANES       8 /* RMS/output-norm arithmetic lanes   */
#define GDN_CONV_LANES       4 /* depthwise-convolution lanes        */
#define GDN_OUTPUT_GATE_LANES 4 /* output norm/gate arithmetic lanes */
#define GDN_SWIGLU_LANES      4 /* independent SwiGLU lanes          */
#define GDN_PK     GDN_RECURRENT_LANES

/* iter19 (step 4): the decode kernel is specialized for the fixed GDN-1.3B shape.
 * These match the .gdnw header (verified by read: vocab 32000, hidden 2048,
 * layers 24, heads 8, v_heads 8, head_dim 256, inter 5632, conv 4, eps 1e-6).
 * Hardcoding them lets gdn_forward drop the runtime `config` pointer and use
 * constant loop bounds -- simplifying the top-level ap_CS_fsm that iter12 flagged
 * as a 95% routing-delay critical path. The host-side helpers keep reading
 * config; only the synthesized kernel path is specialized. */
#define GDN_HIDDEN    2048
#define GDN_LAYERS      24
#define GDN_V_HEADS      8
#define GDN_HEAD_DIM  GDN_DK      /* 256 */
#define GDN_INTER     5632
#define GDN_CONV         4
#define GDN_VOCAB    32000
#define GDN_NORM_EPS  1e-6f       /* .gdnw header 0x358637BD, bit-exact as 1e-6f */
/* Compile-time form of gdn_aux_layer_stride() for the hardcoded shape; must equal
 * H + 2*nh + 2*nh*H + 3*H*cs + hd + H exactly (see gdn_aux_layer_stride). */
#define GDN_AUX_LAYER_STRIDE ((size_t)GDN_HIDDEN + 2*(size_t)GDN_HEADS \
    + 2*(size_t)GDN_HEADS*GDN_HIDDEN + 3*(size_t)GDN_HIDDEN*GDN_CONV \
    + GDN_HEAD_DIM + GDN_HIDDEN)
/* GEMV_CHANNELS lives in gdn_model.h (shared by the kernel and the host shard
 * builder / run-state); do not redefine it here. */

/* step 4 Stage B: the gdn_model.h workspace layout uses literal sizes; assert
 * they equal the GDN_* dims so the two can never silently drift. */
static_assert(GDN_WSF_HID == GDN_HIDDEN, "workspace hidden size drift");
static_assert(GDN_WSF_MLP == GDN_INTER, "workspace mlp size drift");
static_assert(GDN_WSF_STATE ==
    (size_t)GDN_LAYERS*GDN_HEADS*GDN_HEAD_DIM*(GDN_HIDDEN/GDN_HEADS),
    "workspace recurrent_state size drift");
static_assert(GDN_WSF_HEADBUF ==
    (size_t)GDN_LAYERS*3*(GDN_CONV-1)*GDN_HIDDEN, "workspace head_buffer drift");
static_assert(GDN_WSF_LOGITS == GDN_VOCAB, "workspace logits size drift");
static_assert(GDN_WSF_HEAD >= GDN_HEADS, "workspace head padding too small");

/* Pack16 = 16 FP32 values = 64 bytes = 512 bits. */
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

/* Compact non-GEMV weights. The host supplies the selected embedding row in x;
 * all large projection weights and lm_head already live in the 32 GEMV shards.
 * Keeping only these small tensors avoids a second 5.6 GB weight copy. */
static size_t gdn_aux_layer_stride(const GDNWeightHeader *config) {
    size_t H = config->hidden_size;
    size_t nh = config->num_heads;
    size_t hd = config->head_dim;
    size_t cs = config->conv_size;
    return H + 2 * nh + 2 * nh * H + 3 * H * cs + hd + H;
}

size_t gdn_aux_weight_floats(const GDNWeightHeader *config) {
    return (size_t)config->num_layers * gdn_aux_layer_stride(config) +
           config->hidden_size;
}

void gdn_build_aux_weights(const float *wd, const GDNWeightHeader *config,
                           float *aux) {
    size_t H = config->hidden_size;
    size_t nh = config->num_heads;
    size_t hd = config->head_dim;
    size_t cs = config->conv_size;
    size_t dst = 0;

    for (uint32_t layer = 0; layer < config->num_layers; ++layer) {
        size_t src = gdn_layer_weight_offset(config, layer);
        size_t count = H + 2 * nh;
        memcpy(aux + dst, wd + src, count * sizeof(float));
        dst += count;
        src += count + 3 * H * H;

        count = 2 * nh * H + 3 * H * cs;
        memcpy(aux + dst, wd + src, count * sizeof(float));
        dst += count;
        src += count + H * H;

        memcpy(aux + dst, wd + src, hd * sizeof(float));
        dst += hd;
        src += hd + H * H;

        memcpy(aux + dst, wd + src, H * sizeof(float));
        dst += H;
    }

    memcpy(aux + dst, wd + gdn_final_norm_offset(config), H * sizeof(float));
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

    /* step 4 Stage B: ONE workspace allocation; the 15 activation/state buffers
     * are views into it at the shared GDN_WS_OFF_* offsets, mirroring the kernel
     * and the on-card host so the csim exercises the identical packed layout.
     * (Decode-only: max_tokens is 1, so the fixed 1-token layout suffices; the
     * static_asserts in gdn_model.cpp tie GDN_WSF_* to the model dims.) */
    /* Decode processes exactly one token per gdn_forward call, so the activation
     * views are 1-token regardless of the caller's max_tokens sizing hint (the
     * legacy prefill sizing is ignored here). recurrent_state / head_buffer are
     * all-layers state, independent of token count. */
    (void)max_tokens; (void)hidden_tokens; (void)head_tokens;
    (void)intermediate; (void)head_dim; (void)value_dim;
    if (gdn_alloc_run_buffer(&state->workspace, GDN_WS_FLOATS) != 0) return -1;
    state->x               = state->workspace + GDN_WS_OFF_X;
    state->x_norm          = state->workspace + GDN_WS_OFF_X_NORM;
    state->q               = state->workspace + GDN_WS_OFF_Q;
    state->k               = state->workspace + GDN_WS_OFF_K;
    state->v               = state->workspace + GDN_WS_OFF_V;
    state->a               = state->workspace + GDN_WS_OFF_A;
    state->b               = state->workspace + GDN_WS_OFF_B;
    state->gate            = state->workspace + GDN_WS_OFF_GATE;
    state->attn            = state->workspace + GDN_WS_OFF_ATTN;
    state->tmp_hidden      = state->workspace + GDN_WS_OFF_TMP_HIDDEN;
    state->mlp_gate        = state->workspace + GDN_WS_OFF_MLP_GATE;
    state->mlp_up          = state->workspace + GDN_WS_OFF_MLP_UP;
    /* Decode persistence: recurrent_state holds ALL layers (24 x 2 MB = 48 MB)
     * and head_buffer is repurposed as the conv tail store: per layer, 3 convs
     * (q/k/v) x (conv_size-1) rows x hidden floats (~1.7 MB). */
    state->recurrent_state = state->workspace + GDN_WS_OFF_REC_STATE;
    state->head_buffer     = state->workspace + GDN_WS_OFF_HEAD_BUF;

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
    if (gdn_alloc_run_buffer(&state->aux_weights,
            gdn_aux_weight_floats(&model->config)) != 0) return -1;
    gdn_build_aux_weights(model->weight_data, &model->config, state->aux_weights);
    /* lm_head gemv scratch view (decode writes logits here, argmaxes to x_norm[0]). */
    state->logits = state->workspace + GDN_WS_OFF_LOGITS;

    return 0;
}

void gdn_run_state_free(GDNRunState *state) {
    /* step 4 Stage B: x..head_buffer and logits are views into workspace; free
     * the single workspace allocation, not each view. */
    free(state->workspace);
    {
        int c;
        for (c = 0; c < GEMV_CHANNELS; ++c) free(state->weight_shards[c]);
    }
    free(state->aux_weights);
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

/* Preserve the balanced pairwise reduction order while reusing one pipelined
 * adder per level. The fully unrolled form instantiated 255 adders at each
 * call site even though recurrent attention invokes these reductions
 * sequentially; that density is counterproductive beside the 32-port GEMV. */
static float gdn_tree_reduce_256(const float arr[256]) {
#pragma HLS inline off
    float l128[128];
    float l64[64];
    float l32[32];
    float l16[16];
    float l8[8];
    float l4[4];
    float l2[2];
    #pragma HLS bind_storage variable=l128 type=ram_2p impl=bram
    #pragma HLS bind_storage variable=l64  type=ram_2p impl=bram
    #pragma HLS bind_storage variable=l32  type=ram_2p impl=bram
    #pragma HLS array_partition variable=l16  complete
    #pragma HLS array_partition variable=l8   complete
    #pragma HLS array_partition variable=l4   complete
    #pragma HLS array_partition variable=l2   complete

    uint32_t i;
    L128: for (i = 0; i < 128; ++i) {
    #pragma HLS pipeline II=1
        l128[i] = arr[2*i] + arr[2*i+1];
    }
    L64: for (i = 0; i < 64; ++i) {
    #pragma HLS pipeline II=1
        l64[i] = l128[2*i] + l128[2*i+1];
    }
    L32: for (i = 0; i < 32; ++i) {
    #pragma HLS pipeline II=1
        l32[i] = l64[2*i] + l64[2*i+1];
    }
    L16: for (i = 0; i < 16; ++i) {
    #pragma HLS pipeline II=1
        l16[i] = l32[2*i] + l32[2*i+1];
    }
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

static void gdn_rmsnorm_rows(
    Pack16 *out,
    const Pack16 *in,
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
    uint32_t col_packs = num_cols / 16;

    /* Buffer the per-channel norm weight once (it is otherwise re-read every
     * row); partitioning follows the independently tuned norm lane count. */
    float w_loc[2048];
    #pragma HLS array_partition variable=w_loc cyclic factor=GDN_NORM_LANES
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
        #pragma HLS pipeline II=2
            Pack16 v = in[(size_t)row * col_packs + cp];
            float s = 0.0f;
            sq_lane: for (int kk = 0; kk < 16; ++kk) {
            #pragma HLS unroll factor=GDN_NORM_LANES
                s += v.data[kk] * v.data[kk];
            }
            sum += (double)s;
        }
        float scale = 1.0f / sqrtf((float)(sum / num_cols) + eps);
        rmsnorm_scale: for (uint32_t cp = 0; cp < col_packs; ++cp) {
        #pragma HLS loop_tripcount min=128 max=128
        #pragma HLS pipeline II=2
            Pack16 v = in[(size_t)row * col_packs + cp];
            Pack16 o;
            scl_lane: for (int kk = 0; kk < 16; ++kk) {
            #pragma HLS unroll factor=GDN_NORM_LANES
                o.data[kk] = v.data[kk] * scale * w_loc[cp * 16 + kk];
            }
            out[(size_t)row * col_packs + cp] = o;
        }
    }
}


/* Compile-time bounds for gdn_gemv_tiny's on-chip buffers (the a/b gate
 * projections: out_dim = num_heads = 8, in_dim = hidden = 2048). */
#define GDN_GEMV_TINY_OUT_MAX 8
#define GDN_GEMV_TINY_IN_MAX  2048
#define GDN_GEMV_TINY_OUT_LANES 2

static void gdn_gemv_tiny(
    float *out,
    const Pack16 *in,
    const float *weights,
    uint32_t num_rows,
    uint32_t in_dim,
    uint32_t out_dim
) {
    /* Decode-shape GEMV for the tiny a/b gate projections (num_rows=1,
     * in_dim=hidden=2048, out_dim=num_heads=8). Three buffered steps:
     *   1. load the single activation row into resident a_loc (read once);
     *   2. preload all out_dim weight rows to BRAM as one CONTIGUOUS burst (a/b are
     *      [out_dim][in_dim] row-major) — avoids the per-(c,kc) strided HBM reads;
     *   3. one k-pass computing two output rows at a time, each with its own
     *      accumulator + the SAME balanced-tree-per-16-chunk sequential reduction.
     * Bit-exact to the prior per-output reduction (each acc[c] keeps the chunk
     * order); removes the 8x per-output pipeline restart + redundant activation
     * reads that made the prior form ~0.18 ms/call. */
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
        Pack16 a = in[kc];
        gvt_la_i: for (i = 0; i < 16; ++i) {
        #pragma HLS unroll
            a_loc[kc * 16 + i] = a.data[i];
        }
    }

    /* (2) preload weights to BRAM — one contiguous burst over [out_dim][in_dim] */
    float w_loc[GDN_GEMV_TINY_OUT_MAX][GDN_GEMV_TINY_IN_MAX];
    #pragma HLS array_partition variable=w_loc dim=1 cyclic factor=GDN_GEMV_TINY_OUT_LANES
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

    /* (3) one k-pass with two physical output lanes. Keep each output's kc
     * accumulation order unchanged, but pipeline groups of independent outputs
     * instead of pipelining gvt_k. Pipelining gvt_k made Vitis completely
     * unroll gvt_c (all 8 outputs), ignoring its partial-unroll factor. */
    float acc[GDN_GEMV_TINY_OUT_MAX];
    #pragma HLS array_partition variable=acc cyclic factor=GDN_GEMV_TINY_OUT_LANES
    gvt_init: for (c = 0; c < out_dim; ++c) {
    #pragma HLS unroll factor=GDN_GEMV_TINY_OUT_LANES
        acc[c] = 0.0f;
    }
    gvt_k: for (kc = 0; kc < k_packs; ++kc) {
    #pragma HLS loop_tripcount min=128 max=128
        gvt_c_group: for (uint32_t cg = 0;
                          cg < GDN_GEMV_TINY_OUT_MAX;
                          cg += GDN_GEMV_TINY_OUT_LANES) {
        #pragma HLS loop_tripcount min=4 max=4
        #pragma HLS pipeline II=2
            gvt_c_lane: for (uint32_t co = 0;
                             co < GDN_GEMV_TINY_OUT_LANES; ++co) {
            #pragma HLS unroll
                uint32_t c_lane = cg + co;
                if (c_lane < out_dim) {
                    float p[16];
                    #pragma HLS array_partition variable=p complete
                    gvt_mul: for (i = 0; i < 16; ++i) {
                    #pragma HLS unroll
                        p[i] = a_loc[kc * 16 + i]
                             * w_loc[c_lane][kc * 16 + i];
                    }
                    /* Same balanced 4-level tree as before. */
                    float s2_0 = p[0]  + p[1],  s2_1 = p[2]  + p[3];
                    float s2_2 = p[4]  + p[5],  s2_3 = p[6]  + p[7];
                    float s2_4 = p[8]  + p[9],  s2_5 = p[10] + p[11];
                    float s2_6 = p[12] + p[13], s2_7 = p[14] + p[15];
                    float s4_0 = s2_0 + s2_1, s4_1 = s2_2 + s2_3;
                    float s4_2 = s2_4 + s2_5, s4_3 = s2_6 + s2_7;
                    float s8_0 = s4_0 + s4_1, s8_1 = s4_2 + s4_3;
                    acc[c_lane] += s8_0 + s8_1;
                }
            }
        }
    }
    gvt_st: for (c = 0; c < out_dim; ++c) {
    #pragma HLS unroll factor=GDN_GEMV_TINY_OUT_LANES
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
    Pack16 *out,
    const Pack16 *in,
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
    #pragma HLS array_partition variable=w_loc dim=1 cyclic factor=GDN_CONV_LANES

    float in_window[GDN_CONV_K_MAX][GDN_CONV_COLS_MAX];
    #pragma HLS array_partition variable=in_window dim=1 complete
    #pragma HLS array_partition variable=in_window dim=2 cyclic factor=GDN_CONV_LANES

    /* Pack16-widened activation I/O: 16 channels (512-bit) per beat. conv is
     * depthwise, so channels are independent and contiguous — index the Pack16
     * base by an integer (row*col_packs + cp). num_cols=hidden=2048 (16|cols). */
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

    /* Streaming conv: per row, separate load/shift from shared-lane compute.
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
            Pack16 v = in[(size_t)row * col_packs + cp];
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
            float o_lane[16];
            #pragma HLS array_partition variable=o_lane complete
            /* Pipeline four-lane groups. Pipelining conv_compute itself forces
             * conv_comp_lane to unroll all 16 channels. */
            conv_comp_group: for (int kb = 0; kb < 16;
                                  kb += GDN_CONV_LANES) {
            #pragma HLS loop_tripcount min=4 max=4
            #pragma HLS pipeline II=1
                conv_comp_lane: for (int kl = 0;
                                     kl < GDN_CONV_LANES; ++kl) {
                #pragma HLS unroll
                    int kk = kb + kl;
                    uint32_t c = cp * 16 + (uint32_t)kk;
                    /* in_window[k] holds source row
                     * (row - kernel_size + 1 + k). */
                    float sum = in_window[0][c] * w_loc[c][0]
                              + in_window[1][c] * w_loc[c][1]
                              + in_window[2][c] * w_loc[c][2]
                              + in_window[3][c] * w_loc[c][3];
                    o_lane[kk] = gdn_silu(sum);
                }
            }
            Pack16 o;
            conv_pack_out: for (int kk = 0; kk < 16; ++kk) {
            #pragma HLS unroll
                o.data[kk] = o_lane[kk];
            }
            out[(size_t)row * col_packs + cp] = o;
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
 *   1. One head-local URAM state buffer
 *   2. Fused HBM restore/retrieval and update/HBM-save passes
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
    Pack16 *attn_out,
    float *recurrent_state,  /* decode: per-layer state at layer_index*8*256*256; prefill: unused */
    float *head_buffer,      /* unused: local buffers used instead */
    const Pack16 *q,
    const Pack16 *k,
    const Pack16 *v,
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
    /* Buffer one 256 x 256 FP32 head rather than restoring all eight heads
     * before computation. Each old-state word is captured while the retrieval
     * pass consumes it; each updated word is written to HBM directly from the
     * update pass. This removes the two standalone 32,768-word layer copies and
     * reduces the state memory from 128 to an expected 16 URAMs without changing
     * the FP32 arithmetic or external state layout. */
    float state[GDN_DK][GDN_DV];
#pragma HLS bind_storage variable=state type=RAM_2P impl=URAM
#pragma HLS array_partition variable=state dim=2 cyclic factor=GDN_PK

    float q_scale = 1.0f / sqrtf((float)GDN_DK);
    uint32_t j, i;
    uint32_t token_index;
    /* Layer slice of the HBM-resident decode state (48 MB across 24 layers).
     * Cast the external state once and address it in native 512-bit words.
     * Merely unrolling scalar float accesses does not make HLS coalesce them:
     * iter27 measured one four-byte AXI transaction per float. */
    size_t st_base = (size_t)layer_index * GDN_HEADS * GDN_DK * GDN_DV;
    size_t st_base16 = st_base >> 4;
    const Pack16 *recurrent_state_in =
        reinterpret_cast<const Pack16 *>(recurrent_state);
    Pack16 *recurrent_state_out =
        reinterpret_cast<Pack16 *>(recurrent_state);

    recur_token: for (token_index = 0; token_index < num_tokens; ++token_index) {
    #pragma HLS loop_tripcount min=1 max=2048
        uint32_t head_index;
        recur_head: for (head_index = 0; head_index < GDN_HEADS; ++head_index) {
        #pragma HLS loop_tripcount min=8 max=8

            const Pack16 *q_head = q +
                (size_t)token_index * (hidden / 16)
                + (size_t)head_index * (GDN_DK / 16);
            const Pack16 *k_head = k +
                (size_t)token_index * (hidden / 16)
                + (size_t)head_index * (GDN_DK / 16);
            const Pack16 *v_head = v +
                (size_t)token_index * (hidden / 16)
                + (size_t)head_index * (GDN_DV / 16);
            Pack16 *out_head = attn_out +
                (size_t)token_index * (hidden / 16)
                + (size_t)head_index * (GDN_DV / 16);
            size_t head_state_base16 = st_base16 +
                (size_t)head_index * GDN_DK * (GDN_DV / 16);

            /* ---- Local per-token buffers ---- */
            float q_loc[GDN_DK];
            float k_loc[GDN_DK];
            float v_loc[GDN_DV];
            float r_buf[GDN_DV];   /* retrieval result           */
            float o_buf[GDN_DV];   /* partial output              */
            float dv[GDN_DV];      /* delta correction            */
#pragma HLS array_partition variable=r_buf cyclic factor=GDN_PK
#pragma HLS array_partition variable=o_buf cyclic factor=GDN_PK
#pragma HLS array_partition variable=dv    cyclic factor=GDN_PK
#pragma HLS array_partition variable=v_loc cyclic factor=GDN_PK

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
            #pragma HLS array_partition variable=qsq_arr cyclic factor=2
            #pragma HLS array_partition variable=ksq_arr cyclic factor=2
            #pragma HLS bind_storage variable=qsq_arr type=ram_2p impl=bram
            #pragma HLS bind_storage variable=ksq_arr type=ram_2p impl=bram

            load_qk: for (j = 0; j < GDN_DK; ++j) {
            #pragma HLS loop_tripcount min=256 max=256
            #pragma HLS pipeline II=1
                float qj = q_head[j >> 4].data[j & 15];
                float kj = k_head[j >> 4].data[j & 15];
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
                v_loc[i] = v_head[i >> 4].data[i & 15];
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
            #pragma HLS array_partition variable=alpha_prod cyclic factor=2
            #pragma HLS bind_storage variable=alpha_prod type=ram_2p impl=bram

            dot_alpha: for (j = 0; j < GDN_DK; ++j) {
            #pragma HLS loop_tripcount min=256 max=256
            #pragma HLS pipeline II=1
                alpha_prod[j] = q_loc[j] * k_loc[j];
            }

            float alpha = gdn_tree_reduce_256(alpha_prod);

            /* ---- Phase 2: FUSED HBM RESTORE + READ PASS ----
             * For each column i, accumulate across rows j:
             *   r_buf[i] = Σ_j S[j][i] * k_norm[j]   (retrieval)
             *   o_buf[i] = Σ_j S[j][i] * q_norm[j]   (partial output)
             * The same 512-bit word is retained in the head-local state buffer
             * for the later update, eliminating a separate restore traversal.
             */
            init_ro: for (i = 0; i < GDN_DV; ++i) {
            #pragma HLS loop_tripcount min=256 max=256
            #pragma HLS pipeline II=1
            #pragma HLS unroll factor=GDN_PK
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
                    Pack16 state_word = recurrent_state_in[
                        head_state_base16 +
                        (size_t)j * (GDN_DV / 16) +
                        (i >> 4)];
                    uint32_t pp;
                    for (pp = 0; pp < GDN_PK; ++pp) {
                    #pragma HLS unroll
                        float s = state_word.data[pp];
                        state[j][i + pp] = s;
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
            #pragma HLS array_partition variable=out_loc cyclic factor=GDN_PK

            delta_out: for (i = 0; i < GDN_DV; ++i) {
            #pragma HLS loop_tripcount min=256 max=256
            #pragma HLS pipeline II=1
            #pragma HLS unroll factor=GDN_PK
                float d = beta * (v_loc[i] - g * r_buf[i]);
                dv[i] = d;
                out_loc[i] = q_scale * (g * o_buf[i] + alpha * d);
            }

            delta_drain: for (i = 0; i < GDN_DV / 16; ++i) {
            #pragma HLS loop_tripcount min=16 max=16
            #pragma HLS pipeline II=1
                Pack16 out_word;
            delta_drain_lane: for (uint32_t lane = 0; lane < 16; ++lane) {
            #pragma HLS unroll
                    out_word.data[lane] = out_loc[i * 16 + lane];
                }
                out_head[i] = out_word;
            }

            /* ---- Phase 4: FUSED UPDATE + HBM SAVE PASS ----
             * S[j][i] = g * S[j][i] + k_norm[j] * Δv[i]
             * Pack and persist each updated word immediately, eliminating a
             * separate save traversal of the complete layer state.
             */
            fused_wr_j: for (j = 0; j < GDN_DK; ++j) {
            #pragma HLS loop_tripcount min=256 max=256
                float kj = k_loc[j];
                fused_wr_i: for (i = 0; i < GDN_DV; i += GDN_PK) {
                #pragma HLS loop_tripcount min=16 max=16
                #pragma HLS pipeline II=1
                    Pack16 state_word;
                    uint32_t pp;
                    for (pp = 0; pp < GDN_PK; ++pp) {
                    #pragma HLS unroll
                        float updated = g * state[j][i + pp]
                                      + kj * dv[i + pp];
                        state[j][i + pp] = updated;
                        state_word.data[pp] = updated;
                    }
                    recurrent_state_out[
                        head_state_base16 +
                        (size_t)j * (GDN_DV / 16) +
                        (i >> 4)] = state_word;
                }
            }
        } /* recur_head */
    } /* recur_token */
}

static void gdn_output_norm_and_gate(
    Pack16 *attn,
    const Pack16 *gate,
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
    uint32_t hd_packs = head_dim / 16;

    /* Pre-load the per-head norm weight once and reuse for every (token, head). */
    float weight_loc[GDN_DV];
    #pragma HLS array_partition variable=weight_loc cyclic factor=GDN_NORM_LANES
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
            #pragma HLS array_partition variable=attn_loc cyclic factor=GDN_NORM_LANES
            #pragma HLS array_partition variable=gate_loc cyclic factor=GDN_NORM_LANES

            /* Phase 1: load attn (Pack16) into local + accumulate sum of squares */
            double sum = 0.0;
            onorm_sq: for (uint32_t ip = 0; ip < hd_packs; ++ip) {
            #pragma HLS loop_tripcount min=16 max=16
            #pragma HLS pipeline II=2
                Pack16 v = attn[base + ip];
                float s = 0.0f;
                onorm_sq_lane: for (int kk = 0; kk < 16; ++kk) {
                #pragma HLS unroll factor=GDN_NORM_LANES
                    float a = v.data[kk];
                    attn_loc[ip * 16 + kk] = a;
                    s += a * a;
                }
                sum += (double)s;
            }

            /* Phase 2: load gate (Pack16) into local buffer */
            onorm_load_g: for (uint32_t ip = 0; ip < hd_packs; ++ip) {
            #pragma HLS loop_tripcount min=16 max=16
            #pragma HLS pipeline II=2
                Pack16 g = gate[base + ip];
                onorm_g_lane: for (int kk = 0; kk < 16; ++kk) {
                #pragma HLS unroll factor=GDN_NORM_LANES
                    gate_loc[ip * 16 + kk] = g.data[kk];
                }
            }

            float scale = 1.0f / sqrtf((float)(sum / (double)head_dim) + eps);

            /* Phase 3: combine and write back (Pack16) */
            onorm_gate: for (uint32_t ip = 0; ip < hd_packs; ++ip) {
            #pragma HLS loop_tripcount min=16 max=16
                float o_lane[16];
                #pragma HLS array_partition variable=o_lane complete
                /* Retain four physical lanes here: fully parallel output-gate
                 * arithmetic saves too few token cycles for its DSP cost. */
                onorm_gate_group: for (int kb = 0; kb < 16;
                                       kb += GDN_OUTPUT_GATE_LANES) {
                #pragma HLS loop_tripcount min=4 max=4
                #pragma HLS pipeline II=1
                    onorm_gate_lane: for (int kl = 0;
                                          kl < GDN_OUTPUT_GATE_LANES; ++kl) {
                    #pragma HLS unroll
                        int kk = kb + kl;
                        uint32_t index = ip * 16 + (uint32_t)kk;
                        float normalized =
                            attn_loc[index] * scale * weight_loc[index];
                        float gate_value = gate_loc[index];
                        o_lane[kk] = normalized * gate_value
                                   * gdn_sigmoid(gate_value);
                    }
                }
                Pack16 o;
                onorm_pack_out: for (int kk = 0; kk < 16; ++kk) {
                #pragma HLS unroll
                    o.data[kk] = o_lane[kk];
                }
                attn[base + ip] = o;
            }
        }
    }
}

/* SwiGLU in place — `gate[i] = silu(gate[i]) * up[i]`. Vectorised over
 * Pack16 (16 FP32 lanes / 64 bytes) so HLS uses the 512-bit m_axi adapter
 * for one wide read + one wide read + one wide write per iter instead of
 * three narrow accesses per element. count is always a multiple of 16
 * (count = num_tokens × intermediate, intermediate=5632 ⇒ 16 | count). */
static void gdn_swiglu_inplace(Pack16 *gate, const Pack16 *up, size_t count) {
    const size_t count16 = count >> 4;  /* count / 16 */
    swiglu_loop: for (size_t i = 0; i < count16; ++i) {
    #pragma HLS loop_tripcount min=352 max=720896  /* count16: 5632/16 .. 2048*5632/16 */
        Pack16 g = gate[i];
        Pack16 u = up[i];
        float g_lane[16];
        #pragma HLS array_partition variable=g_lane complete
        swiglu_group: for (int jb = 0; jb < 16;
                           jb += GDN_SWIGLU_LANES) {
        #pragma HLS loop_tripcount min=4 max=4
        #pragma HLS pipeline II=1
            swiglu_lane: for (int jl = 0;
                              jl < GDN_SWIGLU_LANES; ++jl) {
            #pragma HLS unroll
                int j = jb + jl;
                g_lane[j] = gdn_silu(g.data[j]) * u.data[j];
            }
        }
        swiglu_pack_out: for (int j = 0; j < 16; ++j) {
        #pragma HLS unroll
            g.data[j] = g_lane[j];
        }
        gate[i] = g;
    }
}

/* A single pair of maximum-size BRAM apertures fronts the shared GEMV engine.
 * These local transfers are one 512-bit word/cycle and prevent Vitis from
 * specializing a complete 16-cluster datapath for each activation buffer
 * shape. */
static void gdn_pack16_copy_local(
    Pack16 *out, const Pack16 *in, uint32_t count16
) {
#pragma HLS inline
copy_local: for (uint32_t i = 0; i < count16; ++i) {
#pragma HLS loop_tripcount min=128 max=352
#pragma HLS pipeline II=1
        out[i] = in[i];
    }
}

static void gdn_pack16_add_local(
    Pack16 *residual, const Pack16 *projection, uint32_t count16
) {
#pragma HLS inline
add_local: for (uint32_t i = 0; i < count16; ++i) {
#pragma HLS loop_tripcount min=128 max=128
#pragma HLS pipeline II=1
        Pack16 sum = residual[i];
        Pack16 value = projection[i];
    add_local_lane: for (uint32_t lane = 0; lane < 16; ++lane) {
#pragma HLS unroll
            sum.data[lane] += value.data[lane];
        }
        residual[i] = sum;
    }
}

/* Forward decl: the decode-only clustered GEMV engine. */
static void gdn_gemv(
    Pack16 *out, const Pack16 *in,
    const float *w0, const float *w1, const float *w2, const float *w3,
    const float *w4, const float *w5, const float *w6, const float *w7,
    const float *w8, const float *w9, const float *w10, const float *w11,
    const float *w12, const float *w13, const float *w14, const float *w15,
    const float *w16, const float *w17, const float *w18, const float *w19,
    const float *w20, const float *w21, const float *w22, const float *w23,
    const float *w24, const float *w25, const float *w26, const float *w27,
    const float *w28, const float *w29, const float *w30, const float *w31,
    uint32_t w_pack_off,
    uint32_t num_rows, uint32_t in_dim, uint32_t out_dim);

/* Vitis HLS 2022.1 does not synthesize arrays of pointers. This expands each
 * call to the explicit scalar pointer interface above. */
#define GDN_GEMV_SHARD_ARGUMENTS \
    weight_data_mm0, weight_data_mm1, weight_data_mm2, weight_data_mm3, \
    weight_data_mm4, weight_data_mm5, weight_data_mm6, weight_data_mm7, \
    weight_data_mm8, weight_data_mm9, weight_data_mm10, weight_data_mm11, \
    weight_data_mm12, weight_data_mm13, weight_data_mm14, weight_data_mm15, \
    weight_data_mm16, weight_data_mm17, weight_data_mm18, weight_data_mm19, \
    weight_data_mm20, weight_data_mm21, weight_data_mm22, weight_data_mm23, \
    weight_data_mm24, weight_data_mm25, weight_data_mm26, weight_data_mm27, \
    weight_data_mm28, weight_data_mm29, weight_data_mm30, weight_data_mm31

int gdn_forward(
    const float *aux_weights,
    float *workspace,   /* step 4 Stage B: 15 activation/state buffers packed here
                         * at GDN_WS_OFF_*; derived into locals below. */
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
) {
    /* Depths match gdn-1.3b-f32.gdnw: hidden=2048 heads=8 head_dim=256
    intermediate=5632 layers=24 conv=4 max_seq_len=2048 vocab=32000 */
    /* This U55C shell exposes 32 HMSS masters. Every scalar weight, activation,
     * and state buffer shares weight port 0 and is allocated in HBM0; ports 1..31
     * remain read-only. A combined x/w0 loader is the sole dataflow reader of
     * port 0. The host writes the selected token embedding directly into x.
     *
     * DO NOT reduce mm0's outstanding depths. iter14/iter15 tried 64->8 read and
     * write (to shrink the 29-deep BRAM cascade RQS_TIMING-6 flagged on iter13's
     * worst path) and BOTH links were REFUSED by the router with
     * [Route 35-3] not routable, where iter13 at 64 completed route_design.
     * iter15 isolated it: link cfg byte-identical to iter13, mm0 depth the only
     * variable. It saves 50 BRAM and costs routability -- the freed BRAM came
     * out of SLR1/SLR2 (SLR0 actually GAINED 7 tiles), which let the placer
     * compact the design, pull cluster 5 back into SLR0 and use SLR2 less. The
     * whole margin between routing and refusal is ~22 K cells of SLR0
     * occupancy. See doc/optimization_log.md sec iter15. */
    #pragma HLS interface m_axi port=aux_weights depth=2000000 offset=slave bundle=mem_weights_mm0 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=64 max_write_burst_length=64 num_write_outstanding=64
    /* Thirty-two compact GEMV shards, each on an independent 512-bit master.
     * The clustered datapath consumes one Pack16 beat per master per cycle. */
    /* One compact GDN-1.3B shard is 43,728,896 floats (166.8125 MiB).
     * Do not use the full-model float count here: it exceeds one AXI address
     * range and corrupts the metadata consumed by the Vitis platform linker. */
    #pragma HLS interface m_axi port=weight_data_mm0 depth=43728896 offset=slave bundle=mem_weights_mm0 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=64 max_write_burst_length=64 num_write_outstanding=64
    #pragma HLS interface m_axi port=weight_data_mm1 depth=43728896 offset=slave bundle=mem_weights_mm1 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=8
    #pragma HLS interface m_axi port=weight_data_mm2 depth=43728896 offset=slave bundle=mem_weights_mm2 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=8
    #pragma HLS interface m_axi port=weight_data_mm3 depth=43728896 offset=slave bundle=mem_weights_mm3 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=8
    #pragma HLS interface m_axi port=weight_data_mm4 depth=43728896 offset=slave bundle=mem_weights_mm4 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=8
    #pragma HLS interface m_axi port=weight_data_mm5 depth=43728896 offset=slave bundle=mem_weights_mm5 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=8
    #pragma HLS interface m_axi port=weight_data_mm6 depth=43728896 offset=slave bundle=mem_weights_mm6 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=8
    #pragma HLS interface m_axi port=weight_data_mm7 depth=43728896 offset=slave bundle=mem_weights_mm7 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=8
    #pragma HLS interface m_axi port=weight_data_mm8 depth=43728896 offset=slave bundle=mem_weights_mm8 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=8
    #pragma HLS interface m_axi port=weight_data_mm9 depth=43728896 offset=slave bundle=mem_weights_mm9 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=8
    #pragma HLS interface m_axi port=weight_data_mm10 depth=43728896 offset=slave bundle=mem_weights_mm10 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=8
    #pragma HLS interface m_axi port=weight_data_mm11 depth=43728896 offset=slave bundle=mem_weights_mm11 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=8
    #pragma HLS interface m_axi port=weight_data_mm12 depth=43728896 offset=slave bundle=mem_weights_mm12 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=8
    #pragma HLS interface m_axi port=weight_data_mm13 depth=43728896 offset=slave bundle=mem_weights_mm13 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=8
    #pragma HLS interface m_axi port=weight_data_mm14 depth=43728896 offset=slave bundle=mem_weights_mm14 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=8
    #pragma HLS interface m_axi port=weight_data_mm15 depth=43728896 offset=slave bundle=mem_weights_mm15 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=8
    #pragma HLS interface m_axi port=weight_data_mm16 depth=43728896 offset=slave bundle=mem_weights_mm16 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=8
    #pragma HLS interface m_axi port=weight_data_mm17 depth=43728896 offset=slave bundle=mem_weights_mm17 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=8
    #pragma HLS interface m_axi port=weight_data_mm18 depth=43728896 offset=slave bundle=mem_weights_mm18 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=8
    #pragma HLS interface m_axi port=weight_data_mm19 depth=43728896 offset=slave bundle=mem_weights_mm19 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=8
    #pragma HLS interface m_axi port=weight_data_mm20 depth=43728896 offset=slave bundle=mem_weights_mm20 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=8
    #pragma HLS interface m_axi port=weight_data_mm21 depth=43728896 offset=slave bundle=mem_weights_mm21 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=8
    #pragma HLS interface m_axi port=weight_data_mm22 depth=43728896 offset=slave bundle=mem_weights_mm22 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=8
    #pragma HLS interface m_axi port=weight_data_mm23 depth=43728896 offset=slave bundle=mem_weights_mm23 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=8
    #pragma HLS interface m_axi port=weight_data_mm24 depth=43728896 offset=slave bundle=mem_weights_mm24 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=8
    #pragma HLS interface m_axi port=weight_data_mm25 depth=43728896 offset=slave bundle=mem_weights_mm25 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=8
    #pragma HLS interface m_axi port=weight_data_mm26 depth=43728896 offset=slave bundle=mem_weights_mm26 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=8
    #pragma HLS interface m_axi port=weight_data_mm27 depth=43728896 offset=slave bundle=mem_weights_mm27 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=8
    #pragma HLS interface m_axi port=weight_data_mm28 depth=43728896 offset=slave bundle=mem_weights_mm28 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=8
    #pragma HLS interface m_axi port=weight_data_mm29 depth=43728896 offset=slave bundle=mem_weights_mm29 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=8
    #pragma HLS interface m_axi port=weight_data_mm30 depth=43728896 offset=slave bundle=mem_weights_mm30 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=8
    #pragma HLS interface m_axi port=weight_data_mm31 depth=43728896 offset=slave bundle=mem_weights_mm31 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=8
    /* step 4 Stage B: the 15 activation/state buffers are packed into this one
     * workspace pointer (GDN_WS_OFF_* layout in gdn_model.h), replacing 15 m_axi
     * ports and their control_s_axi base-address registers. Read+write, HBM0. */
    #pragma HLS interface m_axi port=workspace depth=13084960 offset=slave bundle=mem_weights_mm0 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=64 max_write_burst_length=64 num_write_outstanding=64
    #pragma HLS interface s_axilite port=return
    /* All projection shapes must time-share the one routed 32-reader,
     * 16-cluster engine. Local activation memories otherwise encourage HLS to
     * specialize one complete dataflow graph per input/output buffer size. */
    #pragma HLS allocation function instances=gdn_gemv limit=1

    /* step 4: fixed GDN-1.3B decode shape, one token per call. config/max_tokens/
     * num_tokens are gone from the signature; these constants replace them so the
     * synthesized loops have literal bounds and control_s_axi loses those regs. */
    const uint32_t hidden = GDN_HIDDEN;
    const uint32_t num_heads = GDN_HEADS;
    const uint32_t head_dim = GDN_HEAD_DIM;
    const uint32_t intermediate = GDN_INTER;
    const uint32_t num_tokens = 1;
    uint32_t layer_index;
    size_t mlp_count;
    const float *final_norm = aux_weights +
        (size_t)GDN_LAYERS * GDN_AUX_LAYER_STRIDE;

    /* Iter32: only persistent state and the host handoff remain in workspace.
     * Preserve every external offset so the committed host/ABI stays unchanged,
     * but keep the complete transient activation lifetime in six BRAM-backed,
     * 16-bank buffers. The two 5632-entry buffers hold q/k during attention and
     * are reused for the MLP gate/up vectors after recurrence consumes q/k. */
    float *workspace_x     = workspace + GDN_WS_OFF_X;
    float *workspace_out   = workspace + GDN_WS_OFF_X_NORM;
    float *recurrent_state = workspace + GDN_WS_OFF_REC_STATE;
    float *head_buffer     = workspace + GDN_WS_OFF_HEAD_BUF;

    Pack16 x_storage[GDN_HIDDEN / 16];
    Pack16 norm_attn_storage[GDN_HIDDEN / 16];
    Pack16 q_mlp_gate_storage[GDN_INTER / 16];
    Pack16 k_mlp_up_storage[GDN_INTER / 16];
    Pack16 v_storage[GDN_INTER / 16];
    Pack16 gate_storage[GDN_HIDDEN / 16];
    Pack16 gemv_in_storage[GDN_INTER / 16];
    Pack16 gemv_out_storage[GDN_INTER / 16];
    float a_storage[16];
    float b_storage[16];
#pragma HLS bind_storage variable=x_storage type=ram_2p impl=bram
#pragma HLS bind_storage variable=norm_attn_storage type=ram_2p impl=bram
#pragma HLS bind_storage variable=q_mlp_gate_storage type=ram_2p impl=bram
#pragma HLS bind_storage variable=k_mlp_up_storage type=ram_2p impl=bram
#pragma HLS bind_storage variable=v_storage type=ram_2p impl=bram
#pragma HLS bind_storage variable=gate_storage type=ram_2p impl=bram
#pragma HLS bind_storage variable=gemv_in_storage type=ram_2p impl=bram
#pragma HLS bind_storage variable=gemv_out_storage type=ram_2p impl=bram
#pragma HLS array_partition variable=a_storage complete dim=1
#pragma HLS array_partition variable=b_storage complete dim=1

    float *a = a_storage;
    float *b = b_storage;

    mlp_count = (size_t)num_tokens * intermediate;

    /* Compact-shard geometry (Pack16 units): each gemv projection's output stripe
     * (out_dim/GEMV_CHANNELS rows) occupies stripe_packs in every shard, packed
     * per layer in the order q,k,v,gate,o,mlp_gate,mlp_up,mlp_down — matching
     * gdn_build_weight_shards. */
    size_t shard_hh = (size_t)(hidden / GEMV_CHANNELS) * (hidden / 16);
    size_t shard_ih = (size_t)(intermediate / GEMV_CHANNELS) * (hidden / 16);
    size_t shard_di = (size_t)(hidden / GEMV_CHANNELS) * (intermediate / 16);
    size_t shard_per_layer = 5 * shard_hh + 2 * shard_ih + shard_di;

    /* One 8 KiB host-to-kernel handoff per token. All following activation
     * traffic stays on chip until the final one-line token result. */
    {
        const Pack16 *workspace_x16 =
            reinterpret_cast<const Pack16 *>(workspace_x);
    load_embedding_local: for (uint32_t p = 0; p < GDN_HIDDEN / 16; ++p) {
#pragma HLS loop_tripcount min=128 max=128
#pragma HLS pipeline II=1
            x_storage[p] = workspace_x16[p];
        }
    }

    layer_loop: for (layer_index = 0; layer_index < GDN_LAYERS; ++layer_index) {
    #pragma HLS loop_tripcount min=24 max=24  /* num_layers=24 */
        size_t layer_offset = (size_t)layer_index * GDN_AUX_LAYER_STRIDE;
        const float *layer_attn_norm = aux_weights + layer_offset;
        const float *layer_a_log;
        const float *layer_dt_bias;
        const float *layer_a_proj;
        const float *layer_b_proj;
        const float *layer_q_conv;
        const float *layer_k_conv;
        const float *layer_v_conv;
        const float *layer_o_norm;
        const float *layer_mlp_norm;

        /* Non-GEMV tensors are packed contiguously in aux_weights. */
        layer_offset += hidden;                          /* past attn_norm */
        layer_a_log = aux_weights + layer_offset;
        layer_offset += num_heads;
        layer_dt_bias = aux_weights + layer_offset;
        layer_offset += num_heads;
        layer_a_proj = aux_weights + layer_offset;
        layer_offset += (size_t)num_heads * hidden;
        layer_b_proj = aux_weights + layer_offset;
        layer_offset += (size_t)num_heads * hidden;
        layer_q_conv = aux_weights + layer_offset;
        layer_offset += (size_t)hidden * GDN_CONV;
        layer_k_conv = aux_weights + layer_offset;
        layer_offset += (size_t)hidden * GDN_CONV;
        layer_v_conv = aux_weights + layer_offset;
        layer_offset += (size_t)hidden * GDN_CONV;
        layer_o_norm = aux_weights + layer_offset;
        layer_offset += head_dim;
        layer_mlp_norm = aux_weights + layer_offset;

        /* Running compact-shard offset (Pack16); order q,k,v,gate,o,mlp_gate,
         * mlp_up,mlp_down — matches gdn_build_weight_shards. */
        size_t soff = (size_t)layer_index * shard_per_layer;

        gdn_rmsnorm_rows(norm_attn_storage, x_storage, layer_attn_norm,
                         num_tokens, hidden, GDN_NORM_EPS);
        gdn_pack16_copy_local(gemv_in_storage, norm_attn_storage,
                              hidden / 16);
        gdn_gemv(gemv_out_storage, gemv_in_storage,
                 GDN_GEMV_SHARD_ARGUMENTS,
                 (uint32_t)soff, num_tokens, hidden, hidden);
        gdn_pack16_copy_local(q_mlp_gate_storage, gemv_out_storage,
                              hidden / 16);
        soff += shard_hh;
        gdn_gemv(gemv_out_storage, gemv_in_storage,
                 GDN_GEMV_SHARD_ARGUMENTS,
                 (uint32_t)soff, num_tokens, hidden, hidden);
        gdn_pack16_copy_local(k_mlp_up_storage, gemv_out_storage,
                              hidden / 16);
        soff += shard_hh;
        gdn_gemv(gemv_out_storage, gemv_in_storage,
                 GDN_GEMV_SHARD_ARGUMENTS,
                 (uint32_t)soff, num_tokens, hidden, hidden);
        gdn_pack16_copy_local(v_storage, gemv_out_storage, hidden / 16);
        soff += shard_hh;
        gdn_gemv_tiny(a, norm_attn_storage, layer_a_proj,
                      num_tokens, hidden, num_heads);
        gdn_gemv_tiny(b, norm_attn_storage, layer_b_proj,
                      num_tokens, hidden, num_heads);
        gdn_gemv(gemv_out_storage, gemv_in_storage,
                 GDN_GEMV_SHARD_ARGUMENTS,
                 (uint32_t)soff, num_tokens, hidden, hidden);
        gdn_pack16_copy_local(gate_storage, gemv_out_storage, hidden / 16);
        soff += shard_hh;

        /* Per-(layer, conv) slice of the persistent conv tail in head_buffer:
         * 3 convs/layer × (conv_size-1) rows × hidden floats. */
        size_t tail_stride = (size_t)(GDN_CONV - 1) * hidden;
        float *q_tail = head_buffer + ((size_t)layer_index * 3 + 0) * tail_stride;
        float *k_tail = head_buffer + ((size_t)layer_index * 3 + 1) * tail_stride;
        float *v_tail = head_buffer + ((size_t)layer_index * 3 + 2) * tail_stride;

        gdn_depthwise_conv_silu(q_mlp_gate_storage, q_mlp_gate_storage,
                                layer_q_conv, q_tail, num_tokens, hidden,
                                GDN_CONV);
        gdn_depthwise_conv_silu(k_mlp_up_storage, k_mlp_up_storage,
                                layer_k_conv, k_tail, num_tokens, hidden,
                                GDN_CONV);
        gdn_depthwise_conv_silu(v_storage, v_storage, layer_v_conv, v_tail,
                                num_tokens, hidden, GDN_CONV);

        gdn_recurrent_attention(
            norm_attn_storage,
            recurrent_state,
            head_buffer,
            q_mlp_gate_storage,
            k_mlp_up_storage,
            v_storage,
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
        gdn_output_norm_and_gate(norm_attn_storage, gate_storage,
                                 layer_o_norm, num_tokens, num_heads,
                                 head_dim, GDN_NORM_EPS);
        gdn_pack16_copy_local(gemv_in_storage, norm_attn_storage,
                              hidden / 16);
        gdn_gemv(gemv_out_storage, gemv_in_storage,
                 GDN_GEMV_SHARD_ARGUMENTS,
                 (uint32_t)soff, num_tokens, hidden, hidden);
        gdn_pack16_add_local(x_storage, gemv_out_storage, hidden / 16);
        soff += shard_hh;

        gdn_rmsnorm_rows(norm_attn_storage, x_storage, layer_mlp_norm,
                         num_tokens, hidden, GDN_NORM_EPS);
        gdn_pack16_copy_local(gemv_in_storage, norm_attn_storage,
                              hidden / 16);
        gdn_gemv(gemv_out_storage, gemv_in_storage,
                 GDN_GEMV_SHARD_ARGUMENTS,
                 (uint32_t)soff, num_tokens, hidden, intermediate);
        gdn_pack16_copy_local(q_mlp_gate_storage, gemv_out_storage,
                              intermediate / 16);
        soff += shard_ih;
        gdn_gemv(gemv_out_storage, gemv_in_storage,
                 GDN_GEMV_SHARD_ARGUMENTS,
                 (uint32_t)soff, num_tokens, hidden, intermediate);
        gdn_pack16_copy_local(k_mlp_up_storage, gemv_out_storage,
                              intermediate / 16);
        soff += shard_ih;
        gdn_swiglu_inplace(q_mlp_gate_storage, k_mlp_up_storage, mlp_count);
        gdn_pack16_copy_local(gemv_in_storage, q_mlp_gate_storage,
                              intermediate / 16);
        gdn_gemv(gemv_out_storage, gemv_in_storage,
                 GDN_GEMV_SHARD_ARGUMENTS,
                 (uint32_t)soff, num_tokens, intermediate, hidden);
        gdn_pack16_add_local(x_storage, gemv_out_storage, hidden / 16);
    }

    gdn_rmsnorm_rows(norm_attn_storage, x_storage, final_norm,
                     num_tokens, hidden, GDN_NORM_EPS);
    /* The LM-head store reduces its existing reorder buffer directly to argmax;
     * no 32,000-float logits tensor is materialized in HBM. */
    {
        size_t lm_soff = (size_t)GDN_LAYERS * shard_per_layer;
        gdn_pack16_copy_local(gemv_in_storage, norm_attn_storage,
                              hidden / 16);
        gdn_gemv(gemv_out_storage, gemv_in_storage,
                 GDN_GEMV_SHARD_ARGUMENTS,
                 (uint32_t)lm_soff, num_tokens, hidden, GDN_VOCAB);
    }

    /* Preserve the old x_norm handoff offset. Write one full 512-bit line so
     * the host's existing workspace sync/read path needs no change. */
    {
        Pack16 token_line;
    token_line_init: for (int lane = 0; lane < 16; ++lane) {
#pragma HLS unroll
            token_line.data[lane] = 0.0f;
        }
        token_line.data[0] = gemv_out_storage[0].data[0];
        reinterpret_cast<Pack16 *>(workspace_out)[0] = token_line;
    }
    return 0;
}

#undef GDN_GEMV_SHARD_ARGUMENTS

/* Decode-only host entry: forward exactly one token against the persistent
 * per-layer recurrent + conv state held in the run-state buffers (loaded from
 * the GPU .gdnstate export). gdn_forward is decode-only — it restores each
 * layer's state at the start and saves the update at the end. */
int gdn_decode_step_host(const GDNModel *model, GDNRunState *state, const int32_t *token) {
    if (*token < 0 || (uint32_t)*token >= model->config.vocab_size) {
        gdn_print_error("token id out of range");
        return -1;
    }
    memcpy(state->x,
           model->embeddings + (size_t)*token * model->config.hidden_size,
           (size_t)model->config.hidden_size * sizeof(float));
    return gdn_forward(
        state->aux_weights,
        state->workspace,
        state->weight_shards[0],  state->weight_shards[1],
        state->weight_shards[2],  state->weight_shards[3],
        state->weight_shards[4],  state->weight_shards[5],
        state->weight_shards[6],  state->weight_shards[7],
        state->weight_shards[8],  state->weight_shards[9],
        state->weight_shards[10], state->weight_shards[11],
        state->weight_shards[12], state->weight_shards[13],
        state->weight_shards[14], state->weight_shards[15],
        state->weight_shards[16], state->weight_shards[17],
        state->weight_shards[18], state->weight_shards[19],
        state->weight_shards[20], state->weight_shards[21],
        state->weight_shards[22], state->weight_shards[23],
        state->weight_shards[24], state->weight_shards[25],
        state->weight_shards[26], state->weight_shards[27],
        state->weight_shards[28], state->weight_shards[29],
        state->weight_shards[30], state->weight_shards[31]
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


/* Decode-only GEMV constants shared by the routed 32-port implementation. */
#define GEMV_PARTIAL  4      /* power of two; >= FP32 fadd latency (4 cyc) in cycles */
#define IN_DIM_MAX    5632   /* max in_dim (intermediate=5632) — sizes a_loc */
/* GEMV_CHANNELS lives in gdn_model.h (shared by the kernel and the host). */

#if 0 /* Retired 8-port implementation; retained temporarily for reference. */
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

    /* FLATTENED MAC: one continuous II=8 pipeline over the whole (stripe x k_packs)
     * shard, so the read/MAC pipeline never goes cold between output rows. Previously
     * the row loop (gemv_pe_o) was sequential and ran the inner k-pipeline start→drain
     * per row, paying the ~73-cycle reduction-tree fill on EVERY row; that per-row fill
     * (plus ~19-cycle reduce/emit) was the 92-cycle/row overhead that capped port BW at
     * ~58% (k_packs=128) and ~79% (k_packs=352): eff = k_packs/(k_packs+92). Flattened,
     * the fill is paid ONCE per call → eff ≈ stripe*k_packs/(stripe*k_packs+92) ~99%.
     *
     * Each accumulator part[buf][p] keeps a COMPILE-TIME-FIXED p index → GEMV_PARTIAL
     * independent fadd chains, each retiring its carried fadd over a full 8-cycle outer
     * iteration (1 pack/cycle, robust to a multi-cycle fabric fadd at a faster clock).
     * The runtime `cur` only selects between two register banks (ping-pong), it does NOT
     * shorten the per-chain recurrence — distinct from the old part[kp & 7] runtime index
     * that created a distance-1 dependency and inflated II.
     *
     * Ping-pong + one-row-deferred emit: a completed row is read for reduce/emit only
     * after a full next row has elapsed, so its partials are fully retired AND live in
     * the OTHER buffer (cur ^ 1) than the row currently accumulating → no row-boundary
     * RAW stall, II stays 8. Bit-exact: part[p] still sums the kp ≡ p (mod 8) lanes in
     * kp order, and the emit keeps the same ((p0+p1)+(p2+p3))+((p4+p5)+(p6+p7)) tree.
     * k_packs is a multiple of GEMV_PARTIAL for all shapes (128, 352). */
    uint32_t groups_per_row = k_packs / GEMV_PARTIAL;
    uint32_t total_groups   = stripe * groups_per_row;

    float part[2][GEMV_PARTIAL];
    #pragma HLS array_partition variable=part complete dim=0

    uint32_t g_in_row = 0;       /* group index within the current output row */
    uint32_t a_base   = 0;       /* = g_in_row * GEMV_PARTIAL (pack base into a_loc) */
    uint32_t cur      = 0;       /* ping-pong buffer for the row being accumulated */
    bool     have_prev = false;  /* a completed row awaits emit in buffer (cur ^ 1) */

    gemv_pe_flat: for (uint32_t g = 0; g < total_groups; ++g) {
    #pragma HLS loop_tripcount min=4096 max=64000
    #pragma HLS pipeline II=8
        bool row_start = (g_in_row == 0);
        bool row_end   = (g_in_row == groups_per_row - 1);

        gemv_pe_p: for (int p = 0; p < GEMV_PARTIAL; ++p) {
        #pragma HLS unroll
            Pack16 w = wf.read();
            float lane = 0.0f;
            /* Reduce in LUT fabric, not DSP: keeps the full_dsp fp-adders out of the
             * (DSP-dense) multiply region. */
            #pragma HLS bind_op variable=lane op=fadd impl=fabric
            gemv_pe_lane: for (int kk = 0; kk < 16; ++kk) {
            #pragma HLS unroll
                lane += w.data[kk] * a_loc[(a_base + (uint32_t)p) * 16 + kk];
            }
            part[cur][p] = (row_start ? 0.0f : part[cur][p]) + lane;
        }

        if (row_end) {
            if (have_prev) {
                /* buffer (cur ^ 1) holds the row completed one row ago — retired */
                float s0 = part[cur ^ 1][0] + part[cur ^ 1][1];
                float s1 = part[cur ^ 1][2] + part[cur ^ 1][3];
                float s2 = part[cur ^ 1][4] + part[cur ^ 1][5];
                float s3 = part[cur ^ 1][6] + part[cur ^ 1][7];
                of.write((s0 + s1) + (s2 + s3));
            }
            have_prev = true;
            cur ^= 1;
            g_in_row = 0;
            a_base   = 0;
        } else {
            g_in_row += 1;
            a_base   += GEMV_PARTIAL;
        }
    }

    /* drain: emit the final completed row (left in buffer cur ^ 1) */
    if (have_prev) {
        float s0 = part[cur ^ 1][0] + part[cur ^ 1][1];
        float s1 = part[cur ^ 1][2] + part[cur ^ 1][3];
        float s2 = part[cur ^ 1][4] + part[cur ^ 1][5];
        float s3 = part[cur ^ 1][6] + part[cur ^ 1][7];
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
#endif

/* 32-port GEMV topology: 32 MM2S readers feed sixteen two-port compute
 * clusters. The smaller clusters preserve one weight beat per port per cycle
 * while reducing each independently placeable FP32 block by roughly half.
 * Activations ripple through one BRAM copy per cluster, and results merge
 * through SLR-local collectors. */
#define GEMV32_MAX_RESULT_PACKS 2048

static float gemv32_dot16(const Pack16 &w, const Pack16 &xv) {
#pragma HLS inline
    float prod[16];
#pragma HLS array_partition variable=prod complete
gemv32_dot_mul: for (int i = 0; i < 16; ++i) {
#pragma HLS unroll
        prod[i] = w.data[i] * xv.data[i];
#pragma HLS bind_op variable=prod op=fmul impl=maxdsp
    }
    float s0 = prod[0] + prod[1];
    float s1 = prod[2] + prod[3];
    float s2 = prod[4] + prod[5];
    float s3 = prod[6] + prod[7];
    float s4 = prod[8] + prod[9];
    float s5 = prod[10] + prod[11];
    float s6 = prod[12] + prod[13];
    float s7 = prod[14] + prod[15];
#pragma HLS bind_op variable=s0 op=fadd impl=fulldsp
#pragma HLS bind_op variable=s1 op=fadd impl=fulldsp
#pragma HLS bind_op variable=s2 op=fadd impl=fulldsp
#pragma HLS bind_op variable=s3 op=fadd impl=fulldsp
#pragma HLS bind_op variable=s4 op=fadd impl=fulldsp
#pragma HLS bind_op variable=s5 op=fadd impl=fulldsp
#pragma HLS bind_op variable=s6 op=fadd impl=fulldsp
#pragma HLS bind_op variable=s7 op=fadd impl=fulldsp
    float a0 = s0 + s1;
    float a1 = s2 + s3;
    float a2 = s4 + s5;
    float a3 = s6 + s7;
    float b0 = a0 + a1;
    float b1 = a2 + a3;
#pragma HLS bind_op variable=a0 op=fadd impl=fulldsp
#pragma HLS bind_op variable=a1 op=fadd impl=fulldsp
#pragma HLS bind_op variable=a2 op=fadd impl=fulldsp
#pragma HLS bind_op variable=a3 op=fadd impl=fulldsp
#pragma HLS bind_op variable=b0 op=fadd impl=fulldsp
#pragma HLS bind_op variable=b1 op=fadd impl=fulldsp
    return b0 + b1;
}

static float gemv32_reduce_part(float part[2][GEMV_PARTIAL], uint32_t bank) {
#pragma HLS inline
#if GEMV_PARTIAL == 8
    float s0 = part[bank][0] + part[bank][1];
    float s1 = part[bank][2] + part[bank][3];
    float s2 = part[bank][4] + part[bank][5];
    float s3 = part[bank][6] + part[bank][7];
    return (s0 + s1) + (s2 + s3);
#elif GEMV_PARTIAL == 4
    float s0 = part[bank][0] + part[bank][1];
    float s1 = part[bank][2] + part[bank][3];
    return s0 + s1;
#else
#error "gemv32_reduce_part: add a balanced tree for this GEMV_PARTIAL"
#endif
}

static void gemv32_load_x_and_w0(const Pack16 *x, const Pack16 *w0,
                                 size_t weight_base,
                                 hls::stream<Pack16> &xr,
                                 hls::stream<Pack16> &ws0,
                                 uint32_t k_packs, uint32_t n_packs) {
#pragma HLS inline off
gemv32_lx: for (uint32_t kp = 0; kp < k_packs; ++kp) {
#pragma HLS loop_tripcount min=128 max=352
#pragma HLS pipeline II=1
        xr.write(x[kp]);
    }
gemv32_w0: for (uint32_t i = 0; i < n_packs; ++i) {
#pragma HLS loop_tripcount min=8192 max=720896
#pragma HLS pipeline II=1
        ws0.write(w0[weight_base + i]);
    }
}

template <int CHANNEL>
static void gemv32_mm2s(const Pack16 *w, size_t base,
                        hls::stream<Pack16> &ws, uint32_t n_packs) {
#pragma HLS inline off
    (void)CHANNEL;
gemv32_mm2s_loop: for (uint32_t i = 0; i < n_packs; ++i) {
#pragma HLS loop_tripcount min=8192 max=720896
#pragma HLS pipeline II=1
        ws.write(w[base + i]);
    }
}

static void gemv32_drain_x(hls::stream<Pack16> &xr, uint32_t k_packs) {
#pragma HLS inline off
gemv32_dx: for (uint32_t kp = 0; kp < k_packs; ++kp) {
#pragma HLS loop_tripcount min=128 max=352
#pragma HLS pipeline II=1
        (void)xr.read();
    }
}

static void gemv32_cluster2(hls::stream<Pack16> &ws0,
                            hls::stream<Pack16> &ws1,
                            hls::stream<Pack16> &x_in,
                            hls::stream<Pack16> &x_out,
                            hls::stream<Pack16> &ys,
                            uint32_t k_packs, uint32_t rows_per_ch) {
#pragma HLS inline off
    float xbuf[IN_DIM_MAX];
#pragma HLS array_partition variable=xbuf cyclic factor=16
#pragma HLS bind_storage variable=xbuf type=ram_2p impl=bram
gemv32_cl_load: for (uint32_t kp = 0; kp < k_packs; ++kp) {
#pragma HLS loop_tripcount min=128 max=352
#pragma HLS pipeline II=1
        Pack16 v = x_in.read();
        x_out.write(v);
    gemv32_cl_load_lane: for (int lane = 0; lane < 16; ++lane) {
#pragma HLS unroll
            xbuf[kp * 16 + (uint32_t)lane] = v.data[lane];
        }
    }

    float part0[2][GEMV_PARTIAL], part1[2][GEMV_PARTIAL];
#pragma HLS array_partition variable=part0 complete dim=0
#pragma HLS array_partition variable=part1 complete dim=0
#pragma HLS bind_op variable=part0 op=fadd impl=fulldsp
#pragma HLS bind_op variable=part1 op=fadd impl=fulldsp
    Pack16 yp0, yp1;
#pragma HLS array_partition variable=yp0.data complete
#pragma HLS array_partition variable=yp1.data complete

    uint32_t groups_per_row = k_packs / GEMV_PARTIAL;
    uint32_t total_groups = rows_per_ch * groups_per_row;
    uint32_t row = 0, g_in_row = 0, a_base = 0, cur = 0;
    bool have_prev = false;

gemv32_cl_flat: for (uint32_t g = 0; g < total_groups; ++g) {
#pragma HLS loop_tripcount min=2048 max=180224
#pragma HLS pipeline II=GEMV_PARTIAL
        bool row_start = (g_in_row == 0);
        bool row_end = (g_in_row == groups_per_row - 1);
    gemv32_cl_p: for (int p = 0; p < GEMV_PARTIAL; ++p) {
#pragma HLS unroll
            Pack16 xv;
#pragma HLS array_partition variable=xv.data complete
        gemv32_cl_x_lane: for (int lane = 0; lane < 16; ++lane) {
#pragma HLS unroll
                xv.data[lane] = xbuf[(a_base + (uint32_t)p) * 16 + (uint32_t)lane];
            }
            Pack16 wv0 = ws0.read();
            Pack16 wv1 = ws1.read();
            float d0 = gemv32_dot16(wv0, xv);
            float d1 = gemv32_dot16(wv1, xv);
            part0[cur][p] = (row_start ? 0.0f : part0[cur][p]) + d0;
            part1[cur][p] = (row_start ? 0.0f : part1[cur][p]) + d1;
        }
        if (row_end) {
            if (have_prev) {
                uint32_t er = row - 1;
                yp0.data[er & 15] = gemv32_reduce_part(part0, cur ^ 1);
                yp1.data[er & 15] = gemv32_reduce_part(part1, cur ^ 1);
                if ((er & 15) == 15) {
                    ys.write(yp0);
                    ys.write(yp1);
                }
            }
            have_prev = true;
            cur ^= 1;
            row++;
            g_in_row = 0;
            a_base = 0;
        } else {
            g_in_row++;
            a_base += GEMV_PARTIAL;
        }
    }
    if (have_prev) {
        uint32_t er = rows_per_ch - 1;
        yp0.data[er & 15] = gemv32_reduce_part(part0, cur ^ 1);
        yp1.data[er & 15] = gemv32_reduce_part(part1, cur ^ 1);
        ys.write(yp0);
        ys.write(yp1);
    }
}

static void gemv32_collect6(hls::stream<Pack16> &ys0,
                            hls::stream<Pack16> &ys1,
                            hls::stream<Pack16> &ys2,
                            hls::stream<Pack16> &ys3,
                            hls::stream<Pack16> &ys4,
                            hls::stream<Pack16> &ys5,
                            hls::stream<Pack16> &local,
                            uint32_t opacks_per_ch) {
#pragma HLS inline off
gemv32_c6_p: for (uint32_t p = 0; p < opacks_per_ch; ++p) {
#pragma HLS loop_tripcount min=4 max=63
    gemv32_c6_a: for (int i = 0; i < 2; ++i) {
#pragma HLS pipeline II=1
            local.write(ys0.read());
        }
    gemv32_c6_b: for (int i = 0; i < 2; ++i) {
#pragma HLS pipeline II=1
            local.write(ys1.read());
        }
    gemv32_c6_c: for (int i = 0; i < 2; ++i) {
#pragma HLS pipeline II=1
            local.write(ys2.read());
        }
    gemv32_c6_d: for (int i = 0; i < 2; ++i) {
#pragma HLS pipeline II=1
            local.write(ys3.read());
        }
    gemv32_c6_e: for (int i = 0; i < 2; ++i) {
#pragma HLS pipeline II=1
            local.write(ys4.read());
        }
    gemv32_c6_f: for (int i = 0; i < 2; ++i) {
#pragma HLS pipeline II=1
            local.write(ys5.read());
        }
    }
}

static void gemv32_collect4(hls::stream<Pack16> &ys0,
                            hls::stream<Pack16> &ys1,
                            hls::stream<Pack16> &ys2,
                            hls::stream<Pack16> &ys3,
                            hls::stream<Pack16> &local,
                            uint32_t opacks_per_ch) {
#pragma HLS inline off
gemv32_c4_p: for (uint32_t p = 0; p < opacks_per_ch; ++p) {
#pragma HLS loop_tripcount min=4 max=63
    gemv32_c4_a: for (int i = 0; i < 2; ++i) {
#pragma HLS pipeline II=1
            local.write(ys0.read());
        }
    gemv32_c4_b: for (int i = 0; i < 2; ++i) {
#pragma HLS pipeline II=1
            local.write(ys1.read());
        }
    gemv32_c4_c: for (int i = 0; i < 2; ++i) {
#pragma HLS pipeline II=1
            local.write(ys2.read());
        }
    gemv32_c4_d: for (int i = 0; i < 2; ++i) {
#pragma HLS pipeline II=1
            local.write(ys3.read());
        }
    }
}

static void gemv32_collect_final(hls::stream<Pack16> &slr0,
                                 hls::stream<Pack16> &slr1,
                                 hls::stream<Pack16> &slr2,
                                 hls::stream<Pack16> &result,
                                 uint32_t opacks_per_ch) {
#pragma HLS inline off
gemv32_cf_p: for (uint32_t p = 0; p < opacks_per_ch; ++p) {
#pragma HLS loop_tripcount min=4 max=63
    gemv32_cf_0: for (int i = 0; i < 8; ++i) {
#pragma HLS pipeline II=1
            result.write(slr0.read());
        }
    gemv32_cf_1: for (int i = 0; i < 12; ++i) {
#pragma HLS pipeline II=1
            result.write(slr1.read());
        }
    gemv32_cf_2: for (int i = 0; i < 12; ++i) {
#pragma HLS pipeline II=1
            result.write(slr2.read());
        }
    }
}

/* The routed microbenchmark emits pack-major/channel-minor results. Full GDN
 * requires ordinary row-major activations, so buffer the small result tensor
 * in URAM and restore the original layout. The scalar fallback handles the
 * lm_head's 1000-row channel stripes, which are not Pack16 aligned. */
static void gemv32_store(hls::stream<Pack16> &result, Pack16 *out,
                         uint32_t rows_per_ch, uint32_t opacks_per_ch,
                         uint32_t total_opacks) {
#pragma HLS inline off
    Pack16 reorder[GEMV32_MAX_RESULT_PACKS];
#pragma HLS bind_storage variable=reorder type=ram_2p impl=uram
gemv32_store_fill: for (uint32_t i = 0; i < total_opacks; ++i) {
#pragma HLS loop_tripcount min=128 max=2016
#pragma HLS pipeline II=1
        reorder[i] = result.read();
    }

    if (rows_per_ch == GDN_VOCAB / GEMV_CHANNELS) {
        /* Read every reordered Pack16 exactly once. A scalar natural-order
         * scan makes HLS reread the same URAM word for all 16 lanes and
         * schedules at II=9. These fully partitioned lane winners preserve
         * strict-'>' comparisons; the final index-aware merge retains the
         * natural scan's first-index tie breaking across lanes. */
        float lane_best[16];
        uint32_t lane_best_index[16];
#pragma HLS array_partition variable=lane_best complete dim=1
#pragma HLS array_partition variable=lane_best_index complete dim=1
    gemv32_argmax_init: for (uint32_t lane = 0; lane < 16; ++lane) {
#pragma HLS unroll
            lane_best[lane] = -3.402823466e38f;
            lane_best_index[lane] = 0;
        }
    gemv32_argmax_c: for (uint32_t c = 0; c < GEMV_CHANNELS; ++c) {
        gemv32_argmax_p: for (uint32_t p = 0;
                              p < opacks_per_ch; ++p) {
#pragma HLS loop_tripcount min=63 max=63
#pragma HLS pipeline II=1
                Pack16 value = reorder[(size_t)p * GEMV_CHANNELS + c];
            gemv32_argmax_lane: for (uint32_t lane = 0;
                                     lane < 16; ++lane) {
#pragma HLS unroll
                uint32_t r = (p << 4) + lane;
                float candidate = value.data[lane];
                if (r < rows_per_ch && candidate > lane_best[lane]) {
                    lane_best[lane] = candidate;
                    lane_best_index[lane] = c * rows_per_ch + r;
                }
            }
        }
        }
        float best = -3.402823466e38f;
        uint32_t best_index = 0;
    gemv32_argmax_merge: for (uint32_t lane = 0; lane < 16; ++lane) {
            float candidate = lane_best[lane];
            uint32_t candidate_index = lane_best_index[lane];
            if (candidate > best ||
                (candidate == best && candidate_index < best_index)) {
                best = candidate;
                best_index = candidate_index;
            }
        }
        out[0].data[0] = (float)best_index;
        return;
    }

    if ((rows_per_ch & 15) == 0) {
    gemv32_store_c: for (uint32_t c = 0; c < GEMV_CHANNELS; ++c) {
        gemv32_store_p: for (uint32_t p = 0; p < opacks_per_ch; ++p) {
#pragma HLS loop_tripcount min=4 max=11
#pragma HLS pipeline II=1
                Pack16 value = reorder[(size_t)p * GEMV_CHANNELS + c];
                size_t out_index = (size_t)c * opacks_per_ch + p;
                out[out_index] = value;
            }
        }
    } else {
    gemv32_store_scalar_c: for (uint32_t c = 0; c < GEMV_CHANNELS; ++c) {
        gemv32_store_scalar_r: for (uint32_t r = 0; r < rows_per_ch; ++r) {
#pragma HLS loop_tripcount min=1000 max=1000
#pragma HLS pipeline II=1
                Pack16 v = reorder[(size_t)(r >> 4) * GEMV_CHANNELS + c];
                size_t out_index = (size_t)c * rows_per_ch + r;
                size_t out_pack = out_index >> 4;
                uint32_t out_lane = out_index & 15;
                out[out_pack].data[out_lane] = v.data[r & 15];
            }
        }
    }
}

/* Decode GEMV with 32 compact weight shards on independent HBM masters. Sixteen
 * two-port clusters consume private activation copies and feed hierarchical
 * collectors. The store stage restores natural output-row order; it also
 * handles the lm_head's partial final pack (1000 rows per channel). */
static void gdn_gemv(
    Pack16 *out, const Pack16 *in,
    const float *w0, const float *w1, const float *w2, const float *w3,
    const float *w4, const float *w5, const float *w6, const float *w7,
    const float *w8, const float *w9, const float *w10, const float *w11,
    const float *w12, const float *w13, const float *w14, const float *w15,
    const float *w16, const float *w17, const float *w18, const float *w19,
    const float *w20, const float *w21, const float *w22, const float *w23,
    const float *w24, const float *w25, const float *w26, const float *w27,
    const float *w28, const float *w29, const float *w30, const float *w31,
    uint32_t shard_off,
    uint32_t num_rows, uint32_t in_dim, uint32_t out_dim
) {
    #pragma HLS inline off

    const Pack16 *sh0 = reinterpret_cast<const Pack16 *>(w0);
    const Pack16 *sh1 = reinterpret_cast<const Pack16 *>(w1);
    const Pack16 *sh2 = reinterpret_cast<const Pack16 *>(w2);
    const Pack16 *sh3 = reinterpret_cast<const Pack16 *>(w3);
    const Pack16 *sh4 = reinterpret_cast<const Pack16 *>(w4);
    const Pack16 *sh5 = reinterpret_cast<const Pack16 *>(w5);
    const Pack16 *sh6 = reinterpret_cast<const Pack16 *>(w6);
    const Pack16 *sh7 = reinterpret_cast<const Pack16 *>(w7);
    const Pack16 *sh8 = reinterpret_cast<const Pack16 *>(w8);
    const Pack16 *sh9 = reinterpret_cast<const Pack16 *>(w9);
    const Pack16 *sh10 = reinterpret_cast<const Pack16 *>(w10);
    const Pack16 *sh11 = reinterpret_cast<const Pack16 *>(w11);
    const Pack16 *sh12 = reinterpret_cast<const Pack16 *>(w12);
    const Pack16 *sh13 = reinterpret_cast<const Pack16 *>(w13);
    const Pack16 *sh14 = reinterpret_cast<const Pack16 *>(w14);
    const Pack16 *sh15 = reinterpret_cast<const Pack16 *>(w15);
    const Pack16 *sh16 = reinterpret_cast<const Pack16 *>(w16);
    const Pack16 *sh17 = reinterpret_cast<const Pack16 *>(w17);
    const Pack16 *sh18 = reinterpret_cast<const Pack16 *>(w18);
    const Pack16 *sh19 = reinterpret_cast<const Pack16 *>(w19);
    const Pack16 *sh20 = reinterpret_cast<const Pack16 *>(w20);
    const Pack16 *sh21 = reinterpret_cast<const Pack16 *>(w21);
    const Pack16 *sh22 = reinterpret_cast<const Pack16 *>(w22);
    const Pack16 *sh23 = reinterpret_cast<const Pack16 *>(w23);
    const Pack16 *sh24 = reinterpret_cast<const Pack16 *>(w24);
    const Pack16 *sh25 = reinterpret_cast<const Pack16 *>(w25);
    const Pack16 *sh26 = reinterpret_cast<const Pack16 *>(w26);
    const Pack16 *sh27 = reinterpret_cast<const Pack16 *>(w27);
    const Pack16 *sh28 = reinterpret_cast<const Pack16 *>(w28);
    const Pack16 *sh29 = reinterpret_cast<const Pack16 *>(w29);
    const Pack16 *sh30 = reinterpret_cast<const Pack16 *>(w30);
    const Pack16 *sh31 = reinterpret_cast<const Pack16 *>(w31);

    uint32_t k_packs      = in_dim / 16;
    uint32_t rows_per_ch  = out_dim / GEMV_CHANNELS;
    uint32_t opacks_per_ch = (rows_per_ch + 15) >> 4;
    uint32_t n_packs      = rows_per_ch * k_packs;
    uint32_t total_opacks = opacks_per_ch * GEMV_CHANNELS;

    hls::stream<Pack16> ws[GEMV_CHANNELS];
    hls::stream<Pack16> xr[GEMV_CLUSTERS + 1];
    hls::stream<Pack16> ys[GEMV_CLUSTERS];
    hls::stream<Pack16> slr0_result, slr1_result, slr2_result, result;
    #pragma HLS array_partition variable=ws complete
    #pragma HLS array_partition variable=xr complete
    #pragma HLS array_partition variable=ys complete
    /* iter16: these 69 512-bit FIFOs were impl=lutram at depth 16/4, costing
     * 60,063 LUT + 111,729 FF of CLB resources -- and CLB in SLR0 is the ONLY
     * congested resource (SLR0 99.7%, and every congested window in iter15's
     * router log lies inside SLR0). BRAM is at 35% (637 of 1,816), so this
     * trades the scarce resource for the abundant one.
     *
     * The routed 32-port microbenchmark (microbench/gemv_tile) does exactly
     * this -- depth 64, impl=bram -- and is the only 32-port GEMV on this
     * device that routes with zero errors; it carries 96.88% BRAM / 90.47% CLB
     * in SLR1. Ours is the same engine with an aux path bolted on.
     *
     * Depth 64 (not 16/4) because BRAM cost is width-dominated: a 512-bit FIFO
     * needs ~8 RAMB36 for width whatever its depth, so depth is nearly free
     * here. That is the opposite of LUTRAM, where depth-4 and depth-16 measured
     * 1,547 vs 1,552 FF -- 0.3% apart. Expect ~550 extra BRAM tiles. */
    #pragma HLS stream variable=ws depth=64
    #pragma HLS stream variable=xr depth=64
    #pragma HLS stream variable=ys depth=64
    #pragma HLS stream variable=slr0_result depth=64
    #pragma HLS stream variable=slr1_result depth=64
    #pragma HLS stream variable=slr2_result depth=64
    #pragma HLS stream variable=result depth=64
    #pragma HLS bind_storage variable=ws type=fifo impl=bram
    #pragma HLS bind_storage variable=xr type=fifo impl=bram
    #pragma HLS bind_storage variable=ys type=fifo impl=bram
    #pragma HLS bind_storage variable=slr0_result type=fifo impl=bram
    #pragma HLS bind_storage variable=slr1_result type=fifo impl=bram
    #pragma HLS bind_storage variable=slr2_result type=fifo impl=bram
    #pragma HLS bind_storage variable=result type=fifo impl=bram

    #pragma HLS dataflow disable_start_propagation
    gemv32_load_x_and_w0(in, sh0, shard_off, xr[0], ws[0],
                         k_packs, n_packs);
    gemv32_mm2s<1>(sh1, shard_off, ws[1], n_packs);
    gemv32_mm2s<2>(sh2, shard_off, ws[2], n_packs);
    gemv32_mm2s<3>(sh3, shard_off, ws[3], n_packs);
    gemv32_mm2s<4>(sh4, shard_off, ws[4], n_packs);
    gemv32_mm2s<5>(sh5, shard_off, ws[5], n_packs);
    gemv32_mm2s<6>(sh6, shard_off, ws[6], n_packs);
    gemv32_mm2s<7>(sh7, shard_off, ws[7], n_packs);
    gemv32_mm2s<8>(sh8, shard_off, ws[8], n_packs);
    gemv32_mm2s<9>(sh9, shard_off, ws[9], n_packs);
    gemv32_mm2s<10>(sh10, shard_off, ws[10], n_packs);
    gemv32_mm2s<11>(sh11, shard_off, ws[11], n_packs);
    gemv32_mm2s<12>(sh12, shard_off, ws[12], n_packs);
    gemv32_mm2s<13>(sh13, shard_off, ws[13], n_packs);
    gemv32_mm2s<14>(sh14, shard_off, ws[14], n_packs);
    gemv32_mm2s<15>(sh15, shard_off, ws[15], n_packs);
    gemv32_mm2s<16>(sh16, shard_off, ws[16], n_packs);
    gemv32_mm2s<17>(sh17, shard_off, ws[17], n_packs);
    gemv32_mm2s<18>(sh18, shard_off, ws[18], n_packs);
    gemv32_mm2s<19>(sh19, shard_off, ws[19], n_packs);
    gemv32_mm2s<20>(sh20, shard_off, ws[20], n_packs);
    gemv32_mm2s<21>(sh21, shard_off, ws[21], n_packs);
    gemv32_mm2s<22>(sh22, shard_off, ws[22], n_packs);
    gemv32_mm2s<23>(sh23, shard_off, ws[23], n_packs);
    gemv32_mm2s<24>(sh24, shard_off, ws[24], n_packs);
    gemv32_mm2s<25>(sh25, shard_off, ws[25], n_packs);
    gemv32_mm2s<26>(sh26, shard_off, ws[26], n_packs);
    gemv32_mm2s<27>(sh27, shard_off, ws[27], n_packs);
    gemv32_mm2s<28>(sh28, shard_off, ws[28], n_packs);
    gemv32_mm2s<29>(sh29, shard_off, ws[29], n_packs);
    gemv32_mm2s<30>(sh30, shard_off, ws[30], n_packs);
    gemv32_mm2s<31>(sh31, shard_off, ws[31], n_packs);

    gemv32_cluster2(ws[0],  ws[1],  xr[0],  xr[1],  ys[0],  k_packs, rows_per_ch);
    gemv32_cluster2(ws[2],  ws[3],  xr[1],  xr[2],  ys[1],  k_packs, rows_per_ch);
    gemv32_cluster2(ws[4],  ws[5],  xr[2],  xr[3],  ys[2],  k_packs, rows_per_ch);
    gemv32_cluster2(ws[6],  ws[7],  xr[3],  xr[4],  ys[3],  k_packs, rows_per_ch);
    gemv32_cluster2(ws[8],  ws[9],  xr[4],  xr[5],  ys[4],  k_packs, rows_per_ch);
    gemv32_cluster2(ws[10], ws[11], xr[5],  xr[6],  ys[5],  k_packs, rows_per_ch);
    gemv32_cluster2(ws[12], ws[13], xr[6],  xr[7],  ys[6],  k_packs, rows_per_ch);
    gemv32_cluster2(ws[14], ws[15], xr[7],  xr[8],  ys[7],  k_packs, rows_per_ch);
    gemv32_cluster2(ws[16], ws[17], xr[8],  xr[9],  ys[8],  k_packs, rows_per_ch);
    gemv32_cluster2(ws[18], ws[19], xr[9],  xr[10], ys[9],  k_packs, rows_per_ch);
    gemv32_cluster2(ws[20], ws[21], xr[10], xr[11], ys[10], k_packs, rows_per_ch);
    gemv32_cluster2(ws[22], ws[23], xr[11], xr[12], ys[11], k_packs, rows_per_ch);
    gemv32_cluster2(ws[24], ws[25], xr[12], xr[13], ys[12], k_packs, rows_per_ch);
    gemv32_cluster2(ws[26], ws[27], xr[13], xr[14], ys[13], k_packs, rows_per_ch);
    gemv32_cluster2(ws[28], ws[29], xr[14], xr[15], ys[14], k_packs, rows_per_ch);
    gemv32_cluster2(ws[30], ws[31], xr[15], xr[16], ys[15], k_packs, rows_per_ch);
    gemv32_drain_x(xr[16], k_packs);
    gemv32_collect4(ys[0], ys[1], ys[2], ys[3],
                    slr0_result, opacks_per_ch);
    gemv32_collect6(ys[4], ys[5], ys[6], ys[7], ys[8], ys[9],
                    slr1_result, opacks_per_ch);
    gemv32_collect6(ys[10], ys[11], ys[12], ys[13], ys[14], ys[15],
                    slr2_result, opacks_per_ch);
    gemv32_collect_final(slr0_result, slr1_result, slr2_result,
                         result, opacks_per_ch);
    gemv32_store(result, out, rows_per_ch, opacks_per_ch, total_opacks);
}
