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

/* =========================================================================
 * Systolic-array matmul (architecture from spcl/gemm_hls).
 *
 *   ReadA ──a_pipes[0]──→ PE[0] ──a_pipes[1]──→ PE[1] ──→ ... ──→ PE[15]
 *   ReadB ──b_pipes[0]──→ PE[0] ──b_pipes[1]──→ PE[1] ──→ ... ──→ PE[15]
 *                            │                  │                   │
 *                                ────────  WriteC  ────────
 *
 * 2 chains × 16 PEs × 16 MAC = 512 MAC/cycle peak. Each PE owns one row of
 * the current outer-N tile and accumulates 16 PARTIAL lanes to break the
 * FP32 fadd recurrence at II=1 under a 4 ns clock. ReadA does an on-chip
 * transpose + broadcast to both chains; ReadB ping-pongs between two
 * tile-buffers so load(next) overlaps stream(cur) at II=1.
 *
 * Kernel constraints (gdn_matmul_systolic returns -k on violation):
 *   out_dim  % (NUM_CHAINS * M_PER_PE) (=32) == 0
 *   in_dim   % 16                       == 0
 *   in_dim   <= IN_DIM_MAX (=5632)
 *   num_rows is unconstrained — the kernel pads to the next multiple of
 *   N_PES on-chip (zero-fills A reads beyond num_rows, skips DRAM writes
 *   beyond num_rows; the c_streams are still drained to keep dataflow
 *   FIFOs balanced).
 *
 * IN_DIM_MAX is sized at 5632 so that mlp_down_proj (in_dim=5632) fits a
 * single kernel invocation — no host-side K-tiling is needed. q/k/v/g/o
 * (in_dim=2048) and mlp_gate/up (in_dim=2048) reuse the same kernel; only
 * the K-loop runtime changes.
 *
 * The dispatch gdn_matmul() below tries gdn_matmul_systolic first and
 * falls back to gdn_matmul_tiled() for shapes that don't fit (a/b_proj
 * have out_dim=8). The standalone matmul-only HLS top in
 * gdn_matmul_systolic.cpp keeps an authoritative copy of these helpers
 * for test_matmul.tcl; this in-file version exists so gdn_model.cpp is
 * self-contained and gdn_eval/gdn_attn_test don't need to link the
 * standalone source file. No malloc/free anywhere — same code path runs
 * under csim and csynth.
 * ======================================================================= */

#define N_PES 16
#define M_PER_PE 16
#define IN_DIM_MAX 5632
/* NUM_CHAINS=1 inside the integrated gdn_forward / gdn_attn_forward path
 * so the dataflow region has exactly one reader on the input bundle, one
 * reader on the weights bundle, and one writer on the output bundle —
 * required by HLS dataflow's single-reader/single-writer rule when the
 * call sites in gdn_attn_forward bind in/weights/out to single physical
 * AXI masters. The standalone matmul-only HLS top in
 * gdn_matmul_systolic.cpp keeps NUM_CHAINS=2 because it has 5 separate
 * m_axi ports (mem_in / mem_wt_lo / mem_wt_hi / mem_out_lo / mem_out_hi)
 * and can run two chains in parallel. */
#define NUM_CHAINS 1

struct Pack16 {
    float data[16];
};

typedef hls::stream<Pack16> Stream16;

static void ProcessingElement(
    Stream16 &aIn, Stream16 &aOut,
    Stream16 &bIn, Stream16 &bOut,
    Stream16 &cOut,
    int location,
    uint32_t num_outer_n,
    uint32_t num_outer_m,
    uint32_t in_dim
) {
    /* PARTIAL=16 lanes, recurrence-free at II=1 even at 320+ MHz Fmax. */
    static const int PARTIAL = 16;
    float c_buf[M_PER_PE][PARTIAL];
    #pragma HLS array_partition variable=c_buf complete

    pe_n0: for (uint32_t n0 = 0; n0 < num_outer_n; ++n0) {
    #pragma HLS loop_tripcount min=1 max=128
        pe_m0: for (uint32_t m0 = 0; m0 < num_outer_m; ++m0) {
        #pragma HLS loop_tripcount min=1 max=64

            pe_init_m: for (int m = 0; m < M_PER_PE; ++m) {
            #pragma HLS unroll
                pe_init_p: for (int p = 0; p < PARTIAL; ++p) {
                #pragma HLS unroll
                    c_buf[m][p] = 0.0f;
                }
            }

            pe_k: for (uint32_t k = 0; k < in_dim; ++k) {
            #pragma HLS loop_tripcount min=1 max=5632
            #pragma HLS pipeline II=1
                Pack16 a_pack = aIn.read();
                float a_val = a_pack.data[location];
                aOut.write(a_pack);

                Pack16 b_pack = bIn.read();
                bOut.write(b_pack);

                uint32_t lane = k & (PARTIAL - 1);
                pe_mac: for (int m = 0; m < M_PER_PE; ++m) {
                #pragma HLS unroll
                    c_buf[m][lane] += a_val * b_pack.data[m];
                }
            }

            /* Tree-reduce 16 lanes per row, emit one row pack. */
            Pack16 my_c;
            pe_pack: for (int m = 0; m < M_PER_PE; ++m) {
            #pragma HLS unroll
                float l0 = c_buf[m][0]  + c_buf[m][1];
                float l1 = c_buf[m][2]  + c_buf[m][3];
                float l2 = c_buf[m][4]  + c_buf[m][5];
                float l3 = c_buf[m][6]  + c_buf[m][7];
                float l4 = c_buf[m][8]  + c_buf[m][9];
                float l5 = c_buf[m][10] + c_buf[m][11];
                float l6 = c_buf[m][12] + c_buf[m][13];
                float l7 = c_buf[m][14] + c_buf[m][15];
                float q0 = l0 + l1;
                float q1 = l2 + l3;
                float q2 = l4 + l5;
                float q3 = l6 + l7;
                float h0 = q0 + q1;
                float h1 = q2 + q3;
                my_c.data[m] = h0 + h1;
            }
            cOut.write(my_c);
        }
    }
}

/* ReadA — loads outer-N stripes of A into BRAM and streams column packs
 * to the PE chain. Handles in-kernel token padding: when num_rows is not
 * a multiple of N_PES, the last outer-N stripe contains some padding rows.
 * For those, the m_axi address is clamped to row 0 (so DDR/HBM never sees
 * an OOB address) and the loaded data is zeroed in `a_buf` — the PE chain
 * then MACs against zero `a_val` for that row, which contributes nothing
 * to the output. WriteC_chain skips the DRAM write for those rows. */
static void ReadA(
    const Pack16 *a_wide,
    Stream16 &a_out,
    uint32_t num_rows,
    uint32_t num_outer_n,
    uint32_t in_dim,
    uint32_t out_dim
) {
    uint32_t num_outer_m = out_dim / M_PER_PE;
    uint32_t num_outer_m_per_chain = num_outer_m / NUM_CHAINS;
    uint32_t k_packs = in_dim / 16;

    static float a_buf[N_PES][IN_DIM_MAX];
    #pragma HLS array_partition variable=a_buf dim=1 complete
    #pragma HLS array_partition variable=a_buf dim=2 cyclic factor=16

    read_a_n0: for (uint32_t n0 = 0; n0 < num_outer_n; ++n0) {
    #pragma HLS loop_tripcount min=1 max=128

        load_a_rkp: for (uint32_t rkp = 0; rkp < (uint32_t)N_PES * k_packs; ++rkp) {
        #pragma HLS loop_tripcount min=2048 max=5632
        #pragma HLS pipeline II=1
            uint32_t r  = rkp / k_packs;
            uint32_t kp = rkp % k_packs;
            uint32_t global_row = n0 * N_PES + r;
            bool valid = (global_row < num_rows);
            /* Address-clamp: when this row is padding, read row 0 instead so
             * the m_axi master never sees an OOB address; we'll zero the
             * loaded pack below before storing to a_buf. */
            uint32_t safe_row = valid ? global_row : 0;
            Pack16 wide = a_wide[(size_t)safe_row * k_packs + kp];
            load_a_unpack: for (int kk = 0; kk < 16; ++kk) {
            #pragma HLS unroll
                a_buf[r][kp * 16 + kk] = valid ? wide.data[kk] : 0.0f;
            }
        }

        stream_a_m0: for (uint32_t m0 = 0; m0 < num_outer_m_per_chain; ++m0) {
        #pragma HLS loop_tripcount min=1 max=64
            stream_a_k: for (uint32_t k = 0; k < in_dim; ++k) {
            #pragma HLS loop_tripcount min=1 max=5632
            #pragma HLS pipeline II=1
                Pack16 col;
                stream_a_pack: for (int r = 0; r < N_PES; ++r) {
                #pragma HLS unroll
                    col.data[r] = a_buf[r][k];
                }
                a_out.write(col);
            }
        }
    }
}

static void ReadB(
    const Pack16 *b_wide,
    Stream16 &b_out,
    uint32_t num_outer_n,
    uint32_t in_dim,
    uint32_t m0_offset,
    uint32_t num_outer_m_per_chain
) {
    uint32_t k_packs = in_dim / 16;
    uint32_t total_tiles = num_outer_n * num_outer_m_per_chain;

    /* Function-local (NOT static): NUM_CHAINS ReadB invocations in dataflow
     * each need their own BRAM instance — a static array would be shared
     * state and trip "HLS 200-979 internal global variable failed dataflow
     * checking". */
    float b_buf[2][M_PER_PE][IN_DIM_MAX];
    #pragma HLS array_partition variable=b_buf dim=1 complete
    #pragma HLS array_partition variable=b_buf dim=2 complete
    #pragma HLS array_partition variable=b_buf dim=3 cyclic factor=16

    if (total_tiles == 0) return;

    preload_ckp: for (uint32_t ckp = 0; ckp < (uint32_t)M_PER_PE * k_packs; ++ckp) {
    #pragma HLS loop_tripcount min=2048 max=5632
    #pragma HLS pipeline II=1
        uint32_t c  = ckp / k_packs;
        uint32_t kp = ckp % k_packs;
        Pack16 wide = b_wide[(size_t)(m0_offset * M_PER_PE + c) * k_packs + kp];
        preload_unpack: for (int kk = 0; kk < 16; ++kk) {
        #pragma HLS unroll
            b_buf[0][c][kp * 16 + kk] = wide.data[kk];
        }
    }

    fused_tile: for (uint32_t tile = 1; tile < total_tiles; ++tile) {
    #pragma HLS loop_tripcount min=1 max=8192
        uint32_t stream_idx       = (tile - 1) & 1u;
        uint32_t load_idx         = tile & 1u;
        uint32_t m0_local_next    = tile % num_outer_m_per_chain;
        uint32_t m0_global_next   = m0_local_next + m0_offset;

        fused_io: for (uint32_t i = 0; i < in_dim; ++i) {
        #pragma HLS loop_tripcount min=1 max=5632
        #pragma HLS pipeline II=1
            Pack16 col;
            fused_stream_pack: for (int c = 0; c < M_PER_PE; ++c) {
            #pragma HLS unroll
                col.data[c] = b_buf[stream_idx][c][i];
            }
            b_out.write(col);

            uint32_t load_c  = i / k_packs;
            uint32_t load_kp = i % k_packs;
            Pack16 wide = b_wide[(size_t)(m0_global_next * M_PER_PE + load_c) * k_packs + load_kp];
            fused_load_unpack: for (int kk = 0; kk < 16; ++kk) {
            #pragma HLS unroll
                b_buf[load_idx][load_c][load_kp * 16 + kk] = wide.data[kk];
            }
        }
    }

    uint32_t last_idx = (total_tiles - 1) & 1u;
    last_stream: for (uint32_t k = 0; k < in_dim; ++k) {
    #pragma HLS loop_tripcount min=1 max=5632
    #pragma HLS pipeline II=1
        Pack16 col;
        last_stream_pack: for (int c = 0; c < M_PER_PE; ++c) {
        #pragma HLS unroll
            col.data[c] = b_buf[last_idx][c][k];
        }
        b_out.write(col);
    }
}

static void SinkAB(
    Stream16 &a_in,
    Stream16 &b_in,
    uint32_t num_outer_n,
    uint32_t num_outer_m,
    uint32_t in_dim
) {
    sink_n0: for (uint32_t n0 = 0; n0 < num_outer_n; ++n0) {
    #pragma HLS loop_tripcount min=1 max=128
        sink_m0: for (uint32_t m0 = 0; m0 < num_outer_m; ++m0) {
        #pragma HLS loop_tripcount min=1 max=64
            sink_k: for (uint32_t k = 0; k < in_dim; ++k) {
            #pragma HLS loop_tripcount min=1 max=5632
            #pragma HLS pipeline II=1
                a_in.read();
                b_in.read();
            }
        }
    }
}

static void WriteC_chain(
    Stream16 c_streams[N_PES],
    Pack16 *c_wide,
    uint32_t num_rows,
    uint32_t num_outer_n,
    uint32_t out_dim,
    uint32_t m0_offset,
    uint32_t num_outer_m_per_chain
) {
    uint32_t m_packs = out_dim / 16;

    write_c_n0: for (uint32_t n0 = 0; n0 < num_outer_n; ++n0) {
    #pragma HLS loop_tripcount min=1 max=128
        write_c_m0: for (uint32_t m0 = 0; m0 < num_outer_m_per_chain; ++m0) {
        #pragma HLS loop_tripcount min=1 max=64
            size_t base = (size_t)(n0 * N_PES) * m_packs + m0 + m0_offset;
            write_c_r: for (int r = 0; r < N_PES; ++r) {
            #pragma HLS pipeline II=1
                /* Always drain c_streams to keep dataflow FIFOs balanced —
                 * the PE chain wrote one Pack16 per (n0, m0, r) regardless
                 * of token padding. Skip the m_axi write for padded rows
                 * (global_row >= num_rows). */
                Pack16 row = c_streams[r].read();
                uint32_t global_row = n0 * N_PES + (uint32_t)r;
                if (global_row < num_rows) {
                    c_wide[base + (size_t)r * m_packs] = row;
                }
            }
        }
    }
}

/* run_dataflow lives in its own function so #pragma HLS dataflow is at
 * function scope (HLS 207-5556 forbids it inside `{ ... }`) and the body
 * stays canonical (only stream decls and function calls). */
static void run_dataflow(
    Pack16 *out,
    const Pack16 *in,
    const Pack16 *weights,
    uint32_t num_rows,
    uint32_t in_dim,
    uint32_t out_dim,
    uint32_t num_outer_n,
    uint32_t num_outer_m_per_chain
) {
    #pragma HLS dataflow

    Stream16 a_pipes[N_PES + 1];
    Stream16 b_pipes[N_PES + 1];
    Stream16 c_streams[N_PES];
    #pragma HLS stream variable=a_pipes   depth=4
    #pragma HLS stream variable=b_pipes   depth=4
    #pragma HLS stream variable=c_streams depth=32

    ReadA(in, a_pipes[0],
          num_rows, num_outer_n, in_dim, out_dim);

    ReadB(weights, b_pipes[0], num_outer_n, in_dim,
          0u, num_outer_m_per_chain);

    /* 16-PE chain. */
    ProcessingElement(a_pipes[0],  a_pipes[1],  b_pipes[0],  b_pipes[1],
                      c_streams[0],  0,  num_outer_n, num_outer_m_per_chain, in_dim);
    ProcessingElement(a_pipes[1],  a_pipes[2],  b_pipes[1],  b_pipes[2],
                      c_streams[1],  1,  num_outer_n, num_outer_m_per_chain, in_dim);
    ProcessingElement(a_pipes[2],  a_pipes[3],  b_pipes[2],  b_pipes[3],
                      c_streams[2],  2,  num_outer_n, num_outer_m_per_chain, in_dim);
    ProcessingElement(a_pipes[3],  a_pipes[4],  b_pipes[3],  b_pipes[4],
                      c_streams[3],  3,  num_outer_n, num_outer_m_per_chain, in_dim);
    ProcessingElement(a_pipes[4],  a_pipes[5],  b_pipes[4],  b_pipes[5],
                      c_streams[4],  4,  num_outer_n, num_outer_m_per_chain, in_dim);
    ProcessingElement(a_pipes[5],  a_pipes[6],  b_pipes[5],  b_pipes[6],
                      c_streams[5],  5,  num_outer_n, num_outer_m_per_chain, in_dim);
    ProcessingElement(a_pipes[6],  a_pipes[7],  b_pipes[6],  b_pipes[7],
                      c_streams[6],  6,  num_outer_n, num_outer_m_per_chain, in_dim);
    ProcessingElement(a_pipes[7],  a_pipes[8],  b_pipes[7],  b_pipes[8],
                      c_streams[7],  7,  num_outer_n, num_outer_m_per_chain, in_dim);
    ProcessingElement(a_pipes[8],  a_pipes[9],  b_pipes[8],  b_pipes[9],
                      c_streams[8],  8,  num_outer_n, num_outer_m_per_chain, in_dim);
    ProcessingElement(a_pipes[9],  a_pipes[10], b_pipes[9],  b_pipes[10],
                      c_streams[9],  9,  num_outer_n, num_outer_m_per_chain, in_dim);
    ProcessingElement(a_pipes[10], a_pipes[11], b_pipes[10], b_pipes[11],
                      c_streams[10], 10, num_outer_n, num_outer_m_per_chain, in_dim);
    ProcessingElement(a_pipes[11], a_pipes[12], b_pipes[11], b_pipes[12],
                      c_streams[11], 11, num_outer_n, num_outer_m_per_chain, in_dim);
    ProcessingElement(a_pipes[12], a_pipes[13], b_pipes[12], b_pipes[13],
                      c_streams[12], 12, num_outer_n, num_outer_m_per_chain, in_dim);
    ProcessingElement(a_pipes[13], a_pipes[14], b_pipes[13], b_pipes[14],
                      c_streams[13], 13, num_outer_n, num_outer_m_per_chain, in_dim);
    ProcessingElement(a_pipes[14], a_pipes[15], b_pipes[14], b_pipes[15],
                      c_streams[14], 14, num_outer_n, num_outer_m_per_chain, in_dim);
    ProcessingElement(a_pipes[15], a_pipes[16], b_pipes[15], b_pipes[16],
                      c_streams[15], 15, num_outer_n, num_outer_m_per_chain, in_dim);

    SinkAB(a_pipes[N_PES], b_pipes[N_PES],
           num_outer_n, num_outer_m_per_chain, in_dim);

    WriteC_chain(c_streams, out, num_rows, num_outer_n, out_dim,
                 0u, num_outer_m_per_chain);
}

/* gdn_matmul_systolic — float-pointer entry to the systolic kernel. No
 * malloc, no #ifdef: the same code path runs under both csim and csynth.
 *
 * Geometry checks:
 *   out_dim % (NUM_CHAINS * M_PER_PE) (=32) == 0  -- caller falls back to
 *                                                   gdn_matmul_tiled if not
 *   in_dim  % 16                       == 0      -- ditto
 *   in_dim  <= IN_DIM_MAX (=5632)               -- ditto (mlp_down fits)
 *
 * num_rows is unconstrained: ReadA zero-fills the loaded data when
 * global_row >= num_rows (with the m_axi address clamped to row 0 so
 * DRAM never sees an OOB address), and WriteC_chain skips the m_axi
 * write for global_row >= num_rows (it still drains c_streams to keep
 * the dataflow FIFOs balanced). The PE chain naturally produces zero
 * outputs for padded rows because every K-cycle MAC for those rows
 * uses a_val=0. */
static int gdn_matmul_systolic(
    float *out, const float *in, const float *weights,
    uint32_t num_rows, uint32_t in_dim, uint32_t out_dim
) {
    if (num_rows == 0 || in_dim == 0 || out_dim == 0)            return -1;
    if ((out_dim % (NUM_CHAINS * M_PER_PE)) != 0)                return -3;
    if ((in_dim  % 16)       != 0)                               return -4;
    if (in_dim > (uint32_t)IN_DIM_MAX)                           return -5;

    uint32_t num_outer_n            = (num_rows + N_PES - 1) / N_PES;
    uint32_t num_outer_m            = out_dim / M_PER_PE;
    uint32_t num_outer_m_per_chain  = num_outer_m / NUM_CHAINS;

    Pack16 *out_p           = reinterpret_cast<Pack16 *>(out);
    const Pack16 *in_p      = reinterpret_cast<const Pack16 *>(in);
    const Pack16 *weights_p = reinterpret_cast<const Pack16 *>(weights);

    run_dataflow(out_p, in_p, weights_p,
                 num_rows, in_dim, out_dim,
                 num_outer_n, num_outer_m_per_chain);

    return 0;
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

/* No dispatch wrapper here — call sites in gdn_forward / gdn_attn_forward
 * call gdn_matmul_systolic directly for systolic-eligible shapes
 * (out_dim %32==0 && in_dim %16==0 && in_dim<=5632) and gdn_matmul_tiled
 * directly for the small a/b_proj shapes (out_dim=8). A runtime dispatch
 * wrapper would force HLS to allocate hardware for both paths at every
 * call site and report sum(systolic + tiled) per call, which inflates
 * the latency estimate by ~30× for the systolic-eligible calls. Direct
 * calls let HLS report just the path actually used. */

/* Compile-time bounds for the conv buffers. The model always calls this with
 * hidden=2048 and conv_size=4; smaller calls still fit. */
#define GDN_CONV_COLS_MAX 2048
#define GDN_CONV_K_MAX    4

static void gdn_depthwise_conv_silu(
    float *out,
    const float *in,
    const float *weights,
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

    float in_window[GDN_CONV_K_MAX][GDN_CONV_COLS_MAX];
    #pragma HLS array_partition variable=in_window dim=1 complete

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

        conv_load: for (col = 0; col < num_cols; ++col) {
        #pragma HLS loop_tripcount min=2048 max=2048
        #pragma HLS pipeline II=1
            in_window[0][col] = in_window[1][col];
            in_window[1][col] = in_window[2][col];
            in_window[2][col] = in_window[3][col];
            in_window[3][col] = in[(size_t)row * num_cols + col];
        }

        conv_compute: for (col = 0; col < num_cols; ++col) {
        #pragma HLS loop_tripcount min=2048 max=2048
        #pragma HLS pipeline II=1
            /* in_window[k] holds source row (row - kernel_size + 1 + k). */
            float sum = in_window[0][col] * w_loc[col][0]
                      + in_window[1][col] * w_loc[col][1]
                      + in_window[2][col] * w_loc[col][2]
                      + in_window[3][col] * w_loc[col][3];
            out[(size_t)row * num_cols + col] = gdn_silu(sum);
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
    float *recurrent_state,  /* unused: state is now on-chip BRAM */
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
    uint32_t num_tokens
) {
    /* On-chip persistent recurrent state: 8 heads × 256 × 256 FP32 = 2 MB */
    static float state[GDN_HEADS][GDN_DK][GDN_DV];
#pragma HLS bind_storage variable=state type=RAM_2P impl=BRAM
#pragma HLS array_partition variable=state dim=3 cyclic factor=16

    float q_scale = 1.0f / sqrtf((float)GDN_DK);
    uint32_t h, j, i, p;
    uint32_t token_index;

    /* Clear on-chip state (parallelised by P_K) */
    state_clr_h: for (h = 0; h < GDN_HEADS; ++h) {
    #pragma HLS loop_tripcount min=8 max=8
        state_clr_j: for (j = 0; j < GDN_DK; ++j) {
        #pragma HLS loop_tripcount min=256 max=256
            state_clr_i: for (i = 0; i < GDN_DV; i += GDN_PK) {
            #pragma HLS loop_tripcount min=16 max=16
            #pragma HLS pipeline II=1
                uint32_t pp;
                for (pp = 0; pp < GDN_PK; ++pp) {
                #pragma HLS unroll
                    state[h][j][i + pp] = 0.0f;
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
    /* Pre-load the per-head norm weight (head_dim=256 floats) once and reuse
     * for every (token, head) pair — eliminates an AXI read per iteration. */
    float weight_loc[GDN_DV];
    #pragma HLS array_partition variable=weight_loc cyclic factor=8
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
            float *attn_head = attn + (size_t)token_index * num_heads * head_dim + (size_t)head_index * head_dim;
            const float *gate_head = gate + (size_t)token_index * num_heads * head_dim + (size_t)head_index * head_dim;

            /* On-chip buffers break the AXI read-after-write hazard on attn_head
             * (was II=160) and avoid a second AXI read of gate_head per element. */
            float attn_loc[GDN_DV];
            float gate_loc[GDN_DV];
            #pragma HLS array_partition variable=attn_loc cyclic factor=8
            #pragma HLS array_partition variable=gate_loc cyclic factor=8

            /* Per-element squares scratch -- store-products + tree-reduce
             * pattern (same as load_qk / dot_alpha). The earlier 8-lane mux
             * pattern stayed at II=2 because HLS muxed the partial array into
             * a single register and tracked the carried dep on the mux. */
            float sq_arr[GDN_DV];
            #pragma HLS array_partition variable=sq_arr complete

            /* Phase 1: load attn_head into local + record squared values */
            uint32_t index;
            onorm_sq: for (index = 0; index < head_dim; ++index) {
            #pragma HLS loop_tripcount min=256 max=256
            #pragma HLS pipeline II=1
                float v = attn_head[index];
                attn_loc[index] = v;
                sq_arr[index] = v * v;
            }

            /* Phase 2: load gate_head into local buffer */
            onorm_load_g: for (index = 0; index < head_dim; ++index) {
            #pragma HLS loop_tripcount min=256 max=256
            #pragma HLS pipeline II=1
                gate_loc[index] = gate_head[index];
            }

            float sum = gdn_tree_reduce_256(sq_arr);
            float scale = 1.0f / sqrtf(sum / (float)head_dim + eps);

            /* Phase 3: combine and write back (writes only on the AXI port) */
            onorm_gate: for (index = 0; index < head_dim; ++index) {
            #pragma HLS loop_tripcount min=256 max=256
            #pragma HLS pipeline II=1
                float normalized = attn_loc[index] * scale * weight_loc[index];
                float gate_value = gate_loc[index];
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
        gdn_matmul_systolic(q, x_norm, layer_q_proj, num_tokens, hidden, hidden);
        gdn_matmul_systolic(k, x_norm, layer_k_proj, num_tokens, hidden, hidden);
        gdn_matmul_systolic(v, x_norm, layer_v_proj, num_tokens, hidden, hidden);
        gdn_matmul_tiled(a, x_norm, layer_a_proj, num_tokens, hidden, num_heads);
        gdn_matmul_tiled(b, x_norm, layer_b_proj, num_tokens, hidden, num_heads);
        gdn_matmul_systolic(gate, x_norm, layer_g_proj, num_tokens, hidden, hidden);

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
        gdn_matmul_systolic(tmp_hidden, attn, layer_o_proj, num_tokens, hidden, hidden);
        attn_residual: for (index = 0; index < hidden_count; ++index) {
        #pragma HLS loop_tripcount min=2048 max=4194304  /* hidden_count = num_tokens*hidden: 1*2048 to 2048*2048 */
            x[index] += tmp_hidden[index];
        }

        gdn_rmsnorm_rows(x_norm, x, layer_mlp_norm, num_tokens, hidden, config->norm_eps);
        gdn_matmul_systolic(mlp_gate, x_norm, layer_mlp_gate_proj, num_tokens, hidden, intermediate);
        gdn_matmul_systolic(mlp_up, x_norm, layer_mlp_up_proj, num_tokens, hidden, intermediate);
        gdn_swiglu_inplace(mlp_gate, mlp_up, mlp_count);
        gdn_matmul_systolic(tmp_hidden, mlp_gate, layer_mlp_down_proj, num_tokens, intermediate, hidden);
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

/* ---------------------------------------------------------------------------
 * HLS-synthesizable single-layer GDN attention forward.
 *
 * Struct pointers are disaggregated into flat arrays so Vitis HLS can attach
 * m_axi interfaces.  The host wrapper gdn_attn_forward_layer (below) packs
 * the call for testbench use.
 * ----------------------------------------------------------------------- */
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
) {
    /* m_axi depth sized for single-layer attention:
       weight_data covers layer 0 attn weights (vocab*hidden + layer_attn_stride ≈ 87M).
       Per-token buffers sized for max_seq_len*hidden = 2048*2048 = 4M. */
    /* AXI bundles:
     *   - q on bundle=mem_q and k on bundle=mem_k put the two parallel reads in
     *     gdn_recurrent_attention.load_qk on separate AXI master ports, lifting
     *     load_qk from II=2 (gmem port contention) to II=1.
     *   - All other ports stay on the default `gmem` bundle.
     */
    #pragma HLS interface m_axi port=config depth=1 offset=slave
    /* weight_data on its own bundle so the systolic ReadB task doesn't
     * collide with ReadA (which reads `input` or `attn`) on the default
     * gmem bundle — HLS dataflow requires distinct bundles per task. */
    #pragma HLS interface m_axi port=weight_data depth=87000000 offset=slave bundle=mem_weights
    #pragma HLS interface m_axi port=input depth=129024 offset=slave
    #pragma HLS interface m_axi port=output depth=129024 offset=slave
    #pragma HLS interface m_axi port=q depth=129024 offset=slave bundle=mem_q
    #pragma HLS interface m_axi port=k depth=129024 offset=slave bundle=mem_k
    #pragma HLS interface m_axi port=v depth=129024 offset=slave
    #pragma HLS interface m_axi port=a depth=504 offset=slave
    #pragma HLS interface m_axi port=b depth=504 offset=slave
    #pragma HLS interface m_axi port=gate depth=129024 offset=slave
    #pragma HLS interface m_axi port=attn depth=129024 offset=slave
    #pragma HLS interface m_axi port=tmp_hidden depth=129024 offset=slave
    #pragma HLS interface m_axi port=recurrent_state depth=524288 offset=slave
    #pragma HLS interface m_axi port=head_buffer depth=256 offset=slave
    #pragma HLS interface s_axilite port=layer_index
    #pragma HLS interface s_axilite port=num_tokens
    #pragma HLS interface s_axilite port=return

    uint32_t hidden = config->hidden_size;
    uint32_t num_heads = config->num_heads;
    uint32_t head_dim = config->head_dim;
    uint32_t conv_size = config->conv_size;
    size_t hidden_count = (size_t)num_tokens * hidden;
    size_t index;

    /* Compute weight pointers for the requested layer */
    size_t layer_offset = gdn_layer_weight_offset(config, layer_index);
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

    if (layer_index >= config->num_layers) {
        gdn_print_error("layer index out of range");
        return -1;
    }
    if (num_tokens == 0 || num_tokens > config->max_seq_len) {
        gdn_print_error("invalid token count for attn forward");
        return -1;
    }

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
    layer_offset += (size_t)hidden * conv_size;
    layer_k_conv = weight_data + layer_offset;
    layer_offset += (size_t)hidden * conv_size;
    layer_v_conv = weight_data + layer_offset;
    layer_offset += (size_t)hidden * conv_size;
    layer_g_proj = weight_data + layer_offset;
    layer_offset += (size_t)hidden * hidden;
    layer_o_norm = weight_data + layer_offset;
    layer_offset += head_dim;
    layer_o_proj = weight_data + layer_offset;

    /* Projections: input -> q, k, v, a, b, gate */
    gdn_matmul_systolic(q, input, layer_q_proj, num_tokens, hidden, hidden);
    gdn_matmul_systolic(k, input, layer_k_proj, num_tokens, hidden, hidden);
    gdn_matmul_systolic(v, input, layer_v_proj, num_tokens, hidden, hidden);
    gdn_matmul_tiled(a, input, layer_a_proj, num_tokens, hidden, num_heads);
    gdn_matmul_tiled(b, input, layer_b_proj, num_tokens, hidden, num_heads);
    gdn_matmul_systolic(gate, input, layer_g_proj, num_tokens, hidden, hidden);

    /* Depthwise conv1d + SiLU on q, k, v */
    gdn_depthwise_conv_silu(tmp_hidden, q, layer_q_conv, num_tokens, hidden, conv_size);
    attn_conv_copy_q: for (index = 0; index < hidden_count; ++index) {
    #pragma HLS loop_tripcount min=2048 max=4194304  /* hidden_count = num_tokens*hidden: 1*2048 to 2048*2048 */
        q[index] = tmp_hidden[index];
    }

    gdn_depthwise_conv_silu(tmp_hidden, k, layer_k_conv, num_tokens, hidden, conv_size);
    attn_conv_copy_k: for (index = 0; index < hidden_count; ++index) {
    #pragma HLS loop_tripcount min=2048 max=4194304  /* hidden_count = num_tokens*hidden: 1*2048 to 2048*2048 */
        k[index] = tmp_hidden[index];
    }

    gdn_depthwise_conv_silu(tmp_hidden, v, layer_v_conv, num_tokens, hidden, conv_size);
    attn_conv_copy_v: for (index = 0; index < hidden_count; ++index) {
    #pragma HLS loop_tripcount min=2048 max=4194304  /* hidden_count = num_tokens*hidden: 1*2048 to 2048*2048 */
        v[index] = tmp_hidden[index];
    }

    /* Recurrent attention */
    gdn_recurrent_attention(
        attn, recurrent_state, head_buffer,
        q, k, v, a, b,
        layer_a_log, layer_dt_bias,
        hidden, num_heads, head_dim, num_tokens
    );

    /* Output norm + gate */
    gdn_output_norm_and_gate(
        attn, gate, layer_o_norm,
        num_tokens, num_heads, head_dim, config->norm_eps
    );

    /* Output projection -> output */
    gdn_matmul_systolic(output, attn, layer_o_proj, num_tokens, hidden, hidden);

    return 0;
}

/* Host-side wrapper: packs GDNModel/GDNRunState into flat args for testbench */
int gdn_attn_forward_layer(
    const GDNModel *model,
    GDNRunState *state,
    uint32_t layer_index,
    const float *input,
    float *output,
    uint32_t num_tokens
) {
    return gdn_attn_forward(
        &model->config,
        model->weight_data,
        layer_index,
        input,
        output,
        state->q,
        state->k,
        state->v,
        state->a,
        state->b,
        state->gate,
        state->attn,
        state->tmp_hidden,
        state->recurrent_state,
        state->head_buffer,
        num_tokens
    );
}

/* gdn_matmul_top — definition lives in gdn_matmul_systolic.cpp (the wide-AXI
 * Pack16 systolic version, used as the standalone matmul-only HLS top by
 * test_matmul.tcl and gdn_matmul_test.cpp). The header keeps the prototype
 * for those callers; gdn_model.cpp uses the inlined systolic helpers above
 * via gdn_matmul → gdn_matmul_systolic instead. */
