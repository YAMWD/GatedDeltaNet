#include "gdn_model.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    uint32_t ctx_len;
    uint32_t cont_len;
    int32_t *ctx;
    int32_t *cont;
} PairReq;

typedef struct {
    uint32_t gold;
    uint32_t num_choices;
    PairReq *choices;
} MCReq;

typedef struct {
    uint32_t word_count;
    uint32_t byte_count;
    uint32_t num_windows;
    PairReq *windows;
} RollingReq;

typedef struct {
    uint32_t kind;
    uint32_t num_examples;
    PairReq *ll_examples;
    MCReq *mc_examples;
    RollingReq *rolling_examples;
} Fixture;

enum {
    REQ_KIND_MC = 1,
    REQ_KIND_LL = 2,
    REQ_KIND_ROLLING = 3,
};

typedef struct {
    time_t start_time;
    uint32_t total_examples;
    uint32_t total_windows;
    uint32_t completed_examples;
    uint32_t completed_windows;
} ProgressState;

static void die(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(1);
}

static void *xmalloc(size_t bytes) {
    void *ptr = malloc(bytes);
    if (ptr == NULL) {
        die("malloc failed");
    }
    return ptr;
}

static void *xcalloc(size_t count, size_t size) {
    void *ptr = calloc(count, size);
    if (ptr == NULL) {
        die("calloc failed");
    }
    return ptr;
}

static uint32_t read_u32(const uint8_t *blob, size_t size, size_t *offset) {
    uint32_t value;
    if (*offset + 4 > size) {
        die("fixture truncated");
    }
    memcpy(&value, blob + *offset, 4);
    *offset += 4;
    return value;
}

static int32_t *read_i32_array(const uint8_t *blob, size_t size, size_t *offset, uint32_t count) {
    int32_t *out = NULL;
    if (*offset + (size_t)count * 4 > size) {
        die("fixture truncated");
    }
    if (count > 0) {
        out = (int32_t *)xmalloc((size_t)count * sizeof(int32_t));
        memcpy(out, blob + *offset, (size_t)count * 4);
    }
    *offset += (size_t)count * 4;
    return out;
}

static void load_fixture(const char *path, Fixture *fixture) {
    FILE *file;
    long file_size_long;
    size_t file_size;
    uint8_t *blob;
    size_t offset;
    uint32_t version;

    memset(fixture, 0, sizeof(*fixture));

    file = fopen(path, "rb");
    if (file == NULL) {
        perror("fopen fixture");
        exit(1);
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        perror("fseek fixture");
        fclose(file);
        exit(1);
    }
    file_size_long = ftell(file);
    if (file_size_long < 0) {
        perror("ftell fixture");
        fclose(file);
        exit(1);
    }
    file_size = (size_t)file_size_long;
    if (fseek(file, 0, SEEK_SET) != 0) {
        perror("fseek fixture");
        fclose(file);
        exit(1);
    }

    blob = (uint8_t *)xmalloc(file_size);
    if (fread(blob, 1, file_size, file) != file_size) {
        perror("fread fixture");
        fclose(file);
        free(blob);
        exit(1);
    }
    fclose(file);

    if (file_size < 20 || memcmp(blob, "GDNREQ1", 7) != 0) {
        free(blob);
        die("unsupported fixture file");
    }

    offset = 8;
    version = read_u32(blob, file_size, &offset);
    fixture->kind = read_u32(blob, file_size, &offset);
    fixture->num_examples = read_u32(blob, file_size, &offset);
    if (version != 1) {
        free(blob);
        die("unsupported fixture version");
    }

    if (fixture->kind == REQ_KIND_MC) {
        uint32_t example_index;
        fixture->mc_examples = (MCReq *)xcalloc(fixture->num_examples, sizeof(MCReq));
        for (example_index = 0; example_index < fixture->num_examples; ++example_index) {
            MCReq *req = &fixture->mc_examples[example_index];
            uint32_t choice_index;
            req->num_choices = read_u32(blob, file_size, &offset);
            req->gold = read_u32(blob, file_size, &offset);
            req->choices = (PairReq *)xcalloc(req->num_choices, sizeof(PairReq));
            for (choice_index = 0; choice_index < req->num_choices; ++choice_index) {
                PairReq *pair = &req->choices[choice_index];
                pair->ctx_len = read_u32(blob, file_size, &offset);
                pair->cont_len = read_u32(blob, file_size, &offset);
                pair->ctx = read_i32_array(blob, file_size, &offset, pair->ctx_len);
                pair->cont = read_i32_array(blob, file_size, &offset, pair->cont_len);
            }
        }
    } else if (fixture->kind == REQ_KIND_LL) {
        uint32_t example_index;
        fixture->ll_examples = (PairReq *)xcalloc(fixture->num_examples, sizeof(PairReq));
        for (example_index = 0; example_index < fixture->num_examples; ++example_index) {
            PairReq *pair = &fixture->ll_examples[example_index];
            pair->ctx_len = read_u32(blob, file_size, &offset);
            pair->cont_len = read_u32(blob, file_size, &offset);
            pair->ctx = read_i32_array(blob, file_size, &offset, pair->ctx_len);
            pair->cont = read_i32_array(blob, file_size, &offset, pair->cont_len);
        }
    } else if (fixture->kind == REQ_KIND_ROLLING) {
        uint32_t example_index;
        fixture->rolling_examples = (RollingReq *)xcalloc(fixture->num_examples, sizeof(RollingReq));
        for (example_index = 0; example_index < fixture->num_examples; ++example_index) {
            RollingReq *req = &fixture->rolling_examples[example_index];
            uint32_t window_index;
            req->word_count = read_u32(blob, file_size, &offset);
            req->byte_count = read_u32(blob, file_size, &offset);
            req->num_windows = read_u32(blob, file_size, &offset);
            req->windows = (PairReq *)xcalloc(req->num_windows, sizeof(PairReq));
            for (window_index = 0; window_index < req->num_windows; ++window_index) {
                PairReq *pair = &req->windows[window_index];
                pair->ctx_len = read_u32(blob, file_size, &offset);
                pair->cont_len = read_u32(blob, file_size, &offset);
                pair->ctx = read_i32_array(blob, file_size, &offset, pair->ctx_len);
                pair->cont = read_i32_array(blob, file_size, &offset, pair->cont_len);
            }
        }
    } else {
        free(blob);
        die("unsupported fixture kind");
    }

    free(blob);
}

static void free_fixture(Fixture *fixture) {
    if (fixture->kind == REQ_KIND_MC && fixture->mc_examples != NULL) {
        uint32_t example_index;
        for (example_index = 0; example_index < fixture->num_examples; ++example_index) {
            MCReq *req = &fixture->mc_examples[example_index];
            uint32_t choice_index;
            for (choice_index = 0; choice_index < req->num_choices; ++choice_index) {
                free(req->choices[choice_index].ctx);
                free(req->choices[choice_index].cont);
            }
            free(req->choices);
        }
        free(fixture->mc_examples);
    } else if (fixture->kind == REQ_KIND_LL && fixture->ll_examples != NULL) {
        uint32_t example_index;
        for (example_index = 0; example_index < fixture->num_examples; ++example_index) {
            free(fixture->ll_examples[example_index].ctx);
            free(fixture->ll_examples[example_index].cont);
        }
        free(fixture->ll_examples);
    } else if (fixture->kind == REQ_KIND_ROLLING && fixture->rolling_examples != NULL) {
        uint32_t example_index;
        for (example_index = 0; example_index < fixture->num_examples; ++example_index) {
            RollingReq *req = &fixture->rolling_examples[example_index];
            uint32_t window_index;
            for (window_index = 0; window_index < req->num_windows; ++window_index) {
                free(req->windows[window_index].ctx);
                free(req->windows[window_index].cont);
            }
            free(req->windows);
        }
        free(fixture->rolling_examples);
    }
    memset(fixture, 0, sizeof(*fixture));
}

static FILE *open_output(const char *path) {
    FILE *file;
    if (path == NULL) {
        return stdout;
    }
    file = fopen(path, "w");
    if (file == NULL) {
        perror("fopen output");
        exit(1);
    }
    return file;
}

static double elapsed_seconds(const ProgressState *progress) {
    return difftime(time(NULL), progress->start_time);
}

static void progress_start(ProgressState *progress, const Fixture *fixture) {
    uint32_t example_index;
    memset(progress, 0, sizeof(*progress));
    progress->start_time = time(NULL);
    progress->total_examples = fixture->num_examples;
    if (fixture->kind == REQ_KIND_ROLLING && fixture->rolling_examples != NULL) {
        for (example_index = 0; example_index < fixture->num_examples; ++example_index) {
            progress->total_windows += fixture->rolling_examples[example_index].num_windows;
        }
    }
}

static void log_progress_message(const char *message) {
    fprintf(stderr, "%s\n", message);
    fflush(stderr);
}

/* Standalone argmax over the logits for a single hidden row. Used by the
 * free-running greedy loop in --decode mode, where there is no target token
 * and no logprob/greedy bookkeeping to do. */
static int argmax_logits(const GDNModel *model, const float *hidden, float *logits) {
    uint32_t vocab = model->config.vocab_size;
    uint32_t vocab_index;
    float max_logit;
    int max_index;

    gdn_compute_logits(model, hidden, logits);

    max_logit = logits[0];
    max_index = 0;
    for (vocab_index = 1; vocab_index < vocab; ++vocab_index) {
        if (logits[vocab_index] > max_logit) {
            max_logit = logits[vocab_index];
            max_index = (int)vocab_index;
        }
    }
    return max_index;
}

static double monotonic_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

/* ===================================================================
 * Disaggregated decode: decode-only from a GPU-exported .gdnstate blob.
 * The GPU prefills the prompt (scripts/export_gdn_state.py) and dumps the
 * constant-size recurrent + conv state to disk; here we load it into the
 * run-state buffers and decode greedily from the exported seed token. No
 * prefill, no re-prefill. The emitted trajectory (seed + decoded) is checked
 * bit-exact against the cached golden by scripts/check_gdn_c_parity.py --decode.
 * =================================================================== */
typedef struct {
    uint32_t num_layers, num_heads, head_dim, value_dim, hidden, conv_size;
    uint32_t prompt_len;
    int32_t  seed_token;
} GDNStateHeader;

static void load_gdnstate(const char *path, const GDNModel *model,
                          GDNRunState *run_state, GDNStateHeader *hdr) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) die("cannot open .gdnstate file");
    char magic[8];
    uint32_t v[9];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "GDNSTAT1", 8) != 0)
        die(".gdnstate: bad magic (expected GDNSTAT1)");
    if (fread(v, sizeof(uint32_t), 9, f) != 9)
        die(".gdnstate: truncated header");
    /* v = {version, num_layers, H, K, V, hidden, W, prompt_len, seed_token} */
    hdr->num_layers = v[1]; hdr->num_heads = v[2]; hdr->head_dim = v[3];
    hdr->value_dim  = v[4]; hdr->hidden    = v[5]; hdr->conv_size = v[6];
    hdr->prompt_len = v[7]; hdr->seed_token = (int32_t)v[8];

    const GDNWeightHeader *c = &model->config;
    if (hdr->num_layers != c->num_layers || hdr->num_heads != c->num_heads ||
        hdr->head_dim != c->head_dim || hdr->hidden != c->hidden_size ||
        hdr->conv_size != c->conv_size)
        die(".gdnstate: dims do not match the loaded model");

    /* Skip prompt_ids (informational only). */
    if (fseek(f, (long)hdr->prompt_len * (long)sizeof(int32_t), SEEK_CUR) != 0)
        die(".gdnstate: cannot skip prompt ids");

    /* Section A: recurrent state -> run_state->recurrent_state (all layers). */
    size_t rec_count = (size_t)hdr->num_layers * hdr->num_heads *
                       hdr->head_dim * hdr->value_dim;
    if (fread(run_state->recurrent_state, sizeof(float), rec_count, f) != rec_count)
        die(".gdnstate: truncated recurrent section");
    /* Section B: conv tails -> run_state->head_buffer (layers x 3 x (W-1) x hidden). */
    size_t conv_count = (size_t)hdr->num_layers * 3u *
                        (hdr->conv_size - 1u) * hdr->hidden;
    if (fread(run_state->head_buffer, sizeof(float), conv_count, f) != conv_count)
        die(".gdnstate: truncated conv section");
    fclose(f);

    fprintf(stderr,
            "[progress] loaded .gdnstate: layers=%u H=%u K=%u V=%u hidden=%u W=%u "
            "prompt_len=%u seed=%d  (recurrent=%.1f MB conv=%.1f KB)\n",
            hdr->num_layers, hdr->num_heads, hdr->head_dim, hdr->value_dim,
            hdr->hidden, hdr->conv_size, hdr->prompt_len, hdr->seed_token,
            rec_count * sizeof(float) / 1e6, conv_count * sizeof(float) / 1e3);
    fflush(stderr);
}

static int run_decode_from_state(
    const GDNModel *model, GDNRunState *run_state, float *logits,
    const char *state_path, uint32_t decode_len, const char *output_path
) {
    GDNStateHeader hdr;
    load_gdnstate(state_path, model, run_state, &hdr);

    uint32_t n = (decode_len != 0) ? decode_len : 64;
    int32_t *traj = (int32_t *)xmalloc((size_t)n * sizeof(int32_t));
    double  *tpot = (double  *)xmalloc((size_t)n * sizeof(double));

    /* traj[0] = the GPU-exported seed (argmax of the prompt's last position);
     * each later token is one decode step against the persistent state. */
    traj[0] = hdr.seed_token;
    tpot[0] = 0.0;
    fprintf(stderr, "[progress] decode-from-state N=%u seed=%d\n", n, traj[0]);
    fflush(stderr);
    for (uint32_t step = 1; step < n; ++step) {
        double t0 = monotonic_ms();
        int32_t prev = traj[step - 1];
        if (gdn_decode_step_host(model, run_state, &prev) != 0)
            die("decode-from-state: single-token step failed");
        /* lm_head + greedy argmax now run on-chip (gdn_forward) and write the
         * next token id into x_norm[0] — read it directly instead of recomputing
         * host-side, so native matches the kernel exactly. */
        traj[step] = (int32_t)run_state->x_norm[0];
        (void)logits;
        tpot[step] = monotonic_ms() - t0;
    }

    FILE *out = open_output(output_path);
    fprintf(out, "{\"kind\": 2, \"decode_len\": %u, \"num_examples\": 1,\n", n);
    fprintf(out, " \"examples\": [\n  {\"index\": 0,\n   \"gen_traj\": [");
    for (uint32_t j = 0; j < n; ++j) fprintf(out, "%s%d", j ? ", " : "", traj[j]);
    fprintf(out, "],\n   \"tf_argmax\": [");
    for (uint32_t j = 0; j < n; ++j) fprintf(out, "%s%d", j ? ", " : "", traj[j]);
    fprintf(out, "],\n   \"per_step_tpot_ms\": [");
    for (uint32_t j = 0; j < n; ++j) fprintf(out, "%s%.6f", j ? ", " : "", tpot[j]);
    fprintf(out, "]}\n ]}\n");
    if (output_path != NULL) fclose(out);

    free(traj);
    free(tpot);
    return 0;
}

int main(int argc, char **argv) {
    GDNModel model;
    GDNRunState run_state;
    Fixture fixture;
    ProgressState progress;
    float *logits;
    int decode_mode = 0;
    const char *state_path = NULL;  /* --decode-from-state: disaggregated decode */
    uint32_t decode_len = 0;   /* 0 => use the fixture's golden cont_len */
    const char *positional[3];
    int positional_count = 0;
    int arg_index;

    /* Parse args: up to 3 positionals (weights, fixture, [output]) plus the
     * optional --decode flags. Without --decode, behavior is unchanged. */
    for (arg_index = 1; arg_index < argc; ++arg_index) {
        const char *arg = argv[arg_index];
        if (strcmp(arg, "--decode") == 0) {
            decode_mode = 1;
        } else if (strcmp(arg, "--decode-from-state") == 0) {
            if (arg_index + 1 >= argc) {
                die("--decode-from-state requires a .gdnstate path");
            }
            state_path = argv[++arg_index];
            decode_mode = 1;
        } else if (strcmp(arg, "--decode-len") == 0) {
            if (arg_index + 1 >= argc) {
                die("--decode-len requires a value");
            }
            decode_len = (uint32_t)strtoul(argv[++arg_index], NULL, 10);
        } else if (positional_count < 3) {
            positional[positional_count++] = arg;
        } else {
            fprintf(stderr, "unexpected argument: %s\n", arg);
            return 1;
        }
    }

    if (positional_count < 2) {
        fprintf(stderr,
                "usage (decode-only): %s <weights.gdnw> <fixture.gdnreq> [output.json]"
                " --decode --decode-from-state <state.gdnstate> [--decode-len N]\n",
                argv[0]);
        return 1;
    }
    {
        /* Rebuild the argc/argv contract the rest of main relies on: argv[1] =
         * weights, argv[2] = fixture, optional argv[3] = output. */
        static char *rebuilt[4];
        rebuilt[0] = argv[0];
        rebuilt[1] = (char *)positional[0];
        rebuilt[2] = (char *)positional[1];
        argc = positional_count + 1;
        argv = rebuilt;
        if (positional_count == 3) {
            rebuilt[3] = (char *)positional[2];
        }
    }

    log_progress_message("[progress] loading model weights");
    if (gdn_model_load(&model, argv[1]) != 0) {
        return 1;
    }
    log_progress_message("[progress] allocating run state");
    if (gdn_run_state_init(&run_state, &model, model.config.max_seq_len) != 0) {
        gdn_model_free(&model);
        return 1;
    }
    logits = (float *)xmalloc((size_t)model.config.vocab_size * sizeof(float));
    log_progress_message("[progress] loading fixture");
    load_fixture(argv[2], &fixture);
    progress_start(&progress, &fixture);
    fprintf(
        stderr,
        "[progress] starting evaluation kind=%u examples=%u windows=%u\n",
        fixture.kind,
        fixture.num_examples,
        progress.total_windows
    );
    fflush(stderr);

    /* Decode-only build: the FPGA never prefills. The post-prompt recurrent +
     * conv state is produced on the GPU (scripts/export_gdn_state.py) and loaded
     * from disk; decode runs token-by-token from the exported seed. */
    if (!decode_mode || state_path == NULL) {
        die("decode-only build: run with "
            "--decode --decode-from-state <state.gdnstate> [--decode-len N]");
    }
    {
        uint32_t n = decode_len;
        if (n == 0) {
            if (fixture.kind != REQ_KIND_LL || fixture.num_examples == 0 || fixture.ll_examples == NULL) {
                die("decode-from-state requires an LL-kind fixture with at least 1 example");
            }
            n = fixture.ll_examples[0].cont_len;
        }
        if (n == 0) die("decode-from-state: zero decode length");
        const char *decode_out = (argc == 4) ? argv[3] : "results_decode_c/decode.c.json";
        run_decode_from_state(&model, &run_state, logits, state_path, n, decode_out);
        fprintf(stderr, "[progress] decode finished elapsed=%.0fs\n", elapsed_seconds(&progress));
        fflush(stderr);
        free_fixture(&fixture);
        free(logits);
        gdn_run_state_free(&run_state);
        gdn_model_free(&model);
        return 0;
    }
}
