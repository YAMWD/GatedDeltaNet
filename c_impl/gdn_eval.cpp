#include "gdn_model.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    uint32_t kind;
    uint32_t num_examples;
    uint32_t first_cont_len;
} Fixture;

#define REQ_KIND_LL 2

typedef struct {
    time_t start_time;
} ProgressState;

typedef struct {
    uint32_t checked_steps;
    uint64_t compared_values;
    uint64_t cpu_tolerance_failures;
    uint64_t exact_reference_mismatches;
    uint32_t argmax_mismatches;
    double max_abs_error;
    double max_rel_error;
} LogitsParity;

typedef struct {
    char magic[8];
    uint32_t version;
    uint32_t vocab_size;
    uint32_t decode_steps;
} LogitsDumpHeader;

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

static uint32_t read_u32(const uint8_t *blob, size_t size, size_t *offset) {
    uint32_t value;
    if (*offset + 4 > size) {
        die("fixture truncated");
    }
    memcpy(&value, blob + *offset, 4);
    *offset += 4;
    return value;
}

static void skip_i32_array(const uint8_t *blob, size_t size, size_t *offset, uint32_t count) {
    if (*offset + (size_t)count * 4 > size) {
        die("fixture truncated");
    }
    (void)blob;
    *offset += (size_t)count * 4;
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

    if (fixture->kind != REQ_KIND_LL) {
        free(blob);
        die("decode-only evaluator requires an LL fixture");
    }

    for (uint32_t example_index = 0;
         example_index < fixture->num_examples; ++example_index) {
        uint32_t ctx_len = read_u32(blob, file_size, &offset);
        uint32_t cont_len = read_u32(blob, file_size, &offset);
        if (example_index == 0) {
            fixture->first_cont_len = cont_len;
        }
        skip_i32_array(blob, file_size, &offset, ctx_len);
        skip_i32_array(blob, file_size, &offset, cont_len);
    }
    if (offset != file_size) {
        free(blob);
        die("unexpected trailing data in decode fixture");
    }
    free(blob);
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

static void progress_start(ProgressState *progress) {
    memset(progress, 0, sizeof(*progress));
    progress->start_time = time(NULL);
}

static void log_progress_message(const char *message) {
    fprintf(stderr, "%s\n", message);
    fflush(stderr);
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
    {
        float *state_stripes[GDN_RECURRENT_STATE_PORTS];
        const size_t shard_floats = gdn_weight_shard_floats(c);
        for (int p = 0; p < GDN_RECURRENT_STATE_PORTS; ++p) {
            state_stripes[p] =
                run_state->weight_shards[GDN_RECURRENT_STATE_FIRST_PORT + p] +
                shard_floats;
        }
        gdn_scatter_recurrent_state(state_stripes,
                                    run_state->recurrent_state, rec_count);
    }
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
    const char *state_path, uint32_t decode_len, const char *output_path,
    const char *logits_dump_path, const char *logits_reference_path
) {
    GDNStateHeader hdr;
    load_gdnstate(state_path, model, run_state, &hdr);

    uint32_t n = (decode_len != 0) ? decode_len : 64;
    int32_t *traj = (int32_t *)xmalloc((size_t)n * sizeof(int32_t));
    double  *tpot = (double  *)xmalloc((size_t)n * sizeof(double));
    float *final_hidden = (float *)xmalloc(
        (size_t)model->config.hidden_size * sizeof(float));
    float *reference_logits = (float *)xmalloc(
        (size_t)model->config.vocab_size * sizeof(float));
    float *saved_logits = NULL;
    FILE *logits_dump = NULL;
    FILE *logits_reference = NULL;
    LogitsParity logits_parity;
    memset(&logits_parity, 0, sizeof(logits_parity));

    if (logits_dump_path != NULL) {
        LogitsDumpHeader dump_header = {{'G','D','N','L','O','G','1','\0'}, 1,
                                        model->config.vocab_size, n - 1};
        logits_dump = fopen(logits_dump_path, "wb");
        if (logits_dump == NULL ||
            fwrite(&dump_header, sizeof(dump_header), 1, logits_dump) != 1) {
            die("decode-from-state: cannot create logits dump");
        }
    }
    if (logits_reference_path != NULL) {
        LogitsDumpHeader reference_header;
        logits_reference = fopen(logits_reference_path, "rb");
        if (logits_reference == NULL ||
            fread(&reference_header, sizeof(reference_header), 1,
                  logits_reference) != 1 ||
            memcmp(reference_header.magic, "GDNLOG1", 7) != 0 ||
            reference_header.version != 1 ||
            reference_header.vocab_size != model->config.vocab_size ||
            reference_header.decode_steps < n - 1) {
            die("decode-from-state: incompatible logits reference");
        }
        saved_logits = (float *)xmalloc(
            (size_t)model->config.vocab_size * sizeof(float));
    }

#ifndef __SYNTHESIS__
    gdn_set_native_debug_buffers(final_hidden, logits);
#endif

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
        gdn_compute_logits(model, final_hidden, reference_logits);
        uint32_t captured_argmax = 0;
        float captured_best = logits[0];
        for (uint32_t vocab_index = 0;
             vocab_index < model->config.vocab_size; ++vocab_index) {
            float captured = logits[vocab_index];
            float reference = reference_logits[vocab_index];
            double abs_error = fabs((double)captured - (double)reference);
            double rel_error = abs_error /
                fmax(1.0, fabs((double)reference));
            if (abs_error > logits_parity.max_abs_error)
                logits_parity.max_abs_error = abs_error;
            if (rel_error > logits_parity.max_rel_error)
                logits_parity.max_rel_error = rel_error;
            if (!isfinite(captured) || !isfinite(reference) ||
                abs_error > 1e-3 + 1e-4 * fabs((double)reference)) {
                logits_parity.cpu_tolerance_failures++;
            }
            if (vocab_index != 0 && captured > captured_best) {
                captured_best = captured;
                captured_argmax = vocab_index;
            }
        }
        logits_parity.checked_steps++;
        logits_parity.compared_values += model->config.vocab_size;
        if ((int32_t)captured_argmax != traj[step])
            logits_parity.argmax_mismatches++;

        if (logits_reference != NULL) {
            if (fread(saved_logits, sizeof(float), model->config.vocab_size,
                      logits_reference) != model->config.vocab_size) {
                die("decode-from-state: truncated logits reference");
            }
            for (uint32_t vocab_index = 0;
                 vocab_index < model->config.vocab_size; ++vocab_index) {
                if (memcmp(&saved_logits[vocab_index], &logits[vocab_index],
                           sizeof(float)) != 0) {
                    logits_parity.exact_reference_mismatches++;
                }
            }
        }
        if (logits_dump != NULL &&
            fwrite(logits, sizeof(float), model->config.vocab_size,
                   logits_dump) != model->config.vocab_size) {
            die("decode-from-state: failed while writing logits dump");
        }
        tpot[step] = monotonic_ms() - t0;
    }
#ifndef __SYNTHESIS__
    gdn_set_native_debug_buffers(NULL, NULL);
#endif

    fprintf(stderr,
            "[logits] steps=%u values=%llu max_abs=%.9g max_rel=%.9g "
            "cpu_tol_fail=%llu exact_ref_mismatch=%llu argmax_mismatch=%u\n",
            logits_parity.checked_steps,
            (unsigned long long)logits_parity.compared_values,
            logits_parity.max_abs_error, logits_parity.max_rel_error,
            (unsigned long long)logits_parity.cpu_tolerance_failures,
            (unsigned long long)logits_parity.exact_reference_mismatches,
            logits_parity.argmax_mismatches);
    fflush(stderr);

    FILE *out = open_output(output_path);
    fprintf(out, "{\"kind\": 2, \"decode_len\": %u, \"num_examples\": 1,\n", n);
    fprintf(out, " \"examples\": [\n  {\"index\": 0,\n   \"gen_traj\": [");
    for (uint32_t j = 0; j < n; ++j) fprintf(out, "%s%d", j ? ", " : "", traj[j]);
    fprintf(out, "],\n   \"tf_argmax\": [");
    for (uint32_t j = 0; j < n; ++j) fprintf(out, "%s%d", j ? ", " : "", traj[j]);
    fprintf(out, "],\n   \"per_step_tpot_ms\": [");
    for (uint32_t j = 0; j < n; ++j) fprintf(out, "%s%.6f", j ? ", " : "", tpot[j]);
    fprintf(out, "]}\n ],\n");
    fprintf(out,
            " \"logits_parity\": {\"checked_steps\": %u, "
            "\"compared_values\": %llu, \"max_abs_error\": %.17g, "
            "\"max_rel_error\": %.17g, \"cpu_tolerance_failures\": %llu, "
            "\"exact_reference_mismatches\": %llu, "
            "\"argmax_mismatches\": %u}}\n",
            logits_parity.checked_steps,
            (unsigned long long)logits_parity.compared_values,
            logits_parity.max_abs_error, logits_parity.max_rel_error,
            (unsigned long long)logits_parity.cpu_tolerance_failures,
            (unsigned long long)logits_parity.exact_reference_mismatches,
            logits_parity.argmax_mismatches);
    if (output_path != NULL) fclose(out);

    free(traj);
    free(tpot);
    free(final_hidden);
    free(reference_logits);
    free(saved_logits);
    if (logits_dump != NULL) fclose(logits_dump);
    if (logits_reference != NULL) fclose(logits_reference);
    return (logits_parity.cpu_tolerance_failures == 0 &&
            logits_parity.exact_reference_mismatches == 0 &&
            logits_parity.argmax_mismatches == 0) ? 0 : -1;
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
    const char *logits_dump_path = NULL;
    const char *logits_reference_path = NULL;
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
        } else if (strcmp(arg, "--logits-dump") == 0) {
            if (arg_index + 1 >= argc) die("--logits-dump requires a path");
            logits_dump_path = argv[++arg_index];
        } else if (strcmp(arg, "--logits-reference") == 0) {
            if (arg_index + 1 >= argc) die("--logits-reference requires a path");
            logits_reference_path = argv[++arg_index];
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
                " --decode --decode-from-state <state.gdnstate> [--decode-len N]"
                " [--logits-dump file] [--logits-reference file]\n",
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
    log_progress_message("[progress] validating exact weight-shard layout");
    if (gdn_validate_weight_shards(model.weight_data, &model.config,
                                   run_state.weight_shards) != 0) {
        gdn_run_state_free(&run_state);
        gdn_model_free(&model);
        return 1;
    }
    logits = (float *)xmalloc((size_t)model.config.vocab_size * sizeof(float));
    log_progress_message("[progress] loading fixture");
    load_fixture(argv[2], &fixture);
    progress_start(&progress);
    fprintf(
        stderr,
        "[progress] starting decode evaluation kind=%u examples=%u\n",
        fixture.kind,
        fixture.num_examples
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
            if (fixture.kind != REQ_KIND_LL || fixture.num_examples == 0) {
                die("decode-from-state requires an LL-kind fixture with at least 1 example");
            }
            n = fixture.first_cont_len;
        }
        if (n == 0) die("decode-from-state: zero decode length");
        const char *decode_out = (argc == 4) ? argv[3] : "results_decode_c/decode.c.json";
        if (run_decode_from_state(&model, &run_state, logits, state_path, n,
                                  decode_out, logits_dump_path,
                                  logits_reference_path) != 0) {
            die("decode-from-state: full-logits parity failed");
        }
        fprintf(stderr, "[progress] decode finished elapsed=%.0fs\n", elapsed_seconds(&progress));
        fflush(stderr);
        free(logits);
        gdn_run_state_free(&run_state);
        gdn_model_free(&model);
        return 0;
    }
}
