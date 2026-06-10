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

static void log_example_progress(
    const ProgressState *progress,
    const char *kind_name,
    uint32_t example_index,
    uint32_t total_examples
) {
    fprintf(
        stderr,
        "[progress] %s example %u/%u elapsed=%.0fs\n",
        kind_name,
        example_index,
        total_examples,
        elapsed_seconds(progress)
    );
    fflush(stderr);
}

static void log_example_start(
    const ProgressState *progress,
    const char *kind_name,
    uint32_t example_index,
    uint32_t total_examples
) {
    fprintf(
        stderr,
        "[progress] %s example %u/%u started elapsed=%.0fs\n",
        kind_name,
        example_index,
        total_examples,
        elapsed_seconds(progress)
    );
    fflush(stderr);
}

static void log_rolling_window_start(
    const ProgressState *progress,
    uint32_t example_index,
    uint32_t total_examples,
    uint32_t window_index,
    uint32_t windows_in_example
) {
    fprintf(
        stderr,
        "[progress] rolling doc %u/%u window %u/%u started total_windows %u/%u elapsed=%.0fs\n",
        example_index,
        total_examples,
        window_index,
        windows_in_example,
        progress->completed_windows + 1,
        progress->total_windows,
        elapsed_seconds(progress)
    );
    fflush(stderr);
}

static void log_rolling_window_progress(
    const ProgressState *progress,
    uint32_t example_index,
    uint32_t total_examples,
    uint32_t window_index,
    uint32_t windows_in_example
) {
    fprintf(
        stderr,
        "[progress] rolling doc %u/%u window %u/%u total_windows %u/%u elapsed=%.0fs\n",
        example_index,
        total_examples,
        window_index,
        windows_in_example,
        progress->completed_windows,
        progress->total_windows,
        elapsed_seconds(progress)
    );
    fflush(stderr);
}

static void score_hidden_step(
    const GDNModel *model,
    const float *hidden,
    float *logits,
    int32_t target,
    double *logprob,
    int *greedy,
    int *out_argmax
) {
    uint32_t vocab = model->config.vocab_size;
    uint32_t vocab_index;
    float max_logit;
    int max_index;
    double sum = 0.0;

    gdn_compute_logits(model, hidden, logits);

    max_logit = logits[0];
    max_index = 0;
    for (vocab_index = 1; vocab_index < vocab; ++vocab_index) {
        if (logits[vocab_index] > max_logit) {
            max_logit = logits[vocab_index];
            max_index = (int)vocab_index;
        }
    }

    for (vocab_index = 0; vocab_index < vocab; ++vocab_index) {
        sum += exp((double)logits[vocab_index] - (double)max_logit);
    }

    *logprob += (double)logits[target] - ((double)max_logit + log(sum));
    if (max_index != target) {
        *greedy = 0;
    }
    if (out_argmax != NULL) {
        *out_argmax = max_index;
    }
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

static void score_pair(
    const GDNModel *model,
    GDNRunState *run_state,
    float *logits,
    const int32_t *ctx,
    uint32_t ctx_len,
    const int32_t *cont,
    uint32_t cont_len,
    double *logprob,
    int *greedy
) {
    uint32_t total_tokens;
    uint32_t hidden = model->config.hidden_size;
    int32_t *tokens;
    uint32_t token_index;

    if (ctx_len == 0 || cont_len == 0) {
        die("invalid sequence for scoring");
    }
    total_tokens = ctx_len + cont_len - 1;
    if (total_tokens == 0 || total_tokens > model->config.max_seq_len) {
        die("invalid sequence for scoring");
    }

    tokens = (int32_t *)xmalloc((size_t)model->config.max_seq_len * sizeof(int32_t));
    memcpy(tokens, ctx, (size_t)ctx_len * sizeof(int32_t));
    if (cont_len > 1) {
        memcpy(tokens + ctx_len, cont, (size_t)(cont_len - 1) * sizeof(int32_t));
    }

    if (gdn_forward_host(model, run_state, tokens, total_tokens) != 0) {
        free(tokens);
        die("forward pass failed");
    }

    *logprob = 0.0;
    *greedy = 1;
    for (token_index = 0; token_index < cont_len; ++token_index) {
        const float *hidden_row =
            run_state->x_norm + (size_t)(ctx_len + token_index - 1) * hidden;
        score_hidden_step(model, hidden_row, logits, cont[token_index], logprob, greedy, NULL);
    }

    free(tokens);
}

/* Monotonic wall-clock in milliseconds, for per-step TPOT measurement. */
static double monotonic_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

/* --decode mode driver.
 *
 * For each LL example (ctx = prompt, cont = golden greedy trajectory), produces
 * two length-N trajectories plus per-step latency:
 *   - tf_argmax: teacher-forced per-position argmax. One forward over
 *     prompt + golden[:-1]; at each cont position i the argmax of the logits
 *     from hidden row (prompt_len + i - 1) — the token the model would have
 *     emitted given the golden prefix. No compounding error vs the golden.
 *   - gen_traj: free-running greedy. Re-prefill prefix from scratch each step
 *     (the forward clears recurrent state every call), take the argmax of the
 *     last hidden row, append it, repeat. O(n^2) re-prefill — the honest
 *     current-kernel baseline.
 *   - per_step_tpot_ms: wall time of each greedy step's forward + argmax.
 *
 * Writes c_impl/results_decode_c/decode.c.json. N = min(cont_len, decode_len)
 * (decode_len==0 means "use cont_len"); E = min(num_examples, limit)
 * (limit==0 means "all"). */
static int run_decode_mode(
    const GDNModel *model,
    GDNRunState *run_state,
    float *logits,
    const Fixture *fixture,
    uint32_t decode_len,
    uint32_t limit,
    const char *output_path
) {
    uint32_t hidden = model->config.hidden_size;
    uint32_t num_examples = fixture->num_examples;
    uint32_t example_count;
    uint32_t example_index;
    int32_t **gen_traj;
    int32_t **tf_argmax;
    double **per_step_tpot_ms;
    uint32_t *example_n;
    FILE *out;

    if (fixture->kind != REQ_KIND_LL || fixture->ll_examples == NULL) {
        die("--decode requires an LL-kind (.gdnreq kind=2) fixture");
    }

    example_count = num_examples;
    if (limit != 0 && limit < example_count) {
        example_count = limit;
    }

    gen_traj = (int32_t **)xcalloc(example_count, sizeof(int32_t *));
    tf_argmax = (int32_t **)xcalloc(example_count, sizeof(int32_t *));
    per_step_tpot_ms = (double **)xcalloc(example_count, sizeof(double *));
    example_n = (uint32_t *)xcalloc(example_count, sizeof(uint32_t));

    for (example_index = 0; example_index < example_count; ++example_index) {
        PairReq *req = &fixture->ll_examples[example_index];
        uint32_t prompt_len = req->ctx_len;
        uint32_t n;
        uint32_t step;
        int32_t *tf_tokens;
        int32_t *prefix;
        uint32_t prefix_len;

        if (prompt_len == 0) {
            die("--decode: empty prompt (ctx_len == 0)");
        }

        n = req->cont_len;
        if (decode_len != 0 && decode_len < n) {
            n = decode_len;
        }
        if (n == 0) {
            die("--decode: zero decode length");
        }
        if (prompt_len + n - 1 > model->config.max_seq_len) {
            die("--decode: prompt + decode_len exceeds max_seq_len");
        }

        example_n[example_index] = n;
        gen_traj[example_index] = (int32_t *)xmalloc((size_t)n * sizeof(int32_t));
        tf_argmax[example_index] = (int32_t *)xmalloc((size_t)n * sizeof(int32_t));
        per_step_tpot_ms[example_index] = (double *)xmalloc((size_t)n * sizeof(double));

        fprintf(stderr, "[progress] decode example %u/%u prompt_len=%u N=%u\n",
                example_index + 1, example_count, prompt_len, n);
        fflush(stderr);

        /* ---- Teacher-forced per-position argmax ---- */
        /* Tokens = prompt + golden[:-1]; total length prompt_len + (n - 1).
         * Position i (0..n-1) predicts golden[i] from hidden row
         * (prompt_len + i - 1). */
        {
            uint32_t tf_total = prompt_len + n - 1;
            uint32_t i;
            tf_tokens = (int32_t *)xmalloc((size_t)tf_total * sizeof(int32_t));
            memcpy(tf_tokens, req->ctx, (size_t)prompt_len * sizeof(int32_t));
            if (n > 1) {
                memcpy(tf_tokens + prompt_len, req->cont, (size_t)(n - 1) * sizeof(int32_t));
            }
            if (gdn_forward_host(model, run_state, tf_tokens, tf_total) != 0) {
                free(tf_tokens);
                die("--decode: teacher-forced forward pass failed");
            }
            for (i = 0; i < n; ++i) {
                const float *hidden_row =
                    run_state->x_norm + (size_t)(prompt_len + i - 1) * hidden;
                tf_argmax[example_index][i] = argmax_logits(model, hidden_row, logits);
            }
            free(tf_tokens);
        }

        /* ---- Free-running greedy + per-step latency ---- */
        prefix = (int32_t *)xmalloc((size_t)(prompt_len + n) * sizeof(int32_t));
        memcpy(prefix, req->ctx, (size_t)prompt_len * sizeof(int32_t));
        prefix_len = prompt_len;
        for (step = 0; step < n; ++step) {
            double t0;
            const float *hidden_last;
            int next;

            t0 = monotonic_ms();
            if (gdn_forward_host(model, run_state, prefix, prefix_len) != 0) {
                free(prefix);
                die("--decode: free-running forward pass failed");
            }
            hidden_last = run_state->x_norm + (size_t)(prefix_len - 1) * hidden;
            next = argmax_logits(model, hidden_last, logits);
            per_step_tpot_ms[example_index][step] = monotonic_ms() - t0;

            gen_traj[example_index][step] = next;
            prefix[prefix_len] = next;
            prefix_len += 1;
        }
        free(prefix);

        fprintf(stderr, "[progress] decode example %u/%u complete\n",
                example_index + 1, example_count);
        fflush(stderr);
    }

    /* ---- Write the C decode JSON (schema consumed by the parity checker) ---- */
    out = open_output(output_path);
    fprintf(out, "{\"kind\": %u, \"decode_len\": %u, \"num_examples\": %u,\n",
            fixture->kind, decode_len, example_count);
    fprintf(out, " \"examples\": [");
    for (example_index = 0; example_index < example_count; ++example_index) {
        uint32_t n = example_n[example_index];
        uint32_t j;
        fprintf(out, "%s\n  {\"index\": %u,\n", example_index ? "," : "", example_index);
        fprintf(out, "   \"gen_traj\": [");
        for (j = 0; j < n; ++j) {
            fprintf(out, "%s%d", j ? ", " : "", gen_traj[example_index][j]);
        }
        fprintf(out, "],\n   \"tf_argmax\": [");
        for (j = 0; j < n; ++j) {
            fprintf(out, "%s%d", j ? ", " : "", tf_argmax[example_index][j]);
        }
        fprintf(out, "],\n   \"per_step_tpot_ms\": [");
        for (j = 0; j < n; ++j) {
            fprintf(out, "%s%.6f", j ? ", " : "", per_step_tpot_ms[example_index][j]);
        }
        fprintf(out, "]}");
    }
    fprintf(out, "\n ]}\n");
    if (output_path != NULL) {
        fclose(out);
    }

    for (example_index = 0; example_index < example_count; ++example_index) {
        free(gen_traj[example_index]);
        free(tf_argmax[example_index]);
        free(per_step_tpot_ms[example_index]);
    }
    free(gen_traj);
    free(tf_argmax);
    free(per_step_tpot_ms);
    free(example_n);
    return 0;
}

int main(int argc, char **argv) {
    GDNModel model;
    GDNRunState run_state;
    Fixture fixture;
    ProgressState progress;
    float *logits;
    int decode_mode = 0;
    uint32_t decode_len = 0;   /* 0 => use each example's cont_len */
    uint32_t decode_limit = 0; /* 0 => all examples */
    const char *positional[3];
    int positional_count = 0;
    int arg_index;

    /* Parse args: up to 3 positionals (weights, fixture, [output]) plus the
     * optional --decode flags. Without --decode, behavior is unchanged. */
    for (arg_index = 1; arg_index < argc; ++arg_index) {
        const char *arg = argv[arg_index];
        if (strcmp(arg, "--decode") == 0) {
            decode_mode = 1;
        } else if (strcmp(arg, "--decode-len") == 0) {
            if (arg_index + 1 >= argc) {
                die("--decode-len requires a value");
            }
            decode_len = (uint32_t)strtoul(argv[++arg_index], NULL, 10);
        } else if (strcmp(arg, "--limit") == 0) {
            if (arg_index + 1 >= argc) {
                die("--limit requires a value");
            }
            decode_limit = (uint32_t)strtoul(argv[++arg_index], NULL, 10);
        } else if (positional_count < 3) {
            positional[positional_count++] = arg;
        } else {
            fprintf(stderr, "unexpected argument: %s\n", arg);
            return 1;
        }
    }

    if (positional_count < 2) {
        fprintf(stderr,
                "usage: %s <weights.gdnw> <fixture.gdnreq> [output.json]\n"
                "       %s <weights.gdnw> <fixture.gdnreq> [output.json] --decode [--decode-len N] [--limit E]\n",
                argv[0], argv[0]);
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

    if (decode_mode) {
        /* Contract output path is c_impl/results_decode_c/decode.c.json; an
         * explicit positional output (argv[3]) overrides it if provided. */
        const char *decode_out = (argc == 4)
            ? argv[3]
            : "results_decode_c/decode.c.json";
        run_decode_mode(&model, &run_state, logits, &fixture,
                        decode_len, decode_limit, decode_out);
        fprintf(stderr, "[progress] decode finished elapsed=%.0fs\n", elapsed_seconds(&progress));
        fflush(stderr);
        free_fixture(&fixture);
        free(logits);
        gdn_run_state_free(&run_state);
        gdn_model_free(&model);
        return 0;
    }

    if (fixture.kind == REQ_KIND_MC) {
        double acc = 0.0;
        double acc_norm = 0.0;
        int *preds = (int *)xmalloc((size_t)fixture.num_examples * sizeof(int));
        int *preds_norm = (int *)xmalloc((size_t)fixture.num_examples * sizeof(int));
        double **scores_all = (double **)xcalloc(fixture.num_examples, sizeof(double *));
        double **scores_norm_all = (double **)xcalloc(fixture.num_examples, sizeof(double *));
        uint32_t example_index;

        for (example_index = 0; example_index < fixture.num_examples; ++example_index) {
            MCReq *req = &fixture.mc_examples[example_index];
            double *scores = (double *)xmalloc((size_t)req->num_choices * sizeof(double));
            double *scores_norm = (double *)xmalloc((size_t)req->num_choices * sizeof(double));
            int pred = 0;
            int pred_norm = 0;
            uint32_t choice_index;

            log_example_start(&progress, "multiple_choice", example_index + 1, fixture.num_examples);
            for (choice_index = 0; choice_index < req->num_choices; ++choice_index) {
                int greedy;
                score_pair(
                    &model,
                    &run_state,
                    logits,
                    req->choices[choice_index].ctx,
                    req->choices[choice_index].ctx_len,
                    req->choices[choice_index].cont,
                    req->choices[choice_index].cont_len,
                    &scores[choice_index],
                    &greedy
                );
                scores_norm[choice_index] =
                    scores[choice_index] /
                    (double)(req->choices[choice_index].cont_len ? req->choices[choice_index].cont_len : 1);
                if (scores[choice_index] > scores[pred]) {
                    pred = (int)choice_index;
                }
                if (scores_norm[choice_index] > scores_norm[pred_norm]) {
                    pred_norm = (int)choice_index;
                }
            }

            if ((uint32_t)pred == req->gold) {
                acc += 1.0;
            }
            if ((uint32_t)pred_norm == req->gold) {
                acc_norm += 1.0;
            }

            preds[example_index] = pred;
            preds_norm[example_index] = pred_norm;
            scores_all[example_index] = scores;
            scores_norm_all[example_index] = scores_norm;
            progress.completed_examples = example_index + 1;
            log_example_progress(&progress, "multiple_choice", progress.completed_examples, fixture.num_examples);
        }

        {
            FILE *out = open_output(argc == 4 ? argv[3] : NULL);
            fprintf(out, "{\n  \"kind\": %u,\n  \"num_examples\": %u,\n  \"metrics\": {\n", fixture.kind, fixture.num_examples);
            fprintf(out, "    \"acc\": %.9f,\n    \"acc_norm\": %.9f\n  },\n  \"examples\": [\n",
                    acc / fixture.num_examples, acc_norm / fixture.num_examples);
            for (example_index = 0; example_index < fixture.num_examples; ++example_index) {
                MCReq *req = &fixture.mc_examples[example_index];
                uint32_t choice_index;
                fprintf(out, "    {\"index\": %u, \"gold\": %u, \"pred\": %d, \"pred_norm\": %d, \"scores\": [",
                        example_index, req->gold, preds[example_index], preds_norm[example_index]);
                for (choice_index = 0; choice_index < req->num_choices; ++choice_index) {
                    fprintf(out, "%s%.9f", choice_index ? ", " : "", scores_all[example_index][choice_index]);
                }
                fprintf(out, "], \"scores_norm\": [");
                for (choice_index = 0; choice_index < req->num_choices; ++choice_index) {
                    fprintf(out, "%s%.9f", choice_index ? ", " : "", scores_norm_all[example_index][choice_index]);
                }
                fprintf(out, "]}%s\n", (example_index + 1 == fixture.num_examples) ? "" : ",");
            }
            fprintf(out, "  ]\n}\n");
            if (argc == 4) {
                fclose(out);
            }
        }

        fprintf(stdout, "acc=%.6f acc_norm=%.6f\n", acc / fixture.num_examples, acc_norm / fixture.num_examples);
        for (example_index = 0; example_index < fixture.num_examples; ++example_index) {
            free(scores_all[example_index]);
            free(scores_norm_all[example_index]);
        }
        free(scores_all);
        free(scores_norm_all);
        free(preds);
        free(preds_norm);
    } else if (fixture.kind == REQ_KIND_LL) {
        double total_logprob = 0.0;
        uint64_t total_tokens = 0;
        double acc = 0.0;
        double *lls = (double *)xmalloc((size_t)fixture.num_examples * sizeof(double));
        int *greedies = (int *)xmalloc((size_t)fixture.num_examples * sizeof(int));
        uint32_t example_index;

        for (example_index = 0; example_index < fixture.num_examples; ++example_index) {
            PairReq *req = &fixture.ll_examples[example_index];
            log_example_start(&progress, "loglikelihood", example_index + 1, fixture.num_examples);
            score_pair(&model, &run_state, logits, req->ctx, req->ctx_len, req->cont, req->cont_len, &lls[example_index], &greedies[example_index]);
            total_logprob += lls[example_index];
            total_tokens += req->cont_len;
            if (greedies[example_index]) {
                acc += 1.0;
            }
            progress.completed_examples = example_index + 1;
            log_example_progress(&progress, "loglikelihood", progress.completed_examples, fixture.num_examples);
        }

        {
            FILE *out = open_output(argc == 4 ? argv[3] : NULL);
            fprintf(out, "{\n  \"kind\": %u,\n  \"num_examples\": %u,\n  \"metrics\": {\n", fixture.kind, fixture.num_examples);
            fprintf(out, "    \"acc\": %.9f,\n    \"perplexity\": %.9f\n  },\n  \"examples\": [\n",
                    acc / fixture.num_examples, exp(-total_logprob / (double)total_tokens));
            for (example_index = 0; example_index < fixture.num_examples; ++example_index) {
                fprintf(out, "    {\"index\": %u, \"logprob\": %.9f, \"greedy\": %s}%s\n",
                        example_index,
                        lls[example_index],
                        greedies[example_index] ? "true" : "false",
                        (example_index + 1 == fixture.num_examples) ? "" : ",");
            }
            fprintf(out, "  ]\n}\n");
            if (argc == 4) {
                fclose(out);
            }
        }

        fprintf(stdout, "acc=%.6f perplexity=%.6f\n", acc / fixture.num_examples, exp(-total_logprob / (double)total_tokens));
        free(lls);
        free(greedies);
    } else if (fixture.kind == REQ_KIND_ROLLING) {
        double total_logprob = 0.0;
        uint64_t total_words = 0;
        uint64_t total_bytes = 0;
        double *doc_lls = (double *)xmalloc((size_t)fixture.num_examples * sizeof(double));
        uint32_t example_index;

        for (example_index = 0; example_index < fixture.num_examples; ++example_index) {
            RollingReq *req = &fixture.rolling_examples[example_index];
            uint32_t window_index;
            doc_lls[example_index] = 0.0;
            log_example_start(&progress, "rolling", example_index + 1, fixture.num_examples);
            for (window_index = 0; window_index < req->num_windows; ++window_index) {
                double ll;
                int greedy;
                log_rolling_window_start(
                    &progress,
                    example_index + 1,
                    fixture.num_examples,
                    window_index + 1,
                    req->num_windows
                );
                score_pair(
                    &model,
                    &run_state,
                    logits,
                    req->windows[window_index].ctx,
                    req->windows[window_index].ctx_len,
                    req->windows[window_index].cont,
                    req->windows[window_index].cont_len,
                    &ll,
                    &greedy
                );
                (void)greedy;
                doc_lls[example_index] += ll;
                progress.completed_windows += 1;
                log_rolling_window_progress(
                    &progress,
                    example_index + 1,
                    fixture.num_examples,
                    window_index + 1,
                    req->num_windows
                );
            }
            total_logprob += doc_lls[example_index];
            total_words += req->word_count;
            total_bytes += req->byte_count;
            progress.completed_examples = example_index + 1;
            fprintf(
                stderr,
                "[progress] rolling doc %u/%u complete elapsed=%.0fs partial_word_ppl=%.6f\n",
                progress.completed_examples,
                fixture.num_examples,
                elapsed_seconds(&progress),
                exp(-total_logprob / (double)total_words)
            );
            fflush(stderr);
        }

        {
            double word_perplexity = exp(-total_logprob / (double)total_words);
            double byte_perplexity = exp(-total_logprob / (double)total_bytes);
            double bits_per_byte = -total_logprob / (double)total_bytes / log(2.0);
            FILE *out = open_output(argc == 4 ? argv[3] : NULL);
            fprintf(out, "{\n  \"kind\": %u,\n  \"num_examples\": %u,\n  \"metrics\": {\n", fixture.kind, fixture.num_examples);
            fprintf(out, "    \"word_perplexity\": %.9f,\n    \"byte_perplexity\": %.9f,\n    \"bits_per_byte\": %.9f\n  },\n  \"examples\": [\n",
                    word_perplexity, byte_perplexity, bits_per_byte);
            for (example_index = 0; example_index < fixture.num_examples; ++example_index) {
                fprintf(out, "    {\"index\": %u, \"logprob\": %.9f}%s\n",
                        example_index,
                        doc_lls[example_index],
                        (example_index + 1 == fixture.num_examples) ? "" : ",");
            }
            fprintf(out, "  ]\n}\n");
            if (argc == 4) {
                fclose(out);
            }
            fprintf(stdout, "word_perplexity=%.6f byte_perplexity=%.6f bits_per_byte=%.6f\n",
                    word_perplexity, byte_perplexity, bits_per_byte);
        }

        free(doc_lls);
    } else {
        die("unsupported fixture kind");
    }

    fprintf(stderr, "[progress] evaluation finished elapsed=%.0fs\n", elapsed_seconds(&progress));
    fflush(stderr);

    free_fixture(&fixture);
    free(logits);
    gdn_run_state_free(&run_state);
    gdn_model_free(&model);
    return 0;
}
