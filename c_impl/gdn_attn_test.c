/*
 * gdn_attn_test.c — Verify single-layer GDN attention against Python golden output.
 *
 * Usage:
 *   ./gdn_attn_test <weights.gdnw> <fixture.gdnblk>
 *
 * The fixture is produced by scripts/export_block_fixture.py and contains:
 *   header  (24 bytes): magic "GDNBLK1\0", version, block_index, num_tokens, hidden_size
 *   attn_input:  float32[num_tokens * hidden_size]  (post-RMSNorm)
 *   attn_output: float32[num_tokens * hidden_size]  (pre-residual golden)
 *
 * Exit 0 on pass, 1 on failure.
 */

#include "gdn_model.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BLK_HEADER_BYTES 24   /* 8 magic + 4*4 fields */
#define ATOL 1e-3f
#define RTOL 1e-3f

typedef struct {
    uint32_t block_index;
    uint32_t num_tokens;
    uint32_t hidden_size;
    float *attn_input;      /* [num_tokens * hidden_size] */
    float *golden_output;   /* [num_tokens * hidden_size] */
} AttnFixture;

static void die(const char *msg) {
    fprintf(stderr, "FATAL: %s\n", msg);
    exit(1);
}

static void *xmalloc(size_t bytes) {
    void *p = malloc(bytes);
    if (!p) die("malloc failed");
    return p;
}

static int load_attn_fixture(const char *path, AttnFixture *fix) {
    FILE *f;
    uint8_t header[BLK_HEADER_BYTES];
    uint32_t version;
    size_t count;

    memset(fix, 0, sizeof(*fix));

    f = fopen(path, "rb");
    if (!f) { perror("fopen fixture"); return -1; }

    if (fread(header, 1, BLK_HEADER_BYTES, f) != BLK_HEADER_BYTES) {
        fprintf(stderr, "failed to read fixture header\n");
        fclose(f);
        return -1;
    }

    if (memcmp(header, "GDNBLK1", 7) != 0) {
        fprintf(stderr, "bad fixture magic\n");
        fclose(f);
        return -1;
    }

    memcpy(&version,          header + 8,  4);
    memcpy(&fix->block_index, header + 12, 4);
    memcpy(&fix->num_tokens,  header + 16, 4);
    memcpy(&fix->hidden_size, header + 20, 4);

    if (version != 1) {
        fprintf(stderr, "unsupported fixture version %u\n", version);
        fclose(f);
        return -1;
    }

    count = (size_t)fix->num_tokens * fix->hidden_size;
    fix->attn_input    = (float *)xmalloc(count * sizeof(float));
    fix->golden_output = (float *)xmalloc(count * sizeof(float));

    if (fread(fix->attn_input, sizeof(float), count, f) != count) {
        fprintf(stderr, "failed to read attn_input\n");
        fclose(f);
        return -1;
    }
    if (fread(fix->golden_output, sizeof(float), count, f) != count) {
        fprintf(stderr, "failed to read golden_output\n");
        fclose(f);
        return -1;
    }

    fclose(f);
    return 0;
}

static void free_attn_fixture(AttnFixture *fix) {
    free(fix->attn_input);
    free(fix->golden_output);
    memset(fix, 0, sizeof(*fix));
}

int main(int argc, char **argv) {
    GDNModel model;
    GDNRunState state;
    AttnFixture fix;
    float *c_output;
    size_t count;
    uint32_t i;
    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;
    uint32_t fail_count = 0;

    if (argc != 3) {
        fprintf(stderr, "usage: %s <weights.gdnw> <fixture.gdnblk>\n", argv[0]);
        return 1;
    }

    /* Load model weights */
    fprintf(stderr, "loading weights: %s\n", argv[1]);
    if (gdn_model_load(&model, argv[1]) != 0) {
        die("failed to load model weights");
    }

    /* Load attention fixture */
    fprintf(stderr, "loading fixture: %s\n", argv[2]);
    if (load_attn_fixture(argv[2], &fix) != 0) {
        gdn_model_free(&model);
        die("failed to load attention fixture");
    }

    /* Validate fixture vs model config */
    if (fix.hidden_size != model.config.hidden_size) {
        fprintf(stderr, "hidden_size mismatch: fixture=%u model=%u\n",
                fix.hidden_size, model.config.hidden_size);
        free_attn_fixture(&fix);
        gdn_model_free(&model);
        return 1;
    }
    if (fix.block_index >= model.config.num_layers) {
        fprintf(stderr, "block_index %u >= num_layers %u\n",
                fix.block_index, model.config.num_layers);
        free_attn_fixture(&fix);
        gdn_model_free(&model);
        return 1;
    }

    fprintf(stderr, "block=%u  num_tokens=%u  hidden=%u\n",
            fix.block_index, fix.num_tokens, fix.hidden_size);

    /* Allocate run state */
    if (gdn_run_state_init(&state, &model, fix.num_tokens) != 0) {
        free_attn_fixture(&fix);
        gdn_model_free(&model);
        die("failed to init run state");
    }

    /* Run attention forward */
    count = (size_t)fix.num_tokens * fix.hidden_size;
    c_output = (float *)xmalloc(count * sizeof(float));

    fprintf(stderr, "running C attention forward (layer %u)...\n", fix.block_index);
    if (gdn_attn_forward_layer(&model, &state, fix.block_index,
                                fix.attn_input, c_output, fix.num_tokens) != 0) {
        free(c_output);
        gdn_run_state_free(&state);
        free_attn_fixture(&fix);
        gdn_model_free(&model);
        die("attention forward failed");
    }

    /* Compare outputs */
    for (i = 0; i < (uint32_t)count; ++i) {
        float diff = fabsf(c_output[i] - fix.golden_output[i]);
        float denom = fmaxf(fabsf(fix.golden_output[i]), 1e-8f);
        float rel = diff / denom;

        if (diff > max_abs_diff) max_abs_diff = diff;
        if (rel > max_rel_diff) max_rel_diff = rel;

        if (diff > ATOL + RTOL * fabsf(fix.golden_output[i])) {
            if (fail_count < 10) {
                uint32_t token = i / fix.hidden_size;
                uint32_t col   = i % fix.hidden_size;
                fprintf(stderr, "  MISMATCH [token=%u col=%u]: C=%.8f  golden=%.8f  diff=%.2e\n",
                        token, col, c_output[i], fix.golden_output[i], diff);
            }
            ++fail_count;
        }
    }

    /* Report */
    printf("--- GDN Attention Parity Test (layer %u) ---\n", fix.block_index);
    printf("num_tokens:   %u\n", fix.num_tokens);
    printf("hidden_size:  %u\n", fix.hidden_size);
    printf("total_values: %u\n", (uint32_t)count);
    printf("max_abs_diff: %.6e\n", max_abs_diff);
    printf("max_rel_diff: %.6e\n", max_rel_diff);
    printf("fail_count:   %u  (atol=%.0e rtol=%.0e)\n", fail_count, ATOL, RTOL);

    if (fail_count == 0) {
        printf("RESULT: PASS\n");
    } else {
        printf("RESULT: FAIL (%u / %u values exceed tolerance)\n", fail_count, (uint32_t)count);
    }

    free(c_output);
    gdn_run_state_free(&state);
    free_attn_fixture(&fix);
    gdn_model_free(&model);
    return fail_count == 0 ? 0 : 1;
}
