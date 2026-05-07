# Tiled Matrix Multiplication (`gdn_matmul`)

**Location:** `gdn_model.c:441`

## Overview

`gdn_matmul` implements a tiled GEMM for all linear projections in
GatedDeltaNet. It computes `out = in * weights^T` where `in` is
`(num_rows × in_dim)` and `weights` is `(out_dim × in_dim)` stored row-major.

This is the single most latency-dominant submodule. In the v7 implementation
of `gdn_attn_forward`, the seven matmul calls (Q, K, V, A, B, gate, output
projection) consume **141.3 G of the 141.5 G total cycles** — ≈ 99.9 % of the
attention layer's runtime.

## Tile Strategy

```c
#define MM_TILE_R 16   /* rows (num_tokens dimension) */
#define MM_TILE_C 16   /* columns (out_dim dimension) */
#define MM_TILE_K 16   /* reduction (in_dim dimension) */
```

These divide evenly into hidden=2048 and intermediate=5632 (= 16 × 352).
Smaller output dimensions (e.g. `num_heads = 8` for A/B projections) are
handled by boundary guards in the load/store loops.

## Local Buffers and Partitioning

```c
float local_in[MM_TILE_R][MM_TILE_K];     // input tile
float local_wt[MM_TILE_K][MM_TILE_C];     // weight tile
float local_out[MM_TILE_R][MM_TILE_C];    // accumulator tile

#pragma HLS array_partition variable=local_in  dim=2 complete   // all K elements parallel
#pragma HLS array_partition variable=local_wt  dim=1 complete   // all K elements parallel
#pragma HLS array_partition variable=local_out dim=2 complete   // independent banks per c
```

`local_in` dim 2 complete + `local_wt` dim 1 complete together expose all 16
`(local_in[r][k], local_wt[k][c])` pairs in parallel for any fixed `(r, c)`.

## Loop Structure

```
mm_zero_row/col:        Zero out[][] in DRAM (II=1)
mm_tile_r:              Tile over rows         (ceil(num_rows/16))
  mm_tile_k:            Tile over reduction    (ceil(in_dim/16))
    mm_load_in_r/k:     Load input tile from DRAM (II=1)
    mm_tile_c:          Tile over columns      (ceil(out_dim/16))
      mm_load_wt_c/k:   Load weight tile from DRAM (II=1)
      mm_load_out OR mm_reload_out: load partial output (first or subsequent k-tile)
      mm_comp_rc:       Manual-flat 256-iter dot-product pipeline (II=1)
      mm_store_r/c:     Store output tile back to DRAM (II=1)
```

## The compute kernel: `mm_comp_rc`

The `(R × C)` compute step is the heart of the GEMM. Two structural choices
matter for II=1.

### Manual flatten of the (R, C) nest

HLS will not auto-flatten `mm_comp_r` × `mm_comp_c` because the surrounding
`mm_tile_c` body has statements before and after the compute (the load and
store sub-loops). The auto-flatten heuristic emits:

```
WARNING: [HLS 200-960] Cannot flatten loop 'mm_comp_r' ... outer loop is not a
                      perfect loop.
```

Without flattening, the inner pipeline is filled and drained 16 times (once
per `r` iter), each fill costing ~30 cycles. The current implementation
collapses the nest into a single 256-iter loop with manual `r`/`c` indexing:

```c
mm_comp_rc: for (uint32_t rc = 0; rc < MM_TILE_R * MM_TILE_C; ++rc) {
#pragma HLS pipeline II=1
#pragma HLS dependence variable=local_out type=inter direction=RAW false
    uint32_t r = rc / MM_TILE_C;
    uint32_t c = rc % MM_TILE_C;
    /* ... 16-input dot product ... */
    local_out[r][c] += dot;
}
```

The `dependence ... false` is correct because each iteration touches a unique
`(r, c)` pair, and `local_out` `dim 2 complete` gives each `c` an independent
register bank — so the iter-N write of `local_out[r][c]` cannot conflict with
the iter-N+1 read of `local_out[r'][c']`. Without the pragma, HLS treats the
RMW conservatively and stalls the pipeline at II=2.

### Explicit balanced tree for the 16-input dot product

A natural-looking inner reduction:

```c
float dot = 0.0f;
for (k = 0; k < MM_TILE_K; ++k) {
#pragma HLS unroll
    dot += local_in[r][k] * local_wt[k][c];
}
```

unrolls to **a 16-deep serial fadd chain** in the synthesised RTL (HLS does
not always auto-tree-balance an unrolled `+=` reduction). That makes the
iteration latency 55 cycles and forces the pipeline into II=2 because the
critical-path register write from the last fadd lands two cycles after the
next iter's first fadd is already scheduled.

The current implementation writes the tree explicitly:

```c
float p[MM_TILE_K];
mm_comp_k: for (k = 0; k < MM_TILE_K; ++k) {
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
```

Each level is a balanced pair-add — 4 stages of fadd plus one fmul stage
gives an iteration latency of 19 cycles, and HLS schedules a new iter every
cycle (II=1).

## Synthesis Results (single matmul call, U55C @ 100 MHz)

| Iteration              | mm_comp inner II | per tile_c iter | per call (2048×2048) | Notes |
|------------------------|-----------------:|----------------:|---------------------:|-------|
| v0 (k pipelined, c unrolled, no flatten) | 2 | 624 cyc | 26.83 G | original |
| v1 (8-lane partial accs)                 | 2 | 528 cyc | 24.34 G | auto-flattened, lane mux still serialised |
| v3 (loop swap, c outer, no flatten)      | 2 | 1456 cyc | 38.88 G | regression: 16x pipeline-fill overhead + serial dot+= |
| v4 (manual flatten + explicit tree + dep false) | **1** | **1266 cyc** | **20.11 G** | final, U55C |

For the seven matmul calls in `gdn_attn_forward`, total cost dropped from
**~190 G** in v0 to **~141 G** in v7 (single-layer attention).

## Where the time still goes (v7)

For each `mm_tile_c` iteration on U55C (from the matmul_1 instance,
`solution2/syn/report/csynth.rpt`):

| Sub-phase           | Cycles | % |
|---------------------|-------:|--:|
| `mm_load_wt`        | 329    | 26 % |
| `mm_load_out` / `mm_reload_out` | 258 / 329 | 23 % |
| `mm_comp_rc`        | 274    | 22 % |
| `mm_store`          | 327    | 26 % |
| **total**           | **1266** | 100 % |

Compute is only **22 %** of per-tile time. The remaining 78 % is **m_axi
load/store overhead** of the 16×16 partial output tile reload and store-back
through DRAM between every adjacent `tk` step. This is the fundamental
bottleneck and is the motivation for the dataflow / streaming-GEMM follow-up
in [optimization_log.md §"Critical follow-ups"](optimization_log.md).

## Resource Cost (per instance, U55C)

| Resource | matmul_1 (mem_q out) | matmul_2 (gmem / mem_k out) |
|----------|---------------------:|----------------------------:|
| BRAM_18K | 0                    | 0                           |
| DSP      | 92 (1 %)             | 98 (1 %)                    |
| FF       | 19.4 k (~0 %)        | 20.6 k (~0 %)               |
| LUT      | 18.4 k (1 %)         | 23.1 k (1 %)                |

Note: with `q` and `k` on separate AXI bundles (`bundle=mem_q`,
`bundle=mem_k`) HLS instantiates two matmul implementations (`gdn_matmul_1`
and `gdn_matmul_2`). The top-level resource summary aggregates both.

## Why the v0 form was suboptimal

The original loop body had `local_out[r][c] += a_val * local_wt[k][c]` with
`mm_comp_k` as the pipelined dim. The carried RAW dep on `local_out[r][c]`
across `k` (FP32 fadd is 5- to 7-cycle pipelined) forced II=2. Worse, HLS
auto-flattened the `mm_comp_r` × `mm_comp_k` body when the partial-accumulator
helper code happened to be present (and lost the flattening when it was
removed), giving inconsistent and brittle results. The v4 manual flatten plus
explicit tree gives the same II=1 deterministically, with depth that is
log₂(16) fadd levels (≈ 19 cycles) instead of a 16-deep linear chain.

## Potential Future Optimisations

- **Larger tiles** — a 32×32×16 or 16×16×32 tile would increase compute
  density but require more BRAM and partition bandwidth.
- **Output-stationary streaming GEMM** — keep the output tile in BRAM across
  the entire `tk` traversal of one `(tr, tc)` and never reload from DRAM.
  This eliminates ~46 % of per-tile time. Requires `#pragma HLS dataflow` and
  `hls::stream` channels for input/weight feeds.
- **Weight-stationary** — for batch-1 inference, stream activations through a
  pre-loaded weight grid. Best when activations are small relative to weights;
  may not fit the GDN-1.3B layer dims (hidden=2048).
