# Tiled Matrix Multiplication (`gdn_matmul`)

**Location:** `gdn_model.c:407`

## Overview

`gdn_matmul` implements a tiled GEMM for all linear projections in
GatedDeltaNet. It computes `out = in * weights^T` where `in` is
`(num_rows x in_dim)` and `weights` is `(out_dim x in_dim)` stored row-major.

This is the single most latency-dominant submodule. In the full 24-layer model,
matmul accounts for ~97% of total execution time (10 matmuls per layer x 24
layers).

## Tile Strategy

Tile dimensions are compile-time constants:

```c
#define MM_TILE_R 16   /* rows (num_tokens dimension) */
#define MM_TILE_C 16   /* columns (out_dim dimension) */
#define MM_TILE_K 16   /* reduction (in_dim dimension) */
```

These divide evenly into hidden=2048 and intermediate=5632 (= 16 x 352).
Smaller output dimensions (e.g. num_heads=8 for A/B projections) are handled
by boundary guards in the inner loops.

## Local Buffers and Partitioning

Three BRAM tile buffers are used:

```c
float local_in[MM_TILE_R][MM_TILE_K];     // input tile
float local_wt[MM_TILE_K][MM_TILE_C];     // weight tile
float local_out[MM_TILE_R][MM_TILE_C];    // accumulator tile
```

HLS partitioning pragmas:

```c
#pragma HLS array_partition variable=local_in  dim=2 complete   // all K elements accessible
#pragma HLS array_partition variable=local_wt  dim=1 complete   // all K elements accessible
#pragma HLS array_partition variable=local_out dim=2 complete   // all C elements accessible
```

This enables the inner compute loop to read one row of `local_in` (1 value),
broadcast it across all 16 columns of `local_wt`, and accumulate into 16
`local_out` entries in a single pipeline stage.

## Loop Structure

```
mm_zero_row/col:        Zero output buffer (pipeline II=1)
mm_tile_r:              Tile over rows       (ceil(num_rows/16) iterations)
  mm_tile_k:            Tile over reduction  (ceil(in_dim/16) iterations)
    mm_load_in_r/k:     Load input tile from DRAM (pipeline II=1)
    mm_tile_c:           Tile over columns   (ceil(out_dim/16) iterations)
      mm_load_wt_c/k:   Load weight tile from DRAM (pipeline II=1)
      mm_load_out:       Load partial sums (first K-tile: zero, else reload)
      mm_comp_r:         Compute:
        mm_comp_k:         for each k (pipeline II=1):
          mm_comp_c:         for each c (fully unrolled):
                               local_out[r][c] += local_in[r][k] * local_wt[k][c]
      mm_store_r/c:      Store output tile to DRAM (pipeline II=1)
```

## Key Design Decisions

1. **Input tile reuse:** The input tile `local_in[r][k]` is loaded once per
   `(tr, tk)` pair and reused across all column tiles `tc`. This reduces DRAM
   reads for the input matrix.

2. **Complete partitioning on reduction dim:** Both `local_in` (dim 2) and
   `local_wt` (dim 1) are fully partitioned along K. This allows the compute
   loop to pipeline at II=1 with 16 parallel multiply-accumulates per cycle.

3. **Boundary guards:** All load/store operations check `gr < num_rows`,
   `gc < out_dim`, `gk < in_dim` to handle cases where dimensions are not
   exact multiples of 16 (e.g., `out_dim=8` for A/B projections).

## Synthesis Results (Full Model, VU11P @ 100 MHz)

From the tiled matmul full-model csynth (24 layers, all projections):

| Metric | Naive Matmul | Tiled Matmul | Improvement |
|--------|-------------|-------------|-------------|
| GEMM latency/layer | ~2.0 x 10^12 cycles | ~0.27 x 10^12 cycles | 7.3x faster |
| Total latency/layer | 2,290.7 x 10^9 | 592.9 x 10^9 | 3.9x faster |
| DSP | 217 | 271 | +54 |
| FF | 63,680 | 89,074 | +25,394 |
| LUT | 178,453 | 192,986 | +14,533 |
| BRAM | 6 | 6 | same |

The tiled approach achieves 7.3x speedup on GEMM operations by exploiting
on-chip BRAM tile buffers and data reuse, at a modest cost in DSP (+25%) and
FF (+40%).

## Potential Future Optimisations

- **Larger tiles:** Increasing tile size (e.g., 32x32x16) would improve
  compute-to-memory ratio but requires more BRAM.
- **Double buffering:** Overlapping tile loads with compute via ping-pong
  buffers would hide memory latency.
- **Weight stationary dataflow:** For single-token inference, a weight-stationary
  approach might be more efficient since weights dominate memory traffic.
