# Systolic Matrix Multiplication

This document describes the current systolic-array GEMM used by the HLS
GatedDeltaNet accelerator. It supersedes the old tiled GEMM as the main engine
for large projections, while the tiled GEMM remains as a fallback for very
small output dimensions.

## Locations

| File | Role |
|------|------|
| `gdn_model.cpp:439` | Integrated copy used by `gdn_forward` and `gdn_attn_forward` |
| `gdn_matmul_systolic.cpp:1` | Standalone matmul-only HLS top and test target |
| `gdn_matmul_test.cpp` | Native C parity testbench for the standalone top |
| `test_matmul.tcl` | HLS script for the standalone matmul test |

Both implementations compute:

```text
out[num_rows x out_dim] = in[num_rows x in_dim] * weights[out_dim x in_dim]^T
```

The weight matrix is row-major with one row per output channel.

## Why the Tiled GEMM Was Replaced

The previous `gdn_matmul_tiled` kernel used 16 x 16 x 16 tiles. Its inner
compute loop reached II=1, but every K tile reloaded and re-stored the 16 x 16
partial output tile through DRAM. In the v7 single-attention report, the large
projection matmul instance still cost about 20.11 G cycles, and the attention
top was almost entirely matmul-bound.

The systolic design changes the data movement model:

- Keep each output tile's partial sums inside processing elements.
- Stream activation and weight tiles through a PE chain with `hls::stream`.
- Use a `#pragma HLS dataflow` region so readers, PEs, sink, and writer run
  concurrently.
- Use packed 16-float words (`Pack16`) to align the datapath with wide AXI
  memory beats where HLS can legally widen the top-level port.

## Current Dispatch Policy

The integrated model has two matmul paths:

| Shape | Current path | Reason |
|-------|--------------|--------|
| Q/K/V/Gate/O projection, `out_dim=2048` | `gdn_matmul_systolic` | Large output dimension, divisible by 16 |
| MLP gate/up, `out_dim=5632` | `gdn_matmul_systolic` | Large output dimension, divisible by 16 |
| MLP down, `in_dim=5632`, `out_dim=2048` | `gdn_matmul_systolic` | `IN_DIM_MAX=5632` in integrated kernel |
| A/B projections, `out_dim=8` | `gdn_matmul_tiled` | Too small for the 16-column systolic tile |

The fallback is intentional. The A/B projection output dimension is only the
number of heads, so forcing it through a 16-column systolic tile would waste
most of the PE array and complicate boundary handling.

## Integrated vs Standalone Kernels

There are two related but not identical systolic kernels.

### Integrated kernel (`gdn_model.cpp`)

```c
#define N_PES       16
#define M_PER_PE   16
#define IN_DIM_MAX 5632
#define NUM_CHAINS 1
```

The integrated kernel uses one PE chain. This keeps the HLS dataflow region
compatible with the single-reader/single-writer rule on each physical AXI
bundle in the full model and single-attention top. `weight_data` is placed on
`bundle=mem_weights`, while activations and outputs remain on the default
`gmem` bundle, except `q` and `k` in `gdn_attn_forward`, which use `mem_q` and
`mem_k` for recurrent-attention load bandwidth.

### Standalone matmul kernel (`gdn_matmul_systolic.cpp`)

```c
#define N_PES       16
#define M_PER_PE   16
#define IN_DIM_MAX 2048
#define NUM_CHAINS 2
```

The standalone top uses two PE chains and five explicit AXI bundles:

```text
mem_in, mem_wt_lo, mem_wt_hi, mem_out_lo, mem_out_hi
```

This is a peak-throughput experiment for the dominant 2048 x 2048 x 2048
projection shape. It is not resource-comparable to the integrated model,
because it deliberately instantiates more parallel hardware and exposes a
different top-level memory interface.

## Geometry

### Integrated kernel

| Constraint | Value |
|------------|-------|
| `out_dim % (NUM_CHAINS * M_PER_PE)` | must be 0, so multiple of 16 |
| `in_dim % 16` | must be 0 |
| `in_dim` | must be <= 5632 |
| `num_rows` | any positive value; padded on chip to a multiple of 16 |

When `num_rows` is not a multiple of 16, `ReadA` zero-fills padded rows and
`WriteC_chain` drains but skips out-of-range writes. This keeps stream counts
balanced without issuing out-of-bounds DRAM accesses.

### Standalone kernel

| Constraint | Value |
|------------|-------|
| `num_rows % 16` | must be 0 |
| `out_dim % 32` | must be 0, because `NUM_CHAINS=2` |
| `in_dim % 16` | must be 0 |
| `in_dim` | must be <= 2048 |

The default standalone test is:

```text
num_rows = 2048
in_dim   = 2048
out_dim  = 2048
seed     = 42
```

This is set in `test_matmul.tcl` with:

```tcl
csim_design -argv {2048 2048 2048 42}
```

## Dataflow Architecture

The integrated one-chain dataflow region is:

```text
ReadA -> a_pipes[0] -> PE0 -> PE1 -> ... -> PE15 -> SinkAB
ReadB -> b_pipes[0] -> PE0 -> PE1 -> ... -> PE15 -> SinkAB
                                      |
                                      v
                                  c_streams
                                      |
                                      v
                                  WriteC_chain
```

The standalone two-chain version duplicates the PE chain and ReadB/WriteC
paths. `ReadA` broadcasts the same activation column pack to both chains.

## `Pack16`

```c
struct Pack16 {
    float data[16];
};
```

`Pack16` is the internal stream word: 16 FP32 values, or 64 bytes. The
standalone HLS top exposes `Pack16 *` ports directly, which forces 512-bit AXI
data width. The integrated top has float-pointer public ports and casts inside
the helper; Vitis may widen some accesses, but the top-level bundle behavior is
controlled by the surrounding `gdn_forward` / `gdn_attn_forward` interfaces.

## `ReadA`

`ReadA` loads one 16-row activation stripe:

```text
a_buf[N_PES][IN_DIM_MAX]
```

Important pragmas:

```c
#pragma HLS array_partition variable=a_buf dim=1 complete
#pragma HLS array_partition variable=a_buf dim=2 cyclic factor=16
```

The load phase reads contiguous row-major activation packs from DRAM. The
stream phase emits one column pack per K index, so each PE receives the value
for its row at the same K step. In the integrated kernel, padded rows are
filled with zero.

## `ReadB`

`ReadB` uses a ping-pong weight buffer:

```text
b_buf[2][M_PER_PE][IN_DIM_MAX]
```

Important pragmas:

```c
#pragma HLS array_partition variable=b_buf dim=1 complete
#pragma HLS array_partition variable=b_buf dim=2 complete
#pragma HLS array_partition variable=b_buf dim=3 cyclic factor=16
```

The intended schedule is:

1. Preload the first 16-column weight tile.
2. Stream the current tile to the PE chain.
3. Load the next tile into the other ping-pong half during the same loop.
4. Repeat until the final tile, then drain it.

In the current integrated single-attention synthesis report, `ReadB` is the
dominant bottleneck and HLS schedules the `fused_tile_fused_io` loop at II=16.
The standalone top is more aggressive because it has explicit `Pack16 *` ports
and separate weight bundles for the two chains.

## Processing Element

Each PE owns one row of the current 16-row output tile and computes 16 output
columns for that row. On every K cycle it:

1. Reads one `Pack16` activation word and selects `a_pack.data[location]`.
2. Forwards the activation pack to the next PE.
3. Reads one `Pack16` weight word and forwards it.
4. Performs 16 parallel MACs, one per output column.

The PE does not accumulate directly into a scalar recurrence. Instead it uses
16 partial lanes:

```text
c_buf[M_PER_PE][PARTIAL], PARTIAL=16
```

The lane is selected by `k & 15`. This gives the FP32 adder pipeline 16 cycles
before the same lane is reused, which avoids a tight loop-carried dependence.
At the end of the K loop, each PE tree-reduces the 16 partial lanes for each
output column and emits one `Pack16` row on its `c_stream`.

## `WriteC_chain`

`WriteC_chain` drains the per-PE output streams in row order and stores the
16-column output packs. In the integrated kernel it also skips writes for
padded rows while still draining all stream entries.

## Synthesis Results

### Single-layer attention top

These rows compare the v7 tiled-matmul attention design with the current
systolic attention design. They are top-level `gdn_attn_forward` estimates, not
isolated single-matmul resources.

| Design | Latency | BRAM_18K | DSP | FF | LUT | Slack |
|--------|--------:|---------:|----:|---:|----:|------:|
| Tiled matmul, `GDN_single_attn/solution2` | 141.03 G cycles | 322 (7%) | 1042 (11%) | 209.8 K (8%) | 237.0 K (18%) | 0.00 ns |
| Systolic matmul, `GDN_single_attn_synth/solution_synth` | 3.976 G cycles | 1602 (39%) | 4690 (51%) | 848.9 K (32%) | 932.0 K (71%) | -0.04 ns |

Top-level speedup:

```text
141.027e9 / 3.9759e9 = 35.5x
```

Within the current single-attention report, each integrated `run_dataflow`
instance is reported as:

| Module | Latency | BRAM_18K | DSP | FF | LUT |
|--------|--------:|---------:|----:|---:|----:|
| `run_dataflow` | 738.29 M cycles | 512 (12%) | 1873 (20%) | 328.9 K (12%) | 344.0 K (26%) |

### Full-model top

The full model does not instantiate 24 physical layers. The `layer_loop` is a
time loop, so resources are for one reused datapath.

| Module | Latency | BRAM_18K | DSP | FF | LUT | Slack |
|--------|--------:|---------:|----:|---:|----:|------:|
| `gdn_forward` | 129.686 G cycles | 1058 (26%) | 2847 (31%) | 508.4 K (19%) | 580.3 K (44%) | -0.04 ns |
| integrated `gdn_matmul_systolic` | 255.42 M cycles | 768 (19%) | 1862 (20%) | 316.3 K (12%) | 334.7 K (25%) | 0.00 ns |

### Standalone matmul-only top

The standalone test uses `2048 x 2048 x 2048`.

| Module | Latency | BRAM_18K | DSP | FF | LUT | Notes |
|--------|--------:|---------:|----:|---:|----:|-------|
| `gdn_matmul_kernel` | 17.285 M cycles | 0 | 2659 (29%) | 976.3 K (37%) | 656.8 K (50%) | Two chains, five AXI bundles |

The standalone top can use more FF/LUT/DSP than the full-model integrated
matmul because it is a different kernel: two PE chains, `IN_DIM_MAX=2048`,
explicit packed AXI ports, and no need to share the top-level interface with
the rest of GatedDeltaNet.

## Interpreting the Numbers

Do not compare standalone and integrated resource rows as if they were the
same hardware. Use:

- Standalone matmul report for a peak-throughput `2048^3` GEMM experiment.
- Single-attention report for the current attention block resource/latency
  impact.
- Full-model report for the reused 24-layer accelerator datapath.

Also note that Vitis HLS latency estimates for the integrated design use
`loop_tripcount` bounds on runtime dimensions. A module row may therefore
reflect the maximum supported bound, not only the exact `2048 x 2048 x 2048`
projection shape.

## Current Limitations

- Integrated `ReadB` is still the bottleneck in the current report; HLS shows
  II=16 for the fused stream/load loop.
- The current systolic single-attention and full-model reports have a small
  timing miss of -0.04 ns at a 10 ns target.
- The integrated kernel uses one PE chain because adding a second chain would
  require additional independent top-level AXI bundles or a more invasive
  memory-interface redesign.
- A/B projections still use the tiled fallback because `out_dim=8`.

