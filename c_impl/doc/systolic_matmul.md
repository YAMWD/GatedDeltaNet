# Systolic Matrix Multiplication

This document describes the current systolic-array GEMM used by the HLS
GatedDeltaNet accelerator. It supersedes the old tiled GEMM as the main engine
for large projections, while the tiled GEMM remains as a fallback for very
small output dimensions.

## Locations

| File | Role |
|------|------|
| `gdn_model.cpp` (in-file static helpers) | Synthesisable kernel: `ProcessingElement`, `ReadA`, `ReadB`, `SinkAB`, `WriteC_chain`, `run_dataflow`, `gdn_matmul_systolic` |
| `gdn_model.cpp` (`gdn_matmul_top`) | Synthesisable HLS top that wraps `gdn_matmul_systolic` with its own `m_axi` bundle layout (`mem_in`, `mem_weights`, `mem_out`) for the matmul-only test path |
| `gdn_matmul_test.cpp` | Native-C parity testbench: random inputs through `naive_matmul` (golden) vs `gdn_matmul_top` (kernel) |
| `test_matmul.tcl` | Vitis HLS script: csim + csynth with `set_top gdn_matmul_top` and `add_files gdn_model.cpp` |

The kernel computes:

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

## Dispatch Policy

`gdn_forward` and `gdn_attn_forward` have two matmul paths:

| Shape | Current path | Reason |
|-------|--------------|--------|
| Q/K/V/Gate/O projection, `out_dim=2048` | `gdn_matmul_systolic` | Large output dimension, divisible by 16 |
| MLP gate/up, `out_dim=5632` | `gdn_matmul_systolic` | Large output dimension, divisible by 16 |
| MLP down, `in_dim=5632`, `out_dim=2048` | `gdn_matmul_systolic` | `IN_DIM_MAX=5632` covers MLP down |
| A/B projections, `out_dim=8` | `gdn_matmul_tiled` | Too small for the 16-column systolic tile |

The fallback is intentional. The A/B projection output dimension is only the
number of heads, so forcing it through a 16-column systolic tile would waste
most of the PE array and complicate boundary handling.

## Configuration

```c
#define N_PES       16     // PE chain length (rows per outer-N stripe)
#define M_PER_PE    16     // output columns produced by each PE
#define IN_DIM_MAX  5632   // largest `in_dim` the kernel handles in one call
#define NUM_CHAINS  1      // single PE chain
```

`NUM_CHAINS=1` keeps the dataflow region compatible with the
single-reader/single-writer rule on each physical AXI bundle in the
integrated tops (`gdn_forward`, `gdn_attn_forward`). Adding a second chain
would require either splitting the m_axi bundles per chain or introducing a
fork task that fans out the weight stream — see the optimisation backlog.

## Geometry

| Constraint | Value |
|------------|-------|
| `out_dim % M_PER_PE` | must be 0 (multiple of 16) |
| `in_dim % 16` | must be 0 |
| `in_dim` | must be <= `IN_DIM_MAX` (5632) |
| `num_rows` | any positive value; padded on chip to a multiple of `N_PES` |

When `num_rows` is not a multiple of 16, `ReadA` zero-fills padded rows and
`WriteC_chain` drains but skips out-of-range writes. This keeps stream counts
balanced without issuing out-of-bounds DRAM accesses.

### Matmul-only test shape

`test_matmul.tcl` exercises the kernel through `gdn_matmul_top` with the
dominant projection shape:

```tcl
csim_design -argv {2048 2048 2048 42}
```

i.e. `num_rows = in_dim = out_dim = 2048`, RNG seed 42. The shape matches
the Q/K/V/Gate/O projections in GDN-1.3B. Drop to `{64 64 64 42}` for fast
iteration during bring-up.

## Dataflow Architecture

The one-chain dataflow region is:

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

## `Pack16`

```c
struct Pack16 {
    float data[16];
};
```

`Pack16` is the internal stream word: 16 FP32 values, or 64 bytes. The
public HLS top functions expose float-pointer ports and cast to `Pack16 *`
inside the kernel; Vitis may widen some accesses through the m_axi adapter,
but the top-level bundle layout is controlled by the surrounding
`gdn_forward` / `gdn_attn_forward` / `gdn_matmul_top` interfaces.

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
for its row at the same K step. Padded rows are filled with zero.

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

In the current single-attention synthesis report, `ReadB` is the dominant
bottleneck: HLS schedules the `fused_tile_fused_io` loop at II=16 because
the m_axi adapter on `weight_data` defaults to 32-bit data width and reading
one `Pack16` (16 floats) costs 16 narrow beats. Adding
`max_widen_bitwidth=512` to the `weight_data` m_axi pragma is the canonical
fix and remains the top item on the optimisation backlog.

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
16-column output packs. It skips writes for padded rows while still draining
all stream entries to keep the dataflow FIFOs balanced.

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

Within the same report, each `run_dataflow` instance is:

| Module | Latency | BRAM_18K | DSP | FF | LUT |
|--------|--------:|---------:|----:|---:|----:|
| `run_dataflow` | 738.29 M cycles | 512 (12%) | 1873 (20%) | 328.9 K (12%) | 344.0 K (26%) |

### Full-model top

The full model does not instantiate 24 physical layers. The `layer_loop` is a
time loop, so resources are for one reused datapath.

| Module | Latency | BRAM_18K | DSP | FF | LUT | Slack |
|--------|--------:|---------:|----:|---:|----:|------:|
| `gdn_forward` | 129.686 G cycles | 1058 (26%) | 2847 (31%) | 508.4 K (19%) | 580.3 K (44%) | -0.04 ns |
| `gdn_matmul_systolic` instance | 255.42 M cycles | 768 (19%) | 1862 (20%) | 316.3 K (12%) | 334.7 K (25%) | 0.00 ns |

Vitis HLS latency estimates use `loop_tripcount` bounds on runtime
dimensions, so a module row may reflect the maximum supported bound rather
than only the exact `2048 x 2048 x 2048` projection shape.

## Current Limitations

- `ReadB` is the dominant bottleneck: HLS shows II=16 on the
  `fused_tile_fused_io` loop because the `weight_data` m_axi adapter
  defaults to 32-bit data width. Setting `max_widen_bitwidth=512` on that
  port should drop it toward II=1 and unlock the next ~10x of matmul
  throughput.
- The single-attention and full-model reports have a small timing miss of
  -0.04 ns at the 10 ns HLS target (the actual hardware build closes
  cleanly at +0.003 ns once placement/route gets the SLR split right).
- The kernel uses one PE chain because adding a second chain would require
  additional independent top-level AXI bundles or a fork task that fans
  out the weight stream.
- A/B projections still use the tiled fallback because `out_dim=8`.

