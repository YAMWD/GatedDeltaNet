# Weight-Traffic-Optimized Matmul (`gdn_matmul_2d`)

This document describes the matmul kernel that replaced the systolic-chain
GEMM in the `gdn_forward` projection path. It was designed directly from
**on-card profiling**, which showed the live design was memory-bound on
weight HBM traffic, not compute-bound. It supersedes
[systolic_matmul.md](systolic_matmul.md) as the engine `gdn_forward` calls for
the eight large projections; the systolic chain and the tiled fallback remain
in the file (the chain is still used by `gdn_attn_forward` / `gdn_matmul_top`).

## Why the systolic chain was replaced: the hardware profile

The previous design (`gdn_matmul_systolic`, a 1-D PE broadcast/forward chain)
synthesised cleanly and ran the compute at II=1, but the real U55C run was
dominated by weight memory traffic. From `summary.csv` of a baseline
`gdn_forward` run (wikitext, 1 window, ~2048 tokens):

| Weights port (`m_axi_mem_weights`, HBM[1:31]) | Baseline |
|---|---:|
| Total weight data read | **507.8 GB** |
| Number of transfers | 8.00 billion |
| Bytes per transfer | 63 B |
| Transfer efficiency | **1.55 %** |
| Sustained bandwidth | 387 MB/s |
| Application runtime | **1,554,830 ms (25.9 min)** |

Two compounding causes:

1. **~95–128x redundant weight re-reads.** `ReadB` reloaded the entire weight
   tile from HBM for every 16-row token stripe (`num_outer_n = num_rows/16`),
   so each weight was fetched ~128x for a 2048-token prefill. The model's
   weights are only ~5.3 GB, but the kernel read 507 GB.
2. **64-byte non-bursted transfers.** HLS reported "could not analyze pattern"
   on the `ReadB` address expression, so every weight read was a single 512-bit
   beat with full AXI overhead (1.55 % efficiency).

## Design: activation-stationary blocking + bursting

`gdn_matmul_2d` is a loop-nest kernel (no hand-instantiated PE chain). It keeps
the same 16x16 PE grid (256 MAC/cycle, FP32 recurrence broken with
`MM2D_PARTIAL=8` accumulator banks, inner loop II=1), but changes the data
movement:

- **Activation-stationary blocking.** A block of `MM2D_ABLK_ROWS = 256` rows of
  activations is held resident on chip (`localA`, bound to URAM), and each
  weight column is streamed past *all* of them before eviction. A weight is
  therefore fetched once per 256-row block (`ceil(num_rows/256)`) instead of
  once per 16 rows — a **16x** cut in weight traffic at this block size.
- **Burst-friendly loads.** Both `loadA_kp` (activations) and `loadB_kp`
  (weights) are clean contiguous inner loops with a monotonic index and the
  boundary test hoisted out, so HLS infers bursts
  (`[HLS 214-115] burst reads ... inferred`) instead of single beats.
- The weight tile is loaded once per (row-block, column-tile) and reused across
  all `MM2D_ABLK_ROWS/16` row sub-tiles, so its load is amortised over 16x more
  compute and needs no double-buffer.

The 5.76 MB activation block is bound to URAM (`#pragma HLS bind_storage
impl=uram`); left as BRAM it overflowed the device (~107 % BRAM). The current
design uses no URAM elsewhere, so this is free headroom.

### Locations

| File | Role |
|------|------|
| `gdn_model.cpp` (`gdn_matmul_2d`) | Synthesisable kernel; called from `gdn_forward`'s 8 projection sites |
| `gdn_model.cpp` (`gdn_matmul2d_top`) | Standalone HLS top (own AXI bundles) for matmul-only csim/csynth |
| `gdn_matmul2d_test.cpp` | Native-C parity testbench vs `naive_matmul` |
| `test_matmul2d.tcl` | Vitis HLS csynth script (`set_top gdn_matmul2d_top`) |

## On-card results

Built and run on U55C (`make TARGET=hw run_hw`, default wikitext fixture, same
workload as the baseline above). The baseline profile is saved in
`c_impl/profile_baseline/`.

| Metric | Baseline (systolic chain) | `gdn_matmul_2d` | Change |
|---|---:|---:|---:|
| Weight transfers | 8.00 B | 205 M | **39x fewer** |
| **Weight data read** | **507.8 GB** | **32.9 GB** | **15.4x less** |
| Bytes / transfer | 63 B | 160 B | 2.5x larger |
| Weight bandwidth | 387 MB/s | 388 MB/s | ~unchanged |
| **Application runtime** | **25.9 min** | **6.5 min** | **4.0x faster** |
| Kernel time | — | 4.7 min | — |
| Wikitext perplexity | 15.81 (golden) | 15.81 | exact match |

Full-model parity also PASSED on all 10 smoke fixtures natively (max abs diff
~1.5e-4, within the 1e-3 tolerance).

Integrated `gdn_forward` resources dropped (the chain freed BRAM/DSP):
BRAM 13 %, DSP 26 %, URAM 53 % (vs the chain's BRAM 38 %, DSP 31 %, URAM 0).

## Stage 1 → Stage 2: the 512-bit weight read

After Stage 1, the matmul weight **bandwidth had not improved (387 → 388
MB/s)** even though the data volume fell 15x. csynth showed the weight read
(`loadB`) was stuck at **32-bit, II=16**. The cause was NOT the shared bundle
(a dedicated bundle still came out 32-bit) — it was **pointer alignment**: the
kernel received `weight_data + layer_offset`, a runtime float offset HLS cannot
prove is 16-aligned, so it refused to widen the Pack16 reads. The activation
read (`loadA`, a base pointer at offset 0) was already 512-bit.

**Stage 2 fix (two parts):**

1. **Aligned base + integer offset.** `gdn_matmul_2d` now takes the weight
   *base* pointer plus a Pack16-unit offset `w_pack_off`, and indexes
   `weights_p[w_pack_off + ...]`. Indexing a `Pack16*` base by an integer is
   provably 64-byte aligned, so HLS widens `loadB` to **512-bit + burst, II=1**.
2. **Dedicated bundle.** `weight_data_mm` is a separate AXI bundle
   (`mem_weights_mm`) for the matmul weights, aliased to the same HBM blob (the
   host uploads to both buffers; `hw.cfg` maps both to HBM[10:31]). Isolates the
   matmul weight traffic from the scalar readers.

### On-card results (three-point progression, same wikitext run)

| Metric | Baseline | Stage 1 | Stage 2 |
|---|---:|---:|---:|
| Application runtime | 25.9 min | 6.5 min | **5.2 min** |
| Kernel time | — | 4.7 min | **3.45 min** |
| Matmul weight bandwidth | 387 MB/s | 388 MB/s | **5,405 MB/s** (14x) |
| Matmul weight bytes/transfer | 63 B | 160 B | **2,869 B** |
| Matmul weight efficiency | 1.55 % | 2.0 % | **28.1 %** |
| Perplexity | 15.81 | 15.81 | **15.81** |

The 32.8 GB of matmul weight reads now take ~6 s (was ~82 s). Parity preserved.

## Confirmed next bottleneck → Stage 3 (compute + activation memory)

With weights no longer the wall, the kernel's 207 s splits roughly between:

- **Matmul compute** ~107 s — 256 MAC/cycle, FP32, 100 MHz. Now exposed.
- **gmem activation port (HBM[0], single channel)** — 78 GB reads @ 1.48 GB/s
  (~53 s) **plus 7.5 GB of 11-byte writes @ 183 MB/s (~41 s)**; those tiny
  scattered writes are now the worst-efficiency traffic in the design.

**Stage 3 levers:** (a) compute — BF16 + widen the PE grid past 16x16 + raise
the clock (each ~2–5x on the 107 s); (b) activation memory — spread gmem across
multiple HBM channels (it is pinned to HBM[0] today) and Pack16 the output/
residual writes (kill the 11-byte transfers); (c) parallelize the recurrent
attention. These target the path from ~3.5 min toward seconds.

## Build note

`hw.cfg`'s `pblock_pe_split.tcl` PRE hook was disabled: it floorplanned the 16
`ProcessingElement` cells of the old chain and errors ("expected 16
ProcessingElement cells, found 0") against this kernel, which has no PE chain.
The lighter kernel places without an explicit pblock under
`SSI_SpreadLogic_high`.
