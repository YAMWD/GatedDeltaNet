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

## Confirmed next bottleneck → Stage 2

The 15x weight-data reduction came entirely from killing the re-reads. **Weight
bandwidth did not improve (387 → 388 MB/s)** because the weight read
(`loadB`) was **width-demoted to 32-bit**: the matmul shares the `mem_weights`
AXI bundle with the scalar weight readers (rmsnorm / conv / embed / onorm),
which prevents 512-bit widening (confirmed in csynth, II=16, "bit width 32",
and on-card: 388 MB/s ≈ a 32-bit path saturated at 100 MHz). Bursting raised
bytes/transfer but cannot raise bandwidth when each beat is 4 bytes.

**Stage 2:** give the matmul weights a dedicated 512-bit AXI bundle, separate
from the scalar weight readers. Projected: `32.9 GB / 6.4 GB/s ≈ 5 s` weight
time (vs 85 s now) → application runtime ~1.5–2 min, at which point compute
(256 MAC/cycle, FP32, 100 MHz) becomes the limit. Subsequent stages:
precision (BF16) + grid widening + higher clock toward seconds.

## Build note

`hw.cfg`'s `pblock_pe_split.tcl` PRE hook was disabled: it floorplanned the 16
`ProcessingElement` cells of the old chain and errors ("expected 16
ProcessingElement cells, found 0") against this kernel, which has no PE chain.
The lighter kernel places without an explicit pblock under
`SSI_SpreadLogic_high`.
