# Decode-Stage Roadmap (and why the PE grid stays 16×16)

This document records the pivot from prefill to **decode** as the optimization
target, why the prefill compute-widening (Phase C) was reverted, and the plan
for decode latency. It follows the activation-memory work in
[optimization_log.md](optimization_log.md) and the matmul in
[weight_stationary_matmul.md](weight_stationary_matmul.md).

**Phase 0 (done):** the premise below is GPU-validated in
[decode_premise.md](decode_premise.md) (GDN TPOT flat ~35 ms/token vs a
transformer's O(n)), and on-card decode is **bit-exact to the GPU golden over
64 tokens** with a re-prefill TPOT baseline in
[decode_correctness.md](decode_correctness.md) — every step of this roadmap is
gated by that correctness check.

## Where we landed (prefill benchmark, on U55C, wikitext 2048 tokens)

| Stage | App runtime | Kernel | Lever |
|-------|------------:|-------:|-------|
| baseline (1-D chain) | 25.9 min | — | — |
| Stage 1 | 6.5 min | — | weight-stationary blocking (kill 95× weight re-reads) |
| Stage 2 | 5.2 min | 207 s | 512-bit bursted weight reads (aligned base) |
| Phase A | 4.4 min | 153 s | Pack16-widen rmsnorm / conv / output_norm (kill 11-byte writes) |
| Phase B | 4.2 min | 141 s | activations across 3 HBM channels (gmem_x/qkv/mlp) |

Net: **25.9 min → 4.2 min (~6.2×); kernel 207 → 141 s.** The weight/activation
memory wall — the original ~22-minute bottleneck — is solved. After Phase B the
kernel is **compute-bound**: matmul ≈ 107 s ≈ 76% of the 141 s kernel, with the
sequential recurrent attention (~39 s) the next floor.

## Phase C (PE-grid widening) — attempted, then REVERTED

Phase C widened `gdn_matmul_2d`'s 16×16 grid (256 MAC/cycle) to cut the prefill
compute floor. Two configs were built; both failed on **infrastructure, not
logic** (csynth II=1 and parity PASS each time):

- **32×32 (1024 MAC/cycle):** BRAM over-utilization — 4054 RAMB18 > 4032
  available (`localB` doubled to 512 banks at 32 cols).
- **32×16 (512 MAC/cycle):** routed-stage failure — **SLR1–2 crossing (SLL)
  congestion at 102%**. The unrolled broadcast grid spanning >1 SLR demands more
  inter-die routing than the boundary provides; closing it needs a pblock
  floorplan + a grid restructure into per-SLR registered sub-grids.

It was reverted (`git reset` to Phase B) because:

1. **It's a prefill lever, not a decode one** (see below) — and decode is now
   the target.
2. Even for prefill, its payoff is capped by the ~39 s sequential attention
   (kernel floor ~50–60 s regardless of grid width), for high routing risk.

The 16×16 grid is retained. Fittable-32×32 notes (for the record, if ever
needed for prefill): shard `localB` half-BRAM/half-URAM to clear the BRAM wall
(keeps II=1 load), then pblock + drop `SSI_SpreadLogic_high` and register the
SLR-boundary operand paths to clear the SLL wall.

## Why decode is the right target

Decode (autoregressive, 1 token/step) is the FPGA-favorable regime:

- It is **single-stream, low-latency, memory-bound** — a GPU needs large batches
  to fill its bandwidth and sits idle on one decode stream; a custom HBM
  dataflow runs flat-out. (Prefill / TTFT is compute-bound and GPU-favorable —
  out of scope here; this targets **TPOT**, time-per-output-token.)
- **GatedDeltaNet is ideal for it:** linear attention = a constant-size
  recurrent state → **O(1) compute per token, no growing KV cache.** The
  attention is nearly free per decode token (read ~2 MB state, update, write).

## The decode bottleneck: weight bandwidth, NOT compute

At decode every matmul is a **GEMV** (num_rows=1): each weight is used exactly
once → **1 MAC per weight read**. Compute per token ≈ 1.4 G MACs (trivial). The
entire cost is **reading every weight once per token**:

```
decode latency/token ≈ weight_bytes / weight_bandwidth
   FP32 weights ≈ 5.6 GB   |   INT8 weights ≈ 1.4 GB
```

The 16×16 grid is already **over-provisioned** for this:

```
256 MAC/cycle × 100 MHz = 2.56e10 MAC/s = can consume ~102 GB/s of weights
current weight port delivers:                                 5.4 GB/s
```

So the grid can chew ~100 GB/s but is fed only 5.4 — **starved ~19×.** Widening
the grid (Phase C) does nothing for decode; it would idle harder and waste more
of the array (a 2-D grid on a 1-row GEMV wastes 31/32 of its rows). **Bandwidth
is the lever, not MAC/cycle** — which is exactly why the grid stays 16×16.

## Decode roadmap (target: well under 1 s/token → ms/token)

| Step | Weight BW | decode/token (FP32) | bottleneck |
|------|----------:|--------------------:|-----------|
| Baseline (re-prefill, O(n)) | — | 6.95 s median (4.2→9.7 s, grows) | re-forwards whole prefix |
| Step 1 — single-token, state persistence (GEMM at 1 row) | ~2.2 GB/s eff. | 2.56 s, flat (on-card, bit-exact) | GEMM-shaped at num_rows=1 |
| Disaggregated decode-only GEMV (coupled read+MAC) | 2.87 GB/s | 1.95 s, flat | weight burst broken per-output |
| **Stage 1 DONE — decoupled reader→MAC (dataflow), ON-CARD** | **5.30 GB/s (83% of 1 port)** | **1.54 s, FLAT (1.66× over Step 1)** | single-port floor: 1.06 s gemv + 0.48 s non-gemv |
| Stage 2 — N parallel HBM weight readers | ~100 GB/s | ~55 ms | memory (multi-master) |
| Multi-channel parallel weight readers (~8 masters) | ~100 GB/s | ~55 ms | memory (16×16 grid keeps pace) |
| + INT8 weights (5.6 → 1.4 GB) | ~100 GB/s | ~14 ms | memory |
| Toward HBM aggregate (~460 GB/s) | ~400 GB/s | ~3–10 ms | compute (only here widen) |

**Architecture pivot — disaggregated decode-only** (full design:
[decode_disaggregated_gemv.md](decode_disaggregated_gemv.md)). The GPU prefills
the prompt and exports the constant-size recurrent+conv state to disk
(`scripts/export_gdn_state.py` → `.gdnstate`, ~50 MB); the FPGA loads it and
decodes through a new activation-stationary **`gdn_gemv`** engine. The prefill
GEMM (`gdn_matmul_2d`), the `decode_flags` mode mux, and all prefill code were
**removed** — the FPGA has one mode, decode. Native csim is bit-exact to the GPU
golden over 32 tokens; on-card synthesis is in progress. This builds Step 1's
unbuilt "GEMV datapath" half *and* frees the whole die from the prefill engine,
de-risking the multi-reader (Step 2) routing.

Work, in order:

1. **GEMV decode datapath — DONE (native bit-exact).** `gdn_gemv`: activation
   vector resident on-chip, weights streamed contiguously one output-row at a
   time (II=1 burst), adder-tree + partial banks. Recovers 2.56 s → ~1.0 s
   single-port floor. Built as the decode-only kernel (no GEMM, no flags).
2. **Multi-channel weight readers** — the dominant lever. One 512-bit master at
   100 MHz = 6.4 GB/s; 8+ masters across HBM channels aggregate toward HBM's
   ~460 GB/s. (Opposite of prefill, where one reused master sufficed.)
3. **INT8 weights** — 4× less data (5.6 → 1.4 GB), the biggest single reduction;
   near-lossless with calibration.
4. **Keep the kernel resident across tokens** — loop the decode step on-device
   feeding back the recurrent state, to avoid the ~100 µs/token PCIe launch.

Only Step 3 of this is shared with the prefill work (the Stage 2 512-bit
weight-read pattern). Steps 1, 2, 4 are decode-specific and not yet built. The
grid (16×16) and the prefill activation-memory work (Stage 1, Phase A/B) are
retained but are not on the decode critical path.
