# GatedDeltaNet HLS Accelerator (`c_impl/`)

Vitis HLS implementation of GatedDeltaNet-1.3B inference, targeting the
Xilinx Alveo U55C (`xcu55c-fsvh2892-2L-e`) at 100 MHz. The C source in this
directory is the **primary HLS synthesis target** — the surrounding Python in
the parent repo serves as a golden reference for parity verification.

The model parameters are fixed to the GatedDeltaNet-1.3B configuration:
hidden = 2048, 8 heads × 256 head-dim, 24 layers, conv kernel = 4,
max sequence length 2048, vocab 32 000.

## What's in this directory

### Synthesisable C
| File | Purpose |
|------|---------|
| `gdn_model.h` | Public types (`GDNModel`, `GDNRunState`, weight header) and prototypes |
| `gdn_model.c` | All HLS-synthesisable compute: top functions `gdn_forward` (full 24-layer) and `gdn_attn_forward` (single-layer attention block), plus submodules — embed, RMSNorm, tiled matmul, depthwise conv1d + SiLU, recurrent attention with persistent BRAM state, output norm + gate, SwiGLU |
| `Makefile` | Builds the native testbenches with `gcc -O3 -std=c11`, no BLAS |

### Host testbenches
| File | Purpose |
|------|---------|
| `gdn_eval.c` | Drives full multi-layer inference end-to-end. Reads a `.gdnw` weight blob and a `.gdnreq` fixture, writes a JSON results file. Used inside `test.tcl` for cosim. |
| `gdn_attn_test.c` | Drives a **single attention layer** in isolation against a `.gdnblk` fixture (input + golden output for one transformer block). Used inside `test_single_GDN_attn.tcl`. |

### Vitis HLS scripts
| File | Top function | Target | What it runs |
|------|--------------|--------|--------------|
| `test_single_GDN_attn.tcl` | `gdn_attn_forward` | Alveo U55C (`xcu55c-fsvh2892-2L-e`) | csim + csynth, single layer — **primary target for optimisation work** |
| `test.tcl` | `gdn_forward` | Alveo U55C | csim + csynth + cosim, full 24-layer model |

### Parity testbench
| File | Purpose |
|------|---------|
| `test_parity.sh` | Rebuild + run `gdn_eval` on every fixture under `fixtures_smoke/`, then `scripts/check_gdn_c_parity.py` diffs against `results_smoke_python/`. Tolerance 1 × 10⁻³ (observed diffs ~1 × 10⁻⁵). |

### Data formats
| Suffix | Used by | Layout |
|--------|---------|--------|
| `.gdnw` | both testbenches | 60-byte `GDNWeightHeader` followed by all FP32 weights in layer order. ~5.6 GB for GDN-1.3B. |
| `.gdnreq` | `gdn_eval` | Pretokenized eval fixture: header + int32 token IDs + golden log-probs / scores. |
| `.gdnblk` | `gdn_attn_test` | Single-block attention fixture: header + post-RMSNorm input + pre-residual golden output. |

### Documentation
| File | Content |
|------|---------|
| [`doc/architecture.md`](doc/architecture.md) | Top-level overview, file map, AXI bundle topology, optimisation history |
| [`doc/tiled_matmul.md`](doc/tiled_matmul.md) | `gdn_matmul`: tile strategy, manual flatten + explicit fadd tree, latency breakdown |
| [`doc/depthwise_conv.md`](doc/depthwise_conv.md) | `gdn_depthwise_conv_silu`: pre-buffered weights, sliding window, two-phase split |
| [`doc/recurrent_attention.md`](doc/recurrent_attention.md) | `gdn_recurrent_attention`: persistent BRAM state, fused two-pass, P_K=16, delta_drain split, tree-reduce, q/k bundle split |
| [`doc/output_norm.md`](doc/output_norm.md) | `gdn_output_norm_and_gate`: on-chip buffers, tree-reduced sum-of-squares |
| [`doc/optimization_log.md`](doc/optimization_log.md) | Iteration-by-iteration log v0 → v7 with concrete latency / II / resource numbers |

## Build & run

### Native C (parity testing)
```bash
make -C c_impl                                  # builds gdn_eval + gdn_attn_test
./gdn_attn_test artifacts/gdn-1.3b-f32.gdnw fixtures_block/block0_attn.gdnblk
./gdn_eval     artifacts/gdn-1.3b-f32.gdnw fixtures_smoke/piqa.gdnreq results/piqa_c.json
```

### Vitis HLS (synthesis)
```bash
cd c_impl
vitis_hls -f test_single_GDN_attn.tcl     # primary: Alveo U55C, single-layer attention (csim + csynth)
vitis_hls -f test.tcl                     # full 24-layer model on U55C (csim + csynth + cosim)
```

`test_single_GDN_attn.tcl` writes its csynth report to
`GDN_single_attn/solution2/syn/report/csynth.rpt`.

### End-to-end parity sweep
```bash
cd c_impl && bash test_parity.sh
```

This rebuilds `gdn_eval`, runs every `fixtures_smoke/*.gdnreq` through it, and
diffs against `results_smoke_python/`. Exit code 0 means all task scores agree
within tolerance.

## Generating fixtures and weights

Run from the repo root with the project's Python environment.

```bash
# One-time: dump weights into the flat .gdnw format
python scripts/export_gdn_c.py weights \
    --output c_impl/artifacts/gdn-1.3b-f32.gdnw

# Pretokenised eval fixtures
python scripts/export_gdn_c.py fixtures \
    --tasks piqa hellaswag winogrande arc_easy arc_challenge \
            social_iqa boolq lambada_openai wikitext \
    --output-dir c_impl/fixtures_smoke

# Single-block attention fixture (input + golden output for one layer)
python scripts/export_block_fixture.py \
    --layer 0 \
    --output c_impl/fixtures_block/block0_attn.gdnblk

# Python golden results for parity check
python scripts/compare_gdn_c.py \
    --fixture c_impl/fixtures_smoke/piqa.gdnreq \
    --output  c_impl/results_smoke_python/piqa.json \
    --device cuda --dtype float32
```

## Current synthesis results (`gdn_attn_forward` v7, U55C @ 100 MHz)

From `GDN_single_attn/solution2/syn/report/csynth.rpt`, target
`xcu55c-fsvh2892-2L-e`, Vitis HLS 2022.1 (post v7 optimisations):

| Metric | Value |
|--------|------:|
| Top-level latency | **141.03 G cycles** (1.41 s) |
| Timing slack at target | **0.00 ns** (closes timing) |
| II violations | **0** |
| BRAM_18K | 322 (7 %) |
| DSP | 1042 (11 %) |
| LUT | 237 k (18 %) |
| FF | 210 k (8 %) |
| URAM | 0 |

Per-submodule latency:

| Submodule | Per-call cycles | Notes |
|-----------|----------------:|-------|
| `gdn_matmul` (instance 1) | 20.105 G | gmem-bundle output |
| `gdn_matmul` (instance 2) | 20.121 G | replicated for `mem_q` / `mem_k` outputs |
| `gdn_depthwise_conv_silu` (×2 instances, called for Q/K and V) | 8.75 M | 32 BRAM, 36/6 DSP |
| `gdn_recurrent_attention` | 157.29 M | 258 BRAM (state), 494 DSP |
| `gdn_output_norm_and_gate` | 17.24 M | 13 DSP |
| `attn_conv_copy_q/k/v` | 4.19 M each | DRAM→DRAM streaming copies |

All inner pipelines hit II=1 except `log_generic_float_s` (an HLS-internal
FP-log helper used by `gdn_softplus`) which carries an intrinsic II marker
with **positive slack** (+1.38 ns) — meets timing, not a violation.

Re-run `vitis_hls -f test_single_GDN_attn.tcl` after any source change to
refresh the report.

| Submodule | Per-call latency | II of inner pipeline |
|-----------|-----------------:|---------------------:|
| `gdn_matmul` (×7 calls) | 20.18 G cyc | mm_comp_rc II=1 |
| `gdn_depthwise_conv_silu` (×3 calls) | 8.76 M cyc | conv_load II=1, conv_compute II=1 |
| `gdn_recurrent_attention` (×1 call) | 157.29 M cyc | load_qk II=1, all phases II=1 |
| `gdn_output_norm_and_gate` (×1 call) | 17.3 M cyc | onorm_sq II=1, onorm_gate II=1 |

| Test | Result |
|------|--------|
| `gdn_attn_test` (single-layer parity) | **PASS** (max abs diff 1.2 × 10⁻⁶) |
| `test_parity.sh` (full 24-layer, 9 fixtures) | **PASS** (worst diff 6.22 × 10⁻⁵) |

See [`doc/optimization_log.md`](doc/optimization_log.md) for the iterative
journey from the v0 baseline (190.96 G cycles, 7 II violations) to v7, and the
list of follow-ups (streaming GEMM, dataflow at `gdn_attn_forward` scope) that
would unlock the next 2–3× on the matmul-dominated runtime.

## Dependencies

- **HLS**: Vitis HLS 2022.1.
- **Native C build**: any C11 compiler. No BLAS or math libraries beyond
  `libm` (already linked by the Makefile).
- **Python golden reference (parity only)**: see the parent repo's `Dockerfile`.
  Key pins: PyTorch 2.3.1 + CUDA 12.1, Lightning 2.1.2, Triton 2.3.0,
  flash-linear-attention (FLA), lm-eval 0.4.1.
