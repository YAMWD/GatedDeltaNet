# Clean 150 MHz build and on-card validation

From any host with the Slurm client (normally `acclhead1`; a running job may
submit too), from the repository root (there is no root Makefile; the target
lives in `c_impl/Makefile`):

```bash
BUILD_EXCLUSIVE=user BUILD_NODE=acclnode01 BUILD_EXCLUDE=acclnode04,acclnode05,harrier make -C c_impl run_hw
```

From inside `c_impl/` the same command is `make run_hw`, and from any directory
`make -C /path/to/GatedDeltaNet/c_impl run_hw`. This submits work and returns;
it does not run Vitis on the login node. The
three environment knobs are what the two demonstrated runs used (see *Evidence
boundary*): `BUILD_EXCLUSIVE=user` keeps other users' jobs off the build node
for the whole link, and the node choice reflects the cluster's disk state at the
time of writing — `harrier` and `acclnode03` fail the 60 GiB node-local
preflight in seconds because their `/tmp` is full, so a scheduler-selected build
may die before Vitis starts and take its `afterok` card job with it.

## What the command reproduces

- Fresh source/config snapshot and new node-local build directory for every
  invocation. No previous XO, DCP, XCLBIN, or binary RQS is an input.
- Vitis/Vivado 2024.2 and `xilinx_u55c_gen3x16_xdma_3_202210_1`.
- HLS target 150 MHz and exact link target 150 MHz; 48 allocated CPUs,
  192 GiB, 16 synthesis workers and eight implementation threads.
- The existing 32-port connectivity and physical constraints, including the
  kernel-clock repair and the eight measured pre-place control-fanout repairs.
- `SSI_SpreadLogic_high`, `AlternateCLBRouting`, and pre/post-route
  `AggressiveExplore`. If kernel setup still fails after post-route optimization,
  run **additional** `Explore`; if it still fails, run focused kernel-path-group
  placement, routing and critical-cell optimization. These are sequential
  passes, not a replacement of `AggressiveExplore` with `Explore`.
- Native 6/32-step trajectory tests, the BF16 layout check, integrated HLS
  architecture/II checks, and the synthesized recurrent-writer address gate
  before linking. No extra cosim is introduced by this recipe change.
- Final route completeness, DRC, bus-skew and DATA/DMA/HBM setup/hold checks
  inside the link (`finish_f150_timing.tcl`); the link aborts on any negative
  slack at the requested period, so no auto-scaled design can reach an image.
- Clock-metadata reconciliation. Vitis decides `AUTO-FREQ-SCALING` from the
  timing report it writes *before* the finishing hook runs, so a design the hook
  closed still gets the scaled value written into the XCLBIN (149 MHz on builds
  3987 and 4022). `reconcile_exact_clock.py` patches `DATA_CLK` to the requested
  frequency only when the hook's `exact_clock_gate.tsv` proves every clock
  non-negative at that period, never touches the bitstream, keeps the original
  as `gdn_forward.xclbin.vpl_metadata.bak`, and fails the build otherwise. The
  copied-back XCLBIN therefore reports exactly 150 MHz or the build fails.
- A separate `afterok` U55C job runs the existing 8/64-token trajectory and
  scale-aware GPU-logit checks, using that submission's host, XCLBIN, Makefile
  and parity checker. It never builds inside the card allocation.

## Required external data

These existing, large evaluation artifacts are not stored in Git. Paths below
are relative to `c_impl/`; an absolute path override is also supported.

| Make variable | Default |
|---|---|
| `WEIGHTS` | `artifacts/gdn-1.3b-bf16w.gdnw` |
| `DECODE_FIXTURE` | `fixtures_decode/decode.gdnreq` |
| `DECODE_STATE` | `fixtures_decode/decode_ex0_native_bf16_product.gdnstate` |
| `DECODE_GOLDEN` | `results_decode_golden/decode_native_bf16_product.decode.json` |
| `GPU_LOGITS_REFERENCE` | `artifacts/decode_native_bf16_product_64.gdnlog` |

The FP32 checkpoint is not a substitute for the BF16-exact weight artifact.
`LOGITS_REFERENCE` remains an optional native diagnostic, disabled by default;
hardware/native bit-exact logits are not the accepted production criterion.

## Logs and outputs

The submitter prints both job IDs and an absolute shared diagnostics directory:
`c_impl/diagnostics/hw_<timestamp>_<pid>/`.

- `build.live.log`: live v++ output; `build.slurm-<job>.log`: job wrapper.
- `native_gate.live.log`, `state_address_gate.live.log`: validation gates.
- `physical_finish.live.log`: the finishing hook's stages and the kernel slack
  after each pass; `exact_clock.log`: the metadata reconciliation verdict.
- `impl_1.runme.log`: complete Vivado implementation log copied at completion.
- `build.phase`, `build.exit`, `oncard.exit`: phase and exit markers.
- `source_snapshot.tar`, `source_hashes.txt`: frozen reproduction inputs.
- `checkpoints/`, `gdn_final_qor/`: saved physical evidence.
- `xclbin.path`, `host.path`: exact successful artifact locations in this tag's
  `artifacts/` directory, isolated from other submissions. There,
  `build_manifest.sha256` describes the shipped (reconciled) image and
  `build_manifest.pre_reconcile.txt` keeps the digest of the vpl-written file.
- `on_card/`: smoke/full-run JSON, logits-gate and trajectory reports, timings.

Measured wall time for the two passing runs: **12 h 24 min** (build 4022) and
**12 h 28 min** (build 3987) from job start to image, on `acclnode01` with 48
CPUs, excluding queue time; the failing draw 3 took 10 h 12 min. The job limit
is 48 hours. On-card validation takes about 40 s once a card is allocated.

## Evidence boundary

Iter75f's checkpoint sequence closed at 150 MHz and its packaged image passed
on-card validation (jobs 3955–3960; TPOT 16.256 ms). This recipe consolidates
those passes into a fresh production link, and **two complete fresh runs have
verified it** (Iter76, 2026-09-12): builds 3987 and 4022 reproduced build
3751's implementation at every Vivado phase checksum, closed 150 MHz in-link
(kernel setup 0.000 / hold +0.002 ns, DMA +0.003, HBM +0.052), and their images
passed the 8/64-token card gates with the same logits gate figures as the
Iter75f image (NRMSE 0.00466 over 2,016,000 logits, zero argmax mismatches) at
16.131 ms kernel / 16.255 ms TPOT. Build 4022 ran the whole chain, including
metadata reconciliation, with no manual step. Both runs were launched through
the launcher's own preparation and submission steps with the arguments
`make run_hw` resolves to, on a user-exclusive `acclnode01`.

Two caveats. First, one identical-input link (build 3968, draw 3) diverged
inside the placer's physical synthesis while other users' jobs shared its node,
and missed by 0.045 ns; the three quiet-node links reproduced each other
exactly. That is one divergent sample, so node exclusivity is a recommendation
grounded in the evidence, not a proven mechanism. Second, physical
implementation is sensitive to netlist/tool changes; a failed timing gate must
be diagnosed, not bypassed. The default 8/64-token tests do not rerun the longer
WikiText, 512-token drift or power qualification campaigns; those were run on
the Iter75f image, whose closed design is checksum-identical to these builds'.
The complete record is the Iter76 entries in `optimization_log.md`.
