# Clean 200 MHz build and on-card validation

From any host with the Slurm client (normally `acclhead1`; a running job may
submit too), from the repository root (there is no root Makefile; the target
lives in `c_impl/Makefile`):

```bash
BUILD_EXCLUSIVE=user BUILD_EXCLUDE=acclnode04,acclnode05,harrier make -C c_impl run_hw
```

From inside `c_impl/` the same command is `make run_hw`, and from any directory
`make -C /path/to/GatedDeltaNet/c_impl run_hw`. This submits work and returns;
it does not run Vitis on the login node. `BUILD_EXCLUSIVE=user` keeps other
users' jobs off the build node for the whole link (an Iter76 recommendation,
see *Evidence boundary*); the exclude list drops the two nodes without the
U55C platform and `harrier`, whose node-local `/tmp` was full. `acclnode01`
and `acclnode03` both built the promoted image. With exclusivity, a node that
hosts another user's job is not selected until it is quiet; drop the knob if
the queue matters more than the one-sample divergence it guards against.

## What the command reproduces

- Fresh source/config snapshot and new node-local build directory for every
  invocation. No previous XO, DCP, XCLBIN, or binary RQS is an input.
- Vitis/Vivado 2024.2 and `xilinx_u55c_gen3x16_xdma_3_202210_1`.
- HLS target 200 MHz and exact link target 200 MHz; 48 allocated CPUs,
  192 GiB, 16 synthesis workers and eight implementation threads.
- **Three kernels from the one source file** — `gdn_forward_p` (SLR1),
  `gdn_k_slr0` (SLR0), `gdn_k_slr2` (SLR2) — linked by `hw_f200_p.cfg`: one
  compute unit each, pinned to their dies, one HBM bank per master, seven
  AXI-Stream links; the two recurrent islands in quarter-SLR pblocks on SLR2
  (`apply_p_islands.tcl` / `apply_p_islands_body.tcl`); the kernel-clock XDC
  override that Vitis 2024.2 needs for any target other than 100 MHz.
- `SSI_SpreadLogic_high`, `AlternateCLBRouting`, and pre/post-route
  `AggressiveExplore`. If kernel setup still fails after post-route optimization,
  run **additional** `Explore`; if it still fails, run focused kernel-path-group
  placement, routing and critical-cell optimization (`finish_p_timing.tcl`).
- Native 6/32-step trajectory tests on the three-thread C model of the
  partition (`GDN_PARTITIONED=1`), the BF16 layout check, the integrated HLS
  architecture/II checks over the three kernels (`check_native_bf16_xo.py`),
  and the synthesized recurrent-writer address gate on `gdn_k_slr2.xo`
  (`check_state_writer_addresses.py`) before linking.
- Final route completeness, DRC, bus-skew and DATA/DMA/HBM setup/hold checks
  inside the link (`check_p_final_timing.tcl`); the link aborts on any negative
  slack at the requested period, so no auto-scaled design can reach an image.
  The HBM AXI clock is held to its 450 MHz period like the others;
  `GDN_HBM_MIN_MHZ=<f>` would let the gate accept a bounded auto-scaling of
  that shell clock and is not set by the flow (the promoted image did not need
  it: +0.015 ns).
- Clock-metadata reconciliation (`reconcile_exact_clock.py`), unchanged: it
  patches `DATA_CLK` only when the hook's `exact_clock_gate.tsv` proves every
  clock non-negative at the requested period, never touches the bitstream, and
  fails the build otherwise. On the promoted image it is a no-op — Vitis wrote
  200 MHz itself because no clock missed.
- A separate `afterok` U55C job runs the existing 8/64-token trajectory and
  scale-aware GPU-logit checks, using that submission's host, XCLBIN, Makefile
  and parity checker. It never builds inside the card allocation. The host
  recognises the three-kernel image by its kernel names, starts the two GEMV
  kernels before the control kernel and waits for all three.

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
- `native_gate.live.log`, `xo_gate.log`, `state_address_gate.live.log`: gates.
- `physical_finish.live.log`: the finishing hook's stages and the kernel slack
  after each pass; `exact_clock.log`: the metadata reconciliation verdict.
- `impl_1.runme.log`: complete Vivado implementation log copied at completion.
- `build.phase`, `build.exit`, `oncard.exit`: phase and exit markers.
- `source_snapshot.tar`, `source_hashes.txt`: frozen reproduction inputs.
- `checkpoints/`, `gdn_final_qor/`: saved physical evidence; `reports/`,
  `reports_compile_<kernel>/`: v++ link and per-kernel HLS reports.
- `xclbin.path`, `host.path`: exact successful artifact locations in this tag's
  `artifacts/` directory, isolated from other submissions. The image is
  `gdn_p.xclbin`; `build_manifest.sha256` describes the shipped image and
  `xo_manifest.sha256` the three XOs.
- `on_card/`: smoke/full-run JSON, logits-gate and trajectory reports, timings,
  `performance_summary.json`.

Measured wall time: the Vivado link alone took **5 h 24 min** (build 6568) and
**6 h 03 min** (build 6548); the three XOs compile in about 25 minutes in
parallel and the state-address gate adds about 50 minutes before the link. The
job limit is 48 hours. On-card validation takes about 40 s once a card is
allocated.

## Evidence boundary

The promoted netlist (source `974936d334f999b5…`) closed 200 MHz twice from the
same inputs: build 6548 (image `0326f01b…`, on-card job 6549) and build 6568
(image `39a88f6d…`, job 6569) implemented identically — kernel +0.029 / +0.010
ns at 5.000 ns, DMA +0.003 / +0.009, HBM +0.015 / +0.010, the same routed
utilization to the cell — and both images passed the 8/64-token card gates
with the production vector-gate figures (NRMSE 0.00466 over 2,016,000 logits,
exact trajectories) at 12.227 / 12.256 ms kernel, 12.350 / 12.389 ms TPOT.
Those two links ran through the campaign's own launcher; the production
command `make run_hw` was then run once from a clean snapshot (tag
`hw_20261004_095144_311244`, build 6642 on `acclnode03`, 6 h 57 min job time
including the 50-minute state-address gate, image `0479e581…`) and its card job
6643 passed the 8/64-token gates at 12.255 ms kernel / 12.387 ms TPOT with the
same vector-gate figures — the integrated flow end to end, with the same
slacks as the two campaign links. The long
qualification campaigns — WikiText-2 perplexity, 512-token drift, paired
power — ran on the `0326f01b…` image (jobs 6570–6572) and are not part of the
default 8/64-token tests.

Two caveats carry over from the 150 MHz flow. Physical implementation is
sensitive to netlist and tool changes; a failed timing gate must be diagnosed,
not bypassed. And one 150 MHz link that shared its node with foreign jobs
diverged in the placer, which is why `BUILD_EXCLUSIVE=user` is recommended
even though the two 200 MHz links reproduced each other without it.
