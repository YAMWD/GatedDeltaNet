# Clean 225 MHz build, on-card validation and qualification — one command

From any host with the Slurm client (normally `acclhead1`), from the repository
root (there is no root Makefile; the target lives in `c_impl/Makefile`):

```bash
make -C c_impl run_hw
```

From inside `c_impl/` the same command is `make run_hw`. It submits work and
returns; it does not run Vitis on the login node. On a busy `acclnode01` add
`BUILD_CPUS=8 BUILD_MEM=96G` (the `build` partition caps a node at 96 CPUs and
the links average ≈ 3 cores with a ≈ 74 GiB peak; Iter88's links ran so in
8 h 41 m and 9 h 0 m). `QUAL=0` stops after the card gates.

## What the command runs

Four chained Slurm jobs, each `afterok` the previous one, from one frozen
source snapshot:

1. **Build** (`slurm/hw_build.slurm`, `build` partition): the native 6/32-step
   trajectory gates on the three-thread C model of the partition, the XO
   architecture gate over the three kernels (every reader II=1 at 4.444 ns),
   the synthesized address gates on the state writers and readers of
   `gdn_k_slr2.xo` and on the convolution-tail writer of `gdn_forward_p.xo`,
   the island-name gate; then the three XOs at 225 MHz and the link at a true
   225 MHz by `hw_f225_p_remap.cfg` — one compute unit per kernel pinned to its
   die, the HBM map of `architecture.md` § *Kernel Interface and HBM Mapping*
   (the SLR2 kernel on HBM[2..9], the SLR0 kernel's ports 2–9 on HBM[24..31]),
   seven AXI-Stream links, the two island pblocks, the kernel-clock XDC override,
   `SSI_SpreadLogic_high`, `AlternateCLBRouting`, pre/post-route
   `AggressiveExplore`, the finishing ladder and the exact-clock gate on all
   three clocks (kernel and DMA exact; HBM within the 440 MHz floor, which the
   re-mapped image does not need: +0.039 and +0.069 ns at 450 MHz on two draws),
   route status, DRC, bus skew, the clock-metadata reconciliation; the image
   and its reports are copied back to `diagnostics/<tag>/`.
2. **Card gates** (`slurm/hw_oncard.slurm`, `light`, any node with a free U55C):
   the host is built on that node from the frozen snapshot
   (`slurm/build_host_on_node.sh`), the image hash is checked against the
   build's manifest, and the exact 8- and 64-token trajectory gates run with the
   scale-aware GPU-logit check, reporting kernel and production TPOT medians.
3. **Drift and WikiText-2** (`slurm/hw_robustness.slurm`): a 512-token
   free-running decode against the GPU 512-step reference with the windowed
   drift trend (verdict must be BOUNDED), then teacher-forced WikiText-2 over
   all 62 documents against the GPU score with the 5 % word-perplexity gate on
   the verified-identical workload.
4. **Power** (`slurm/hw_power.slurm`): the paired-power protocol's FPGA arm —
   4,096-token context, 60 s loaded idle, 30 s warm-up, 3 × 60 s of sustained
   teacher-forced decode, the card's sensor sampled once a second — summarized
   as W and J/token (gross and idle-subtracted). `POWER_GPU_ARMS="float32
   bfloat16"` adds the paired A100 arms and the FPGA-vs-GPU comparison; the
   launcher then requests a GPU (it also does when the fixture has to be
   generated).

Results land in `diagnostics/<tag>/`: `build.exit`, `exact_clock.exit`,
`gdn_final_qor/`, `artifacts/.../gdn_p.xclbin`, `on_card/`, `qual/robustness/`
(`card512_trend.txt`, `wikitext_full62_vs_gpu.json`, `robustness.exit` =
"rc1 rc2 rc3 verdict"), `qual/power/` (`fpga_summary.json`, `power.exit`).

## Inputs the clone must regenerate

Beyond the four blobs the root `CLAUDE.md` lists (weights, GPU reference
logits, `.gdnstate`, optional native logits), the qualification needs three
more, all produced by committed scripts on a GPU: the 512-step GPU logit dump
(`QUAL_GPU512`, the 64-step exporter at `--decode-len 512`), the WikiText-2
GPU score of `fixtures_full/wikitext.gdnreq` (`QUAL_WIKITEXT_GPU`,
`scripts/score_wikitext_fixture_gpu.py`), and the 4,096-token power fixture
(`POWER_FIXTURE_DIR`, `scripts/prepare_steady_decode_fixture.py` +
`scripts/export_fpga_handoff_state.py`; the power job generates it itself when
it is missing). The launcher refuses to start the qualification without the
first two.

## Evidence

The Iter88 recipe closed on three independent draws (builds 10168, 10417 with
the call probe compiled in, and 10440), the first and third with identical
slacks to the picosecond. Build 10440 is the first run of this one-command
flow: 7 h 24 m link, gates 47 s, drift and WikiText-2 52 m, power 5 m, every
stage passed on acclnode01, and its image is the production image (8.233 ms
TPOT / 8.141 ms kernel, 0.5895 J/token). `optimization_log.md` (Iter88 and its
promotion entry) carries every number. The 200 MHz predecessor flow is
described in `reproduce_f200.md`.
