# c_impl/ — HLS accelerator internals

Loads only when working under `c_impl/`. Project-wide guidance (the real-data principle, build
commands, the iteration workflow, commit discipline, current status) stays in the root `CLAUDE.md`.
`doc/architecture.md` is the authoritative spec (currently the **150 MHz production image of the Iter76
flow**: the Iter67c datapath — all-BF16 weights and state, a native `ap_float<16,8>` product,
free-running cluster pipelines, on-chip strict argmax, five-phase II=1 recurrent read — plus the Iter73b2/75b
state-writer and cluster-emit changes, timing-closed at 150 MHz; 16.255 ms/token production TPOT /
16.131 ms kernel on card. Iter67c, 24.208 / 24.099 ms at 100 MHz, is the predecessor). `README.md`
in this directory is current — it was rewritten for the decode-only design; the earlier note calling it
stale was wrong.

## Files

- **`gdn_model.h`** — dimensions, the workspace layout, and the topology constants.
  - `GEMV_CHANNELS` (32), `GEMV_CLUSTERS` (16), `GEMV_CHANNELS_PER_CLUSTER` (2). **Cross-cutting
    invariant:** `GEMV_CHANNELS` must equal the number of `weight_data_mm*` kernel args, the number of
    host shard BOs, *and* the `sp=` groups in `hw.cfg`. Changing it is a four-file edit.
  - `GDN_WS_OFF_*` / `GDN_WSF_*` — the packed workspace: 15 activation/state buffers live in **one**
    `HBM[0]` allocation instead of 15 kernel args (which had jammed `control_s_axi`). Every offset is
    16-float (512-bit) aligned so `max_widen_bitwidth=512` still applies. The kernel, the csim host
    (`gdn_run_state_init`), and `host.cpp` all derive from this one layout; `static_assert`s in
    `gdn_model.cpp` tie `GDN_WSF_*` to the `GDN_*` dims so it cannot drift silently.
  - `GDN_RECURRENT_STATE_PORTS` / `_FIRST_PORT` — the Iter37 experiment that stripes the recurrent
    state over the tails of weight shards 28–31 (those masters are idle during recurrent attention).
    The external `.gdnstate` format is unchanged; striping happens at upload.
- **`gdn_model.cpp`** — all synthesizable compute plus the host-side shard builders.
  - `gdn_forward` — the only kernel top. 24 layers, **one token per call**; restores recurrent+conv
    state at entry and saves it at exit. Args: `workspace`, `aux_weights`, and `weight_data_mm0..31`.
  - **32-port GEMV dataflow** (the decode engine, activation-stationary — weights stream from HBM
    once and are never cached): `gemv32_load_x_and_w0` → 32× `gemv32_mm2s` readers → 16×
    `gemv32_cluster2` two-port clusters (native `ap_float<16,8>` BF16 products, RNE-rounded to BF16 and widened into
    FP32 dot16 trees — zero FP32 multipliers; each cluster has its own activation copy) → SLR-local
    `gemv32_collect6`/`gemv32_collect4` → `gemv32_collect_final` → `gemv32_store` (restores natural
    output-row order and handles `lm_head`'s partial final pack of 1000 rows/channel). The
    hierarchical, SLR-local collector tree exists **because a single global collector would not
    route** — see the "high-fanout dataflow" note in `doc/optimization_log.md`.
  - The retired 8-port GEMV that once sat inside `#if 0` has been **deleted**; `gdn_gemv` is the only GEMV engine
    in the file. Do not look for it.
  - Other submodules: `gdn_rmsnorm_rows`, `gdn_gemv_tiny` (the two tiny a/b gate projections),
    `gdn_depthwise_conv_silu`, `gdn_recurrent_attention` (gated delta rule, head-local fused state
    buffer), `gdn_output_norm_and_gate`, `gdn_swiglu_inplace`.
  - **No embedding-lookup or argmax module exists, by design.** The 32000×2048 embedding table stays
    in *host* memory and `host.cpp` writes the selected row straight into `workspace[X]`. The greedy
    argmax is *fused into the LM-head store stage* (`gemv32_argmax_*` labels inside `gemv32_store`),
    which reduces the reorder buffer directly to a token id — so 32000 logits are never materialized
    off-chip.
  - Pragmas: `array_partition`, `pipeline II=1`, `unroll`, `dataflow`; every loop carries
    `loop_tripcount`. `memcpy`/`memset` in synthesized paths are written as explicit labeled loops so
    latency estimates are real. Reductions use explicit balanced adder trees — HLS emits a *serial*
    fadd chain for `#pragma HLS unroll` on a `sum +=` loop.
- **`gdn_eval.cpp`** — decode-only native testbench (loads weights + `.gdnstate`, runs the decode
  loop, writes/checks JSON). It also applies the same recurrent-state striping as the on-card host, so
  csim and hardware see identical layouts. `gdn_attn_test.cpp` / `gdn_matmul_test.cpp` /
  `gdn_matmul2d_test.cpp` remain on disk but are **retired** (their tops were removed).
- **`host.cpp`** — XRT host: builds the 32 compact weight shards and the compact aux-weight image from
  the flat blob, allocates one `xrt::bo` per kernel arg on disjoint HBM banks, uploads the 32 packed-BF16 shards (2.799 GB, exactly the dense weight bytes with no padding) plus the aux image via
  `sync_bo_chunked`, loads the `.gdnstate` into the resident state BOs, then decodes token-by-token,
  emitting the same JSON schema as `gdn_eval`. **`kSyncChunk` is 8 MiB**: a 16 MiB `bo.sync()` at a
  nonzero workspace offset returns `EINVAL` on this XRT — a host-only bug that looked like a kernel
  failure.
- **`hw_iter*.cfg`, `apply_iter*.tcl`, `check_*.tcl`** (there is no `hw.cfg`; the link template is `HW_CFG_TEMPLATE` in the Makefile, default `hw_f150.cfg`; `hw_iter66e_frp_unpair_f100.cfg` is the 100 MHz predecessor. Untracked `build_iter*.sh`/`hw_iter*.cfg` in the tree are in-flight or rejected experiments — check the log before reusing one)
  — the per-iteration build bundle; see the root `CLAUDE.md` for how they fit together.
  `hls_gdn_forward.tcl` is the HLS pre-TCL shared by `test.tcl` and `v++ -c`.
  `pblock_pe_split.tcl` is the disabled prefill floorplan, kept for reference.
  `probe_alloc.cpp` diagnosed HBM `sp=` range overlaps (which surface as `std::bad_alloc`); it was deleted in `dfb977c5f` and is recoverable from Git history.
- **`microbench/gemv_tile/`** — a *separate* standalone kernel for characterizing the HBM ceiling.
  Its results are not `gdn_forward` results.
- **Formats**: `.gdnw` (5.87 GB FP32-container blob whose values are BF16-exact; `host.cpp` packs it into BF16 shards at upload), `.gdnstate` (GPU-exported recurrent+conv decode
  state, ~50 MB — the prefill→decode handoff), `.gdnreq` (pretokenized eval fixtures), `.gdnblk`
  (retired single-layer attention fixture). `.gdnw` and `.gdnstate` are gitignored and regenerable;
  generate both locally before running any correctness gate.
- **`doc/`** — see the document-status table in the root `CLAUDE.md` before citing any of it.
