# Disaggregated Decode-Only Accelerator (GEMV datapath)

**Status:** Historical GEMV scaling and optimization record. The current
production design is the integrated Iter36 32-port/16-cluster,
activation-resident, head-local-state kernel documented in
[architecture.md](architecture.md).
The standalone 32-port microbenchmark remains documented separately in
[`../microbench/gemv_tile/README.md`](../microbench/gemv_tile/README.md).

The final integrated U55C image routes at 100 MHz with zero failed/unrouted
nets and zero overlaps. It passes exact 64-token parity at 59.578 ms/token mean
latency, 2.04x faster than the 121.4 ms eight-port baseline. Sections below
retain the progression that led to that design; older statements describing an
eight-port kernel as “current” are historical in their section context.

This documents the pivot to a **decode-only** GatedDeltaNet accelerator: the GPU
prefills the prompt and exports a constant-size recurrent + conv state to disk;
the FPGA loads that state and decodes token-by-token through a new **GEMV**
datapath. The prefill GEMM, the prefill/decode mode flags, and all prefill code
were removed — the FPGA's only job is decode.

This supersedes the intermediate dual-mode design, which kept the prefill GEMM
and selected decode through a `decode_flags` mux.

## 1. Why disaggregate

Decode and prefill want opposite hardware, and they never run at the same time:

| | Prefill (rows ≫ 1) | Decode (1 token) |
|---|---|---|
| Compute | GEMM, compute-bound, batchable | GEMV, **memory-bound**, single-stream |
| Best device | GPU (high FLOPs, big batches) | FPGA (flat low-latency HBM dataflow) |
| Weight reuse | each weight × many rows | each weight × **1 MAC** |

GatedDeltaNet makes the split clean because it is a **linear-attention
(recurrent)** model: the entire prompt compresses into a **fixed-size** state,
independent of prompt length. The GPU→FPGA handoff is therefore a constant
**~50 MB**, not a growing KV cache:

```
recurrent_state : num_layers × H × K × V  fp32 = 24×8×256×256×4 = 48 MB
conv tails      : num_layers × {q,k,v} × (W-1) × hidden fp32 ≈ 1.7 MB
```

Over PCIe gen3×16 (~16 GB/s) that handoff is **~3 ms, once per request** —
negligible against decode. (A transformer would ship a context-growing KV cache,
often making disaggregation a loss; for GDN it is free.)

## 2. The GPU→FPGA state contract (`.gdnstate`)

`scripts/export_gdn_state.py` runs the FLA GPU model over an (arbitrary) prompt
with `use_cache=True`, pulls each layer's `(conv_state_q/k/v, recurrent_state)`
from the cache, and writes a flat blob. Layout derivation (validated bit-exact):

- **recurrent_state** — FLA `final_state` is `[N,H,K,V]` row-major, V contiguous.
  The FPGA stores `state[h][k][v] = recurrent_state[(h·K+k)·V+v]` — **identical,
  no transpose.**
- **conv_state** — FLA `cache` is `[N,D,W]` (newest at index W−1). The FPGA conv
  tail is `[W-1][D]` with `tail[r][d] = cache[0][d][r+1]` (drop the oldest of the
  W cached inputs; transpose D↔W). Per layer the conv order is q(0), k(1), v(2),
  matching the `head_buffer` slot `(layer·3 + conv)`.

Blob format (little-endian):

```
"GDNSTAT1" (8B)
u32 ×9 : version, num_layers, H, K, V, hidden, W(conv_size), prompt_len, seed_token
i32 ×prompt_len : prompt ids (informational)
f32 : Section A — recurrent_state, all layers   → recurrent_state BO (48 MB)
f32 : Section B — conv tails, all layers         → head_buffer BO (~1.7 MB)
```

`seed_token` is the argmax of the prompt's last-position logits — the **first
generated token**. The FPGA decode produces tokens 1..N-1 from that seed; the
full trajectory `[seed, …]` is checked against the committed golden.

The blob is large + regenerable, so it is gitignored (like the weight blob).

## 3. The GEMV engine (`gdn_gemv`)

Decode is a GEMV: `out[o] = Σ_k in[k]·W[o][k]`, one token, each weight read once.
The design follows the SOTA bandwidth-bound FPGA-HBM decode engines (see
§8 References) — **FlightLLM** [1] (LLaMA2-7B on one U280, "always-on-chip
decode"), **DFX** [2] (single-token GPT core), and the embedded-FPGA decode work
[3] (KV260, multiplier + adder tree at 85% of the decode-bandwidth limit):

1. **Activation-stationary.** The single activation vector (≤ 5632 fp32 = 22 KB)
   is loaded once into on-chip `a_loc` and reused for every output; **weights
   stream from HBM once and are never cached** — the opposite of the prefill
   weight-stationary GEMM (which reused each weight across 256 rows).
2. **Contiguous weight stream.** Weights are row-major `[out_dim][in_dim]`, so one
   output's row `W[o][:]` is contiguous. Scanning **one output at a time** keeps
   the 512-bit weight port reading one Pack16 (16 fp32) beat/cycle at **II=1** — a
   clean burst. (Interleaving 16 output rows would scatter the port to II=16.)
3. **Adder-tree + partial banks.** Each beat does 16 multiplies reduced by a
   combinational tree; the running sum rotates across `GEMV_PARTIAL=8` banks to
   hide FP32 fadd latency so the k-loop holds II=1.
4. **Pack16 output.** 16 dot-products are buffered and written as one 512-bit
   beat (every projection's `out_dim` is a multiple of 16).

Per-token cost is then `weight_bytes / port_bandwidth`: one 512-bit master at
100 MHz = 6.4 GB/s → ~1 s for the 5.6 GB fp32 blob — the **single-port GEMV
floor**, vs the 2.56 s the prefill GEMM spent at num_rows=1 (255/256 of its 16×16
array idle, weights not streamed back-to-back). The next lever (roadmap Step 2)
replicates the reader across N HBM channels toward HBM's ~460 GB/s aggregate.

All eight large projections (q/k/v/gate/o, mlp gate/up/down) use `gdn_gemv`; the
two tiny gate projections (a/b, out_dim=8) use `gdn_gemv_tiny`.

## 4. What was removed

- **`gdn_matmul_2d`**, `gdn_matmul_tiled`, and their prefill test tops.
- **`decode_flags` / `GDN_DECODE_RESTORE` / `GDN_DECODE_SAVE`** — there is one
  mode. The conv and recurrent attention always restore state at the start and
  save at the end; `gdn_forward` always forwards exactly one token.
- **Prefill paths** — `gdn_attn_forward`, `gdn_attn_forward_layer`, the old
  systolic test top `gdn_matmul_top`, and the prefill host wrappers
  (`gdn_forward_host`, `gdn_prefill_save_host`, `gdn_forward_flags_host`).
- **`gdn_eval` prefill modes** — the MC/LL perplexity scoring and the re-prefill
  decode benchmark; `gdn_eval` is now a decode-only csim driven by
  `--decode-from-state <blob>`.

`gdn_forward` lost its `decode_flags` argument — it is now a 20-arg decode-only
kernel top. `hw.cfg` is unchanged (the flag was an s_axilite scalar, not a memory
port; all `sp=` port names are identical).

## 5. Files at the Initial Decode-Only Pivot

| File | Change |
|---|---|
| `scripts/export_gdn_state.py` | **new** — GPU prefill → `.gdnstate` export + self-check |
| `gdn_model.cpp` | `gdn_gemv` engine; decode-only `gdn_forward` (no GEMM/flags); conv + attention always restore/save; single `gdn_decode_step_host` |
| `gdn_model.h` | decode-only `gdn_forward` (20 args at this stage); flag macros + prefill/matmul decls removed |
| `gdn_eval.cpp` | `--decode-from-state` loader + decode loop; MC/LL/re-prefill modes removed |
| `host.cpp` | `upload_decode_state` into the resident BOs; `run_decode_hw` decodes from `.gdnstate`; `--decode-from-state`; 20-arg kernel call |
| `scripts/decode_correctness_check.sh` | gates on the decode-only-from-state path |
| `Makefile` | `host_tb` = `gdn_eval` only; retired the prefill testbenches |

## 6. Validation

Bit-exact to the GPU golden at every stage (fp32 → exact-match gate):

| Stage | Result |
|---|---|
| GPU export self-check (cache-decode vs golden, 8 tok) | exact |
| Native csim, state contract through old kernel (8 tok) | exact |
| Native csim, **decode-only GEMV** (8 / 32 / 64 tok) | **exact, first_div −1, 100% top-1** |
| **On-card, decode-only GEMV (64 tok)** | **exact, first_div −1, 100% top-1; flat 1949.6–1949.8 ms** |

The standing PostToolUse hook (`decode_correctness_check.sh --fast`) now exercises
the decode-only path and stays green.

### On-card result (U55C, 1×64, device 0)

| metric | value |
|---|---|
| correctness | 64/64 bit-exact vs GPU golden |
| per-token shape | **flat 1949.6–1949.8 ms (0.2 ms spread)** — O(1) |
| kernel TPOT | **1.95 s/token** |
| vs Step-1 (GEMM at num_rows=1, 2.56 s) | **1.31× faster** |
| effective weight bandwidth | 2.87 GB/s = **45%** of the 6.4 GB/s single-port peak |
| build | 4 h 21 m; kernel timing +4.26 ns, `hbm_aclk` +0.065 ns (0 failing) |

Removing the GEMM (16×16 grid + 5.76 MB URAM) let the kernel close timing with
margin and **no `hbm_aclk` marginality** (Step-1 had −50 ps/21 paths).

In the coupled version, `gdn_gemv` read weights and MAC'd in the *same* loop
nest, one output row at a time — each output restarted the `gemv_k` pipeline
(Depth=54) and **broke the AXI burst**, chopping the 5.6 GB stream into `out_dim`
latency-bound reads (2.87 GB/s = 45% of one port).

## 6b. Stage 1 — decoupled reader → MAC (on-card, bit-exact)

`gdn_gemv` is now an HLS **dataflow** of two processes: `gemv_read` issues one
monotonic sweep over the whole projection's weights (a single long 512-bit
burst) into an `hls::stream`; `gemv_compute` drains the FIFO and MACs against the
resident activation. The burst no longer breaks between output rows.

| metric | coupled | **decoupled** |
|---|---:|---:|
| kernel TPOT (flat) | 1949 ms | **1543 ms** (1.26×) |
| weight-port rate (XRT) | 2.87 GB/s (45%) | **5.30 GB/s (83% of one 512-bit port)** |
| avg read burst | — | **4.1 KB (full 64-beat)**, 97 ns latency |
| vs Step-1 (2.56 s) | 1.31× | **1.66×** |
| build / timing | — | 3 h 41 m; +3.77 ns, `hbm_aclk` 0 failing |

**1.54 s is the single-port floor**, and the decouple reached it. XRT shows the
port reads 5.6 GB/token at 5.30 GB/s ⇒ **~1.06 s of weight streaming**; the
kernel is 1.54 s ⇒ **~0.48 s is non-gemv serial work** (recurrent-attention state
R/W, the a/b `gdn_gemv_tiny`, conv) during which the weight port idles. So
`1.06 (gemv @ 83% of one port) + 0.48 (non-gemv) = 1.54 s`. The ~1.0 s target had
assumed a full 6.4 GB/s port *and* zero non-gemv — neither reachable on one
master. The decouple's *purpose* (burst efficiency) was met and exceeded:
**port 2.87 → 5.30 GB/s, 45% → 83%.** Further single-port tuning
(`num_read_outstanding`) has <0.1 s of headroom (latency already hidden under
4 KB bursts) — the lever that breaks the floor is multi-port.

## 6c. Stage 2 — multi-channel weight readers, N=2 (on-card, bit-exact)

The weight stream is split across **two parallel 512-bit HBM readers**. The
projection weights are **compact-sharded**: each projection's output rows are cut
into N stripes, shard *c* holds stripe *c* of every projection (one copy split N
ways — no replication). The two shards sit on **disjoint HBM channels**, so the
two readers' bursts hit different channels concurrently. `gdn_gemv` is now a
dataflow of **one PE per channel**: `gemv_pe_bcast` reads the resident activation
once and fans it to the PEs; each `gemv_pe_mac` MACs its shard's stripe; a single
MAC-free `gemv_collect` writes the output. The bit-exactness is preserved (same
resident `a_loc`, same beat order, same partial-bank adder-tree reduction).

| metric | singleport (Stage 1) | **N=2 (Stage 2)** |
|---|---:|---:|
| kernel TPOT (flat) | 1543 ms | **875 ms** (1.76×) |
| TPOT incl. host | 1540 ms | **937 ms** (1.64×) |
| weight readers | 1 × 512-bit | **2 × 512-bit, disjoint HBM** |
| decode bit-exact (1×64) | ✓ | **✓ top1 100%, first_div −1** |
| kernel timing (WNS) | +3.77 ns | **+0.506 ns, 0 failing** |
| shell `hbm_aclk` | 0 failing | −0.044 ns, 23 ep (**benign** — bit-exact) |
| build | — | xo (PE) 3 h 50 m + relink 3 h 24 m |

**Two routing/allocation walls, both load-bearing for N≥2:**

1. **PE distribution (routing).** A *monolithic* 2-wide consumer (one module doing
   both channels' 16-wide fp32 reductions) failed `route_design` **twice** —
   partially-conflicted `full_dsp` fp-adder nets, congestion level 7, after ~7 h
   routes. Splitting into **one independent dataflow PE per channel** (each with
   the Stage-1 single-port density that routed) lets Vivado place them apart;
   `SSI_SpreadLogic_high` + a LUT-fabric `bind_op` on the lane adds did the rest.
   Closed at WNS **+0.506 ns**, 0 failing kernel endpoints.

2. **Disjoint HBM bank map (allocation).** The first disjoint-sharding map put the
   shards on **narrow HBM groups that overlap** weight_data's wide group
   (`mm:HBM[10:20]` ⊂ `weight_data:HBM[10:31]`). XRT 2022.1 allocates each `xrt::bo`
   bottom-up/contiguous, so weight_data (5.46 GiB → 11 banks) takes banks 10–20
   and the shard's HBM[10:20] group is left with **zero free banks → `std::bad_alloc`**.
   `probe_alloc.cpp` pinned it (weight_data OK, shard0 FAIL as the 2nd bo). Fix:
   **dedicated, non-overlapping bank ranges** — `weight_data:HBM[20:31]` (12 banks),
   `weight_data_mm:HBM[10:14]`, `weight_data_mm2:HBM[15:19]` (5 each). 12+5+5 = 22 =
   HBM[10:31] exactly; activations stay on HBM[0:9]. Link-only change (the PE `.xo`
   is unchanged), so the second build was a relink.

**Why 1.76× and not 2×.** Amdahl on the kernel: `1543 = W + S`, `875 = W/2 + S`
⇒ **W ≈ 1336 ms** (parallel weight streaming) and **S ≈ 207 ms** serial floor —
recurrent-attention state R/W (~50 MB/token), the a/b `gdn_gemv_tiny`, conv,
and the per-layer scalar `weight_data` reads (norms / conv / a-b proj). Logits are
computed **host-side** (`host.cpp:compute_logits` reads `lm_head` from host RAM),
so `lm_head` is *not* a kernel cost. The readers halved W exactly; S is untouched.
Adding readers (W/N + S) is the next lever — realised at N=4 in §6d.

## 6d. Stage 2b — N=4 readers (on-card, bit-exact)

Four parallel readers: `GEMV_CHANNELS=4`, four compact shards on four disjoint HBM
bank groups, two more m_axi masters (`mem_weights_mm3/mm4`). The shard builder,
run-state, and `gdn_gemv` were generalised to an N-element shard array, so the
only N-specific surface is the kernel arg count, the host BO count, and the hw.cfg
bank map. Bit-exactness holds (each output row is still one PE's identical reduction).

| metric | N=2 | **N=4 (Stage 2b)** |
|---|---:|---:|
| kernel TPOT (flat) | 875 ms | **600 ms** (1.46× over N=2; **2.57× over Stage 1**) |
| TPOT incl. host | 937 ms | **662 ms** |
| weight readers | 2 | **4 × 512-bit, disjoint HBM** |
| decode bit-exact (1×64) | ✓ | **✓ top1 100%, first_div −1** |
| kernel timing (WNS) | +0.506 ns | **+0.104 ns, 0 failing** |
| shell `hbm_aclk` | −0.044 ns, 23 ep | **+0.081 ns, 0 failing** (clean) |
| build | xo + relink | **relink-only 3 h 22 m** (xo reused) |

**Two non-obvious build issues, both fixed:**

1. **HBM bank budget.** weight_data (11 banks) + 4 shards (3 banks each = 12) = 23
   weight banks > the 22 of HBM[10:31]. Decode reads activations once per token,
   so the 3 gmem masters collapse to **1 bank each** (HBM[1:3]), freeing HBM[4:9].
   Final disjoint map: shards HBM[4:6]/[7:9]/[10:12]/[13:15], weight_data
   HBM[16:26], activations HBM[1:3], control HBM[0] — 27 of 32 banks, no overlap.

2. **Profiling IP crashed design-init.** The first N=4 link aborted in Vivado
   design-init — `HADAFileSet::getSrcOptions() : NULL pointer` — inside the DPA
   trace s2mm IP that `--profile.data all:all:all` inserts (trace buffer on
   HBM[0]). With 8+ HBM masters spanning SLR0/1/2 it could not initialise.
   Dropping `--profile.data` removed that trace master: the relink not only built
   but **improved `hbm_aclk`** (−0.044 → +0.081 ns) by taking the writer off
   HBM[0]. TPOT (host timing) + bit-exactness need no on-card trace.

**Scaling is sub-linear past 2 readers.** Ideal W/N + S predicts N=4 ≈ 541 ms;
measured 600 ms (~60 ms over). Four readers contend more on the shared HBM
crossbar and the per-shard bursts are smaller, so effective per-reader bandwidth
drops. The 2.57× cumulative is real, but each doubling now returns less.

## 6e. Stage 2c — N=8 readers (on-card, bit-exact)

Eight parallel readers (`GEMV_CHANNELS=8`): eight 2-bank shards on disjoint HBM
groups (HBM[4:19]) + weight_data (HBM[20:30]) + activations on 1 bank each — 31 of
32 banks, **12 HBM masters**. Only the explicit kernel-arg / host-BO / hw.cfg
surface changed; the generalised PE body scaled untouched. Bit-exact (top1 100%,
first_div −1).

**The full bandwidth ladder (kernel TPOT, flat / O(1), all bit-exact):**

| readers | kernel ms | step | cumulative | hbm_aclk |
|---:|---:|---:|---:|---:|
| 1 (singleport) | 1543 | — | 1× | +3.77 ns |
| 2 | 875 | 1.76× | 1.76× | −0.044 ns, benign |
| 4 | 600 | 1.46× | 2.57× | +0.081 ns |
| **8** | **462** | **1.30×** | **3.34×** | −0.008 ns, benign |

kernel-clock WNS stayed clean throughout (+0.342 ns at N=8); build 4 h 29 m.

**The lever is spent.** Each doubling returns less (1.76 → 1.46 → 1.30×) — the
serial floor S grows as a share of TPOT and the HBM crossbar contends more with
every added master. Re-fitting S on the N=4/N=8 points gives S ≈ 325 ms (vs the
N=1/N=2 fit's 207 ms — the gap is the sub-linear W). N=16 would need ~20 masters
(likely unroutable) for ~1.2×. **N=8 (3.34×) is the practical end of the
weight-bandwidth roadmap.**

## 6f. On-chip lm_head + argmax — a complete decode step (on-card, bit-exact)

The kernel now emits the next token id directly (token in → token out) instead of
returning the hidden vector for the host to project and argmax. `lm_head` is a 9th
**sharded gemv** (out_dim = vocab = 32000, reusing the 8-reader engine; its [V,H]
rows split into 8 stripes appended after every layer's projections in each shard).
A greedy **on-chip argmax** (`gdn_argmax`, first-max tie-break matching the host)
reduces the 32000 logits and writes the token id into `x_norm[0]` — x_norm is spent
as an activation by then, so it doubles as the 1-int output (no extra port). The
logits scratch shares the `gmem_mlp` master (no 13th HBM master).

| metric | N=8 (host logits) | **N=8 + on-chip lm_head/argmax** |
|---|---:|---:|
| kernel ms/tok | 462 | **470** (+8: lm_head gemv + argmax) |
| TPOT incl. host | 525 | **470** (host `compute_logits` removed: −55 ms) |
| kernel WNS / hbm_aclk | +0.342 / −0.008 | +0.169 / **+0.001** (clean) |
| decode bit-exact (1×64) | ✓ | **✓ top1 100%, first_div −1** |

Two results: (1) the complete-decode TPOT *drops* 525 → 470 ms — the single-threaded
host lm_head matmul (~65 M MAC/token) cost more than the ~8 ms on-chip gemv; (2) the
number is now **fair** — the kernel does forward + lm_head + argmax, exactly what the
GPU TPOT measures.

**Fair GPU comparison (both fp32, both forward + lm_head + argmax).** A100 GDN decode
is launch-overhead-bound at 1.3B / batch-1 — flat ~33.5 ms fp32 ≈ bf16 (doubling the
weight bytes costs ~0, proving it is *not* bandwidth-bound). FPGA 470 ms ⇒ **~14×**
slower, dominated by the ~325 ms serial floor (recurrent-attention state I/O +
compute), which the GPU pays ~nothing for. fp32↔fp32 normalisation does **not** close
the gap (the GPU is idle, not byte-limited). The FPGA's defensible angle is perf/Watt
(~50–75 W vs ~250–300 W), which needs the latency gap down to <~4–5×, not <1×.

## 6g. Clock re-synth — 100 → 150 MHz (on-card, bit-exact)

The kernel re-pipelines cleanly to ~200 MHz-class paths (csynth est. 3.837 ns at a 5 ns
target), but two things had to line up, each caught at a cheap gate before an expensive one.

**(1) Over-synthesize for routing margin.** The congested route adds ~2.7 ns of overhead,
so building *and* linking at the same higher freq misses (the uncertainty reserve shrinks
faster than the overhead). Synthesize the xo at a tighter target than you run:
```
make xo FREQ=200 && make xclbin FREQ=150
```
The 3.837 ns logic linked at 150 MHz (6.67 ns budget) leaves ~2.83 ns for routing — just over
the ~2.7 ns overhead — so it closes (WNS +0.003 ns, all constraints met). A naive same-freq
build (150-target xo @ 150) would miss; v++ accepts the synth/link freq split.

**(2) A faster clock breaks the gemv MAC's II=1 — the fix is a loop restructure, not a pragma.**
fp-add latency is ~constant in ns, so its latency *in cycles* = latency_ns ÷ clock_period: a
~1-cycle fadd at 100 MHz becomes ~3 cycles at 150 MHz, and any *carried* accumulator dependency
inflates II directly. `gemv_pe_k`'s rotating accumulator `part[kp & 7] += lane` reuses each of 8
banks every 8 iters (so the fadd really has 8 cycles), but the **runtime** index makes HLS treat
it as distance-1 → II inflates → the dominant gemv slows more than the clock gains (a naive
150 MHz build measured **309 ms**, *worse* than 212.5). A `#pragma HLS dependence` hint did not
convince HLS. Fix (commit `8b70225`): **unroll the k-loop by `GEMV_PARTIAL`** so each `part[p]`
has a compile-time-fixed index → 8 independent fadd chains; the 8 stream reads/iter floor the
outer loop at II=8 = 1 pack/cycle (same cycle count as the old II=1 over `k_packs`) but robust
to the multi-cycle fadd. **Bit-exact**: `part[p]` still sums exactly the `kp ≡ p (mod 8)` lanes
in order. The same break hits the smaller fp-reductions (`rmsnorm_sq`, `onorm_sq`, `gvt_k`,
II=2–3) — left as-is (≈2.3 ms total, see below).

| metric | Build A @ 100 MHz | **restructured @ 150 MHz** |
|---|---:|---:|
| TPOT ms/tok (on-card, 1×64) | 212.5 | **165.6** |
| speedup | — | **1.28× (2.84× over the 470 original)** |
| decode bit-exact | ✓ | **✓ 64/64** |
| routed WNS / timing | +0.003 | **+0.003, all met (clean)** |

**Per-op breakdown @ 150 MHz (per-op profiler) — the gemv is sliding toward HBM-bound.**

| op | ms | % | scaled |
|---|---:|---:|---|
| gemv | 119.0 | 71.5% | **~1.19×** (not 1.5) |
| recurrent | 26.9 | 16.2% | ~1.45× |
| conv | 10.9 | 6.5% | ~1.5× |
| gemv_tiny | 6.8 | 4.1% | ~ok |
| rmsnorm / onorm / swiglu | 2.3 | 1.3% | the II loops — negligible |

The compute/latency-bound ops (recurrent, conv) scaled with the clock; the **gemv (72%) only
sped up 1.19×**. 5.6 GB ÷ 119 ms = ~47 GB/s across 8 ports = ~61% of the 76.8 GB/s the kernel
now requests (8×512-bit×150 MHz), down from ~76% at 100 MHz: the faster clock asks HBM for packs
faster than the read pipeline (burst gaps + 8-channel crossbar contention + HBM latency) delivers.
So more clock now buys diminishing gemv returns — the dominant op's ceiling is HBM **efficiency/
bandwidth**, addressed only by more channels (§6e, exhausted) or less traffic (INT8), not Fmax.
**[Superseded by §6h: this HBM-bound read was *wrong*. The 61% was the gemv MAC's per-output-row
pipeline fill, not HBM. Flattening the loop recovered it for a further 1.36×.]**

## 6h. gemv loop-flatten — the "HBM ceiling" was the per-row pipeline fill (on-card, bit-exact)

§6g blamed the gemv's ~61% port utilisation on HBM read efficiency. **That diagnosis was wrong.**
A per-projection breakdown (one CU launch per projection) showed utilisation tracking `k_packs`,
not bandwidth:

| projection | k_packs | measured BW/port | port util |
|---|---:|---:|---:|
| q/k/v/o, mlp_gate, mlp_up, lm_head | 128 | ~5.6 GB/s | ~59% |
| **mlp_down** | **352** | **~7.5 GB/s** | **~78%** |

If HBM were the wall, the high-`k_packs` projection could not pull *more* per port. The real
cause: `gemv_pe_o` (the output-row loop) was **sequential** (csynth `Pipelined: no`), running the
inner `gemv_pe_k` read/MAC pipeline start→drain **per row**. Each row paid a fixed ~92-cycle tax —
the ~81-stage fadd-reduction-tree fill (Depth) + ~19-cycle reduce/emit — on top of `k_packs`
cycles of useful weight reads. So util = `k_packs / (k_packs + 92)`: 128 → 58%, 352 → 79%,
matching the measured BW to <1%. The weight port idled during the fill; HBM was never the wall.

**Fix (commit `abe46f4`): flatten the loop nest** into one continuous II=8 `gemv_pe_flat` pipeline
over the whole `stripe × k_packs` shard, so the read/MAC pipeline never goes cold between rows —
fill paid **once per call**, not per row. (You cannot just `#pragma HLS pipeline` the outer loop:
that fully unrolls the inner k-loop → resource blowup. The restructure is real.)
- **Ping-pong partials** `part[2][8]` + a **one-row-deferred emit**: a finished row is reduced/
  written only after the next row has fully elapsed, so its accumulators are retired *and* live in
  the other buffer than the row currently accumulating → no row-boundary RAW stall, II stays 8.
  Runtime `cur` selects between two register banks only; the per-`p` index stays compile-time (the
  8 independent fadd chains from §6g are unchanged).
- **Bit-exact**: each `part[p]` still sums the `kp ≡ p (mod 8)` lanes in order, same
  `((p0+p1)+(p2+p3))+((p4+p5)+(p6+p7))` tree. Per-row tax 92 → ~0 cyc; port util ~59% → ~99%.

| metric | restructured @150 (§6g) | **flatten @150** |
|---|---:|---:|
| TPOT ms/tok (on-card, 1×64) | 165.6 | **121.4** |
| speedup over §6g | — | **1.36× (−27%)** |
| gemv total | ~109 ms | **~68 ms** |
| decode bit-exact | ✓ 64/64 | **✓ 64/64** |

csynth: II=8 holds (depth 98→109, **paid once**); +6.6K LUT total. The denser netlist needed
**FREQ=250** over-synth (vs 200) to close at 150 MHz — routed **WNS +0.025 ns, 0 failing
endpoints**. The failing path at xo@200 was a high-fanout loop-control reg → fadd clock-enables
(route-dominated, −0.298 ns); the tighter 4 ns synthesis re-placed/replicated it closed. (The
xo@200 build ran bit-exact at **129 ms** despite −0.298 — the violation was benign — but a
negative-WNS bitstream is not committed.)

## 7. Current Integrated Result and Next Step

Decode-only is bit-exact and flat at **121.4 ms/token** *complete* decode steps
(forward + lm_head + argmax all on-chip), **~3.9× over the 470 ms baseline** (§6f).
After the loop-flatten (§6h) the gemv is ~68 ms (~56%); the remainder is the **serial
floor** — recurrent-attention state I/O on one HBM[0] master + the gated-delta-rule
update (~27 ms) + conv/tiny — and the FPGA is now ~3.6× the A100's launch-bound 33.5 ms. Levers:

The standalone 32-port mono-kernel has since routed and sustained 263.063 GB/s
at an achieved 130.6 MHz. That result proves the weight path, not full-model
integration: `gdn_forward` still uses the eight-reader engine described above.
Integration must preserve the clustered physical hierarchy without crowding
the recurrent, convolution, activation, and state datapaths.

- **Raise the kernel clock** — DONE to 150 MHz (§6g): 1.28× / 165.6 ms, bit-exact, via
  over-synth (xo@200, link@150) + the gemv-MAC unroll restructure.
- **Flatten the gemv loop** — DONE (§6h, `abe46f4`): a further **1.36× / 121.4 ms**, bit-exact.
  §6g's "gemv is HBM-bound" was a misread — the 61% was the per-output-row pipeline fill, killed
  by folding the row loop into one continuous II=8 pipeline. Needed **FREQ=250** over-synth to
  close the denser netlist (WNS +0.025). Fixing the small fp-reduction II loops (rmsnorm/onorm/
  gvt_k, ≈2.3 ms) is the same unroll trick but near-negligible; rmsnorm/onorm reorder the sum
  (token-exactness risk).
- **Attack S** (bit-exact): spread the 50 MB/tok state across channels, parallelise
  the gated-delta-rule update, overlap state R/W with the weight stream.
- **INT8 weights** (deferred all-fp32 plan): 4× less W + relieves the crossbar; the
  biggest single lever, at the cost of bit-exactness (tolerance gate).

## 8. References

The `gdn_gemv` engine (§3) follows the established design principles for
bandwidth-bound LLM decode on HBM-equipped FPGAs — activations resident on-chip,
weights streamed once, MAC width matched to the memory port, and (for scaling)
the weight stream sharded across HBM channels.

1. **FlightLLM** — S. Zeng, J. Liu, G. Dai, X. Yang, T. Fu, H. Wang, W. Ma,
   H. Sun, S. Li, Z. Huang, Y. Dai, J. Li, Z. Wang, R. Zhang, K. Wen, X. Ning,
   and Y. Wang. "FlightLLM: Efficient Large Language Model Inference with a
   Complete Mapping Flow on FPGAs." *ACM/SIGDA Int'l Symp. on Field-Programmable
   Gate Arrays (FPGA)*, 2024, pp. 223–234. arXiv:2401.03868. doi:10.1145/3626202.3637562.
   — *Source of the "always-on-chip decode" scheme (activations resident across
   the whole decode pass); LLaMA2-7B on a single Alveo U280, 65.9% HBM bandwidth
   utilization.*

2. **DFX** — S. Hong, S. Moon, J. Kim, S. Lee, M. Kim, D. Lee, and J. Kim.
   "DFX: A Low-latency Multi-FPGA Appliance for Accelerating Transformer-based
   Text Generation." *55th IEEE/ACM Int'l Symp. on Microarchitecture (MICRO)*,
   2022. arXiv:2209.10797. doi:10.1109/MICRO56248.2022.00051.
   — *Single-token GEMV-centric decode core with an HBM-bandwidth-oriented tiling
   and dataflow; GPT-2 on 4× Alveo U280.*

3. **Embedded-FPGA decode (bandwidth/capacity limit)** — J. Li, T. Li, G. Shen,
   D. Zhao, Q. Zhang, and Y. Zeng. "Pushing up to the Limit of Memory Bandwidth
   and Capacity Utilization for Efficient LLM Decoding on Embedded FPGA." 2025.
   arXiv:2502.10659.
   — *The closest analogue to `gdn_gemv`'s datapath: hidden state kept on-chip,
   N multipliers + an adder tree sized to the 512-bit memory beat, and a weight
   data-arrangement that enables consecutive bursts — reaching 85% of the
   theoretical decode-bandwidth limit.*

4. **Serpens** — L. Song, Y. Chi, L. Guo, and J. Cong. "Serpens: A High Bandwidth
   Memory Based Accelerator for General-Purpose Sparse Matrix-Vector
   Multiplication." *59th ACM/IEEE Design Automation Conference (DAC)*, 2022,
   pp. 211–216. arXiv:2111.12555. doi:10.1145/3489517.3530420.
   — *Reference for the multi-channel HBM streaming pattern that the Step-2
   widening of `gdn_gemv` adopts (sharding the weight stream across HBM pseudo-
   channels with on-chip partial accumulation).*
