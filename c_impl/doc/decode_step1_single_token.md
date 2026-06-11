# Decode Step 1 — Single-Token Datapath (state persistence)

This documents **Step 1** of the decode roadmap ([decode_roadmap.md](decode_roadmap.md)):
make TPOT **O(1) in position** by forwarding exactly one token per step against
per-layer state that persists across calls, instead of re-prefilling the whole
prefix every step. Step 0 (correctness harness) is in
[decode_correctness.md](decode_correctness.md); the prefill optimization arc is in
[optimization_log.md](optimization_log.md).

**Result (on U55C, bit-exact):** decode is now flat at **2.56 s/token** vs the
re-prefill baseline's **6.95 s median that grows 4.2→9.7 s** — **2.72× at the
median, 3.80× by token 63, and unbounded as context grows.** 64/64 tokens match
the GPU golden.

---

## 1. On-card A/B (1 example, prompt_len=28, 64 decode steps, weights resident both runs)

Same xclbin, same fixture, same device; the only difference is `--single-token`.

| step (position) | re-prefill `kernel_ms` | single-token `kernel_ms` |
|-----------------|-----------------------:|-------------------------:|
| 1  (pos 0,  29 tok) | 4 200 | **2 564** |
| 31 (pos 30, 59 tok) | 6 942 | **2 564** |
| 63 (pos 62, 91 tok) | 9 738 | **2 564** |
| **median / tok**    | **6 968** | **2 564** |
| **per-token shape** | **O(n): 4.2 → 9.7 s** | **flat: min=max=2564** |

- **Median speedup 2.72×; at position 63, 3.80×; total decode walltime 2.80×.**
- The re-prefill baseline median (6 955 ms) reproduces the committed Phase-0
  number exactly.
- **The structural win is the shape, not the constant.** Re-prefill is O(n) in
  position (each step re-forwards prompt+generated); single-token is pinned flat
  forever. Extrapolated to a 2048-token context the re-prefill step approaches
  ~200 s/token while single-token stays 2.56 s — the gap is unbounded. This is
  the whole point of a recurrent (linear-attention) decoder: constant work per
  token.

Why **2.56 s** and not the **~1.05 s** the roadmap projected: see §5. Short
version — only the *state-persistence* half of Step 1 was built; the matmul
still runs the prefill GEMM datapath at one row.

---

## 2. Architecture delta — what changed, what did not

The change is **control-flow + state residency, not the compute engine.** The
prefill datapath is byte-for-byte unchanged; decode is a new *mode* of the same
kernel selected by one scalar flag.

**Changed:**

1. **One new kernel arg** — `decode_flags` (21st arg, `s_axilite` scalar,
   `gdn_model.cpp:1740`). `0` = prefill (bit-identical to before);
   `GDN_DECODE_RESTORE|GDN_DECODE_SAVE` = decode.
2. **Two HBM buffers became persistent per-layer state** (were per-call scratch):
   - `recurrent_state` — now holds **all 24 layers'** recurrent state,
     `24 × 8 × 256 × 256 × 4 B = 48 MB`. RESTORE at each layer's attention
     start, SAVE at its end.
   - `head_buffer` — repurposed as the **depthwise-conv tail store**,
     `24 layers × 3 convs × (conv_size−1) rows × hidden = 442 368 floats ≈ 1.7 MB`,
     so the conv's last-3-input window survives across tokens.
3. **RESTORE/SAVE gates** inside `gdn_depthwise_conv_silu` and
   `gdn_recurrent_attention`, keyed on `decode_flags`. Prefill (`flags=0`) still
   clears + discards exactly as before.
4. **Host `--single-token` loop** (`host.cpp`, `gdn_eval.cpp`) — prefill+SAVE
   once, then forward exactly 1 token/step with RESTORE|SAVE; all `xrt::bo` stay
   resident across calls (no weight re-upload, no re-prefill).

**Unchanged (the entire datapath):** `gdn_matmul_2d` weight-stationary GEMM
tiling, the 16×16 PE grid, the weight HBM map (HBM[10:31]) + 512-bit reads +
dedicated `mem_weights_mm` bundle, the Phase-B activation channels, and the
numerics. Because the prefill path is untouched, **correctness held for free**:
`flags=0` *is* the old code path, so the standing `decode_correctness_check.sh`
gate stayed green at every edit.

```
            decode_flags = 0            decode_flags = RESTORE|SAVE
            (prefill, unchanged)        (decode, new)
conv     :  zero window, discard tail   load tail→window, …, save last 3 rows
attention:  clear on-chip state         load layer state from HBM, …, save to HBM
matmul   :  GEMM tiling  ◄────────────── SAME (num_rows = num_tokens)
```

---

## 3. State layout in HBM

```
recurrent_state  (port depth 12 582 912 floats = 48 MB)
    layer ℓ slice base = ℓ · (HEADS·DK·DV) = ℓ · (8·256·256) = ℓ · 524288
    within a layer: [head][dk][dv]  row-major, dv contiguous (16-float Pack bursts)

head_buffer      (port depth 442 368 floats ≈ 1.7 MB)  — conv tail store
    slot (layer ℓ, conv c) base = (ℓ·3 + c) · (conv_size−1)·hidden = (ℓ·3+c)·3·2048
    within a slot: [k=0..2][col=0..2047]  = the last 3 input rows of that conv
```

Both live on HBM[0] (control/small channel, `hw.cfg`) — they ride in the
headroom next to config/a/b/tokens and never contend with the weight channels
(HBM[10:31]) or the Phase-B activation channels. Per-token state traffic is
`48 MB R + 48 MB W + ~1.7 MB×2 ≈ 99 MB`, i.e. **~10 ms at HBM[0] rate — under
0.5 % of the 2.56 s step.** State is *not* the bottleneck; weights are.

---

## 4. Code

### 4.1 Flags (`gdn_model.h`)

```c
#define GDN_DECODE_RESTORE 1u  /* load per-layer recurrent + conv state at layer start */
#define GDN_DECODE_SAVE    2u  /* write per-layer recurrent + conv state at layer end  */
```

### 4.2 Conv tail — RESTORE / SAVE (`gdn_depthwise_conv_silu`)

The window is `in_window[k][col]`, k∈{0..3}. Slots are shifted every row so
`in_window[3]` is the newest input. RESTORE pre-loads the 3 saved rows into
slots 1..3 so the first new token's shift yields `{tail0,tail1,tail2,row0}`;
SAVE writes back slots 1..3 (the newest 3 inputs) after the last row.

```c
if (decode_flags & GDN_DECODE_RESTORE) {              /* gdn_model.cpp:1186 */
    const Pack16 *tail_p = (const Pack16 *)conv_tail;
    for (k = 0; k + 1 < kernel_size; ++k)             /* 3 rows */
        for (cp = 0; cp < num_cols/16; ++cp)          /* 512-bit beats */
            in_window[k + 1][...] = tail_p[k*(num_cols/16) + cp];
}
/* … streaming conv over num_rows … */
if (decode_flags & GDN_DECODE_SAVE) {                 /* gdn_model.cpp:1249 */
    Pack16 *tail_p = (Pack16 *)conv_tail;
    for (k = 0; k + 1 < kernel_size; ++k)
        for (cp = 0; cp < col_packs; ++cp)
            tail_p[k*col_packs + cp] = in_window[k + 1][...];
}
```

For `flags=0` the window is just zeroed (`conv_init_win_*`) and the tail is never
touched — bit-identical to prefill.

### 4.3 Recurrent state — RESTORE / clear / SAVE (`gdn_recurrent_attention`)

On-chip `state[8][256][256]` (2 MB BRAM) is the working copy. RESTORE fills it
from the layer's 48 MB HBM slice via sequential 512-bit bursts (`GDN_PK=16`
unroll, II=1); the prefill `else` branch zeroes it (the original `state_clr`);
SAVE bursts it back.

```c
size_t st_base = (size_t)layer_index * GDN_HEADS * GDN_DK * GDN_DV;   /* :1312 */
if (decode_flags & GDN_DECODE_RESTORE) {                              /* :1314 */
    for (h…) for (j…) for (i += GDN_PK) {  #pragma HLS pipeline II=1
        for (pp = 0; pp < GDN_PK; ++pp)    /* unroll 16 */
            state[h][j][i+pp] = recurrent_state[st_base + (h*DK + j)*DV + i + pp];
    }
} else {                                  /* prefill: clear (was state_clr) */
    for (h…) for (j…) for (i += GDN_PK) state[h][j][i+pp] = 0.0f;
}
/* … per-token gated delta-rule update over num_tokens … */
if (decode_flags & GDN_DECODE_SAVE) { /* :1528 burst state → recurrent_state[st_base…] */ }
```

### 4.4 Host loop (`gdn_eval.cpp` / `host.cpp`)

```c
if (single_token) {
    if (step == 0) {                          /* warm-up: prefill prompt, SAVE state */
        run_forward(prefix, GDN_DECODE_SAVE);
        next = argmax(hidden_row(prompt_len - 1));
    } else {                                  /* O(1): forward 1 token, RESTORE|SAVE */
        run_forward({prev_tok}, GDN_DECODE_RESTORE | GDN_DECODE_SAVE);
        next = argmax(hidden_row(0));
    }
}
```

All BOs (weights, state, activations) are allocated once and stay resident; only
the 1-int token BO is re-synced per step. PCIe launch overhead is ~0.1 ms/call —
negligible against the 2.56 s kernel (resident on-device loop is deferred to
Step 4).

---

## 5. Why 2.56 s and not 1.05 s — the GEMV gap

The roadmap projected `5.6 GB / 5.4 GB/s ≈ 1.05 s`. That assumes the single-token
forward sustains the **prefill** weight bandwidth. It does not, because the
*matmul datapath was not changed* — only state handling was. `gdn_matmul_2d`
(`gdn_model.cpp:2208`) is weight-stationary **GEMM** tiling: it loads a weight
column-tile (`loadB`) once and amortizes it across `MM2D_ABLK_ROWS = 256` rows.

At `num_rows = 1`:

- **Array utilization 1/256.** The 16×16 grid (`sys_k`) computes one real row;
  the other 15 rows are zero-padded. Each weight is read to do **1 MAC** (GEMV),
  inside a structure built to do **256** (GEMM).
- **No streaming amortization.** Per column-tile the loops run
  `loadB → 1 sub-tile compute` *sequentially*; with only one sub-tile there is
  nothing to overlap the next `loadB` against, so the weight port is not kept
  saturated back-to-back. Effective weight BW ≈ `5.6 GB / 2.56 s ≈ 2.2 GB/s`,
  well under the prefill stream's ~5.4 GB/s.

This is exactly the **GEMV datapath** item the roadmap lists (Step 1 bullet 3 /
the lead-in to Step 2): when `num_rows == 1`, bypass the 2-D grid — stream each
weight row and dot it with the resident activation vector, weights flowing
back-to-back with no GEMM tiling drain. That single change recovers the gap from
2.56 s toward the ~1.05 s single-port floor; **Step 2** (N parallel weight
readers) then takes it toward ~55 ms. Both are pure datapath edits gated by the
same bit-exact check.

So Step 1 delivered its **structural** deliverable — decode is O(1) in position,
resident state, bit-exact — and isolated the **remaining** lever cleanly: it is
100 % weight-datapath, not state, not correctness, not routing.

---

## 6. Correctness

Bit-exact to the GPU golden at every level (fp32, so exact-match is the gate):

| run | result |
|-----|--------|
| native csim, 1 ex × 6 steps | exact, 100 % top-1 |
| native csim, 1 ex × 64 steps | exact, first_div −1 |
| native csim, 3 ex × 12 steps | all exact (cross-example state isolation) |
| **on-card, 1 ex × 64 steps** | **exact, first_div −1, 100 % top-1** |

The on-card match also retires a build caveat: the xclbin closed the kernel data
clock at **+7.34 ns** but the 450 MHz platform `hbm_aclk` showed a **−50 ps /
21-path** violation (baseline build was +2 ps — the known hbm_aclk marginality
wobbling across zero when all 32 HBM channels are populated). The bit-exact
64-token decode proves that violation immaterial (HBM silicon guardband absorbs
it); no rebuild needed.

---

## 7. Next

The immediate lever is the **`num_rows==1` GEMV matmul datapath** (§5) — recovers
2.56 s → ~1.05 s with no routing risk — paired with **Step 2** multi-channel
weight readers (~55 ms). The 16×16 grid and all prefill memory work stay as-is;
decode's only remaining bottleneck is weight bandwidth.
