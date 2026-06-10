# Decode Correctness + TPOT Benchmark (FPGA)

Validates that the FPGA implementation **decodes the right tokens** — and reports
per-token latency (TPOT) — on the **current bitstream, no RTL changes**. It is the
correctness counterpart to the GPU latency study in
[decode_premise.md](decode_premise.md) and the gate for the future single-token
decode datapath in [decode_roadmap.md](decode_roadmap.md).

**Acceptance (met):** the on-card U55C free-running greedy trajectory is
**bit-exact to the GPU fp32 golden over 64 generated tokens** (0 divergences).

## How it works

Decode here is **re-prefill greedy**: each step runs a full `gdn_forward` over the
growing prefix and takes the argmax of the last token (the current kernel clears
recurrent state every call, so re-prefill is required and correct). The Python
golden is generated the **same way** (fp32, full-forward + argmax + append — not
`model.generate`), so the two share prefill numerics and can match exactly.

Two correctness views, greedy only (deterministic → exact oracle):
- **Teacher-forced top-1 agreement** — feed the golden prefix, compare the FPGA's
  per-position argmax to the golden's. Rigorous, no compounding.
- **Free-running trajectory match** — generate autoregressively, compare the whole
  trajectory to the golden (the bit-exact acceptance).

## Pipeline / files

| Stage | File | Role |
|-------|------|------|
| Golden + fixtures | `scripts/export_gdn_c.py` (`decode` task), `scripts/compare_gdn_c.py` (`--decode-golden`) | Prompt set → `c_impl/fixtures_decode/decode.gdnreq` (LL kind: `ctx`=prompt, `cont`=golden traj); fp32 GPU golden → `c_impl/results_decode_golden/decode.decode.json` |
| Native harness | `c_impl/gdn_eval.cpp` (`--decode`) | Same synthesizable `gdn_model.cpp` on CPU → `results_decode_c/decode.c.json` (gen_traj, tf_argmax, per-step TPOT) |
| On-card harness | `c_impl/host.cpp` (`--decode`) | U55C bitstream via XRT → `results_decode_hw/decode.hw.json` (+ `kernel_ms`) |
| Checker | `scripts/check_gdn_c_parity.py` (`--decode`) | exact_traj_match, first_divergence_index, top1_agreement_rate, tpot_summary |
| Driver | `scripts/decode_correctness_check.sh` | `--fast` (hook smoke) and full modes; gates on exact match (exit ≠ 0 on mismatch) |

Fixture/golden/JSON schemas are documented at the top of each script.

## Results

GPU golden: `m-a-p/1.3B-100B-GatedDeltaNet-pure`, fp32, re-prefill greedy.
Prompt 0 (28 tokens) golden continuation: *"…the empire. From its humble
beginnings as a small settlement on the Tiber…"*

| Target | Tokens | exact_traj_match | top-1 | per-token (median) |
|--------|-------:|:----------------:|:-----:|-------------------:|
| Native C (CPU, FPGA logic) | 64 | ✅ MATCH (0 div) | 100% | ~32 s (CPU) |
| **On-card U55C** (dev 0) | 8 | ✅ MATCH | 100% | 4.27 s kernel |
| **On-card U55C** (dev 1) | **64** | ✅ **MATCH (0 div)** | 100% | **6.95 s kernel** |

The FPGA decode is numerically identical to the reference model: no argmax tie
ever flips across 64 autoregressive steps.

## TPOT — read this carefully

These per-token times are the **O(n²) re-prefill baseline of the *current*
kernel**, not a decode TPOT: every step re-forwards the whole prefix, so per-step
latency *grows* with position (≈4.3 s/token at a 30-token prefix → ≈7.0 s/token
averaged to 92 tokens). This is **not** the flat O(1) decode the architecture
allows. For reference:
- GPU GDN decode (proper recurrent step): **flat ~34 ms/token** ([decode_premise.md](decode_premise.md)).
- FPGA single-token decode datapath (Phase 1, not yet built): ~1 s/token today
  (one weight read, no re-prefill) → ms/token with the bandwidth/INT8 levers in
  [decode_roadmap.md](decode_roadmap.md).

So this benchmark establishes **correctness now** and a **latency baseline** that
Phase 1 will improve by orders of magnitude.

## Run it

```bash
# 1. fixtures + fp32 GPU golden (once)
.micromamba/envs/gdn-hf/bin/python scripts/export_gdn_c.py decode --output-dir c_impl/fixtures_decode
.micromamba/envs/gdn-hf/bin/python scripts/compare_gdn_c.py --decode-golden c_impl/fixtures_decode/decode.gdnreq \
    --output c_impl/results_decode_golden/decode.decode.json --device cuda --dtype float32

# 2. native correctness + TPOT (full driver), or --fast for a smoke check
bash scripts/decode_correctness_check.sh                 # PASS/FAIL banner, gates on exact match
bash scripts/decode_correctness_check.sh --fast          # ~1-2 min, used by the inference-edit hook

# 3. on-card spot-check (U55C)
cd c_impl && ./host.exe build.hw/gdn_forward.xclbin artifacts/gdn-1.3b-f32.gdnw \
    fixtures_decode/decode.gdnreq results_decode_hw/decode.hw.json 0 --decode --limit 1 --decode-len 64
python ../scripts/check_gdn_c_parity.py --decode \
    --golden results_decode_golden/decode.decode.json --c results_decode_hw/decode_full.hw.json
```

## Standing workflow hook

`scripts/decode_correctness_check.sh --fast` is intended to run automatically
whenever inference code changes (`c_impl/gdn_model.{cpp,h}`, `c_impl/host.cpp`,
`c_impl/gdn_eval.cpp`, `lit_gpt/gated_delta_net.py`) via a PostToolUse hook in
`.claude/settings.json`, so any edit that breaks decode correctness fails fast
against the cached golden. (The fast check rebuilds `gdn_eval`, runs 1 short
decode, and gates on exact trajectory match — no GPU needed.)
