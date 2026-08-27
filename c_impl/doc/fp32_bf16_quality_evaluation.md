# FP32 and BF16-Mixed Checkpoint Quality Evaluation

Status: complete as of 2026-08-27. The FP32 arm and both BF16 arms have run
Tables 2, 3, and 5 at their full reference sample counts, plus a BF16
recurrent-state follow-up, an all-BF16 arm (weights, activations and state
together), a native-BF16-product arm matching the prospective HLS multiplier,
and one on-hardware WikiText measurement. Everything is GPU-measured except the
section explicitly marked on-hardware.

Two caveats that apply throughout: RULER (Table 2) is not bit-reproducible and
carries a measured +/-0.6 run-to-run band, so differences at or below that are
not resolvable; and Table 5's as-run configuration understates every F1 task for
an answer-length reason documented in its section.

## Scope and interpretation

This evaluation asks two separate questions:

1. Does the current FP32 checkpoint reproduce the quality reported for the
   1.3B/100B-token Gated DeltaNet in Tables 2, 3, and 5 of the ICLR 2025 paper?
2. Does a persisted BF16-weight, BF16-activation checkpoint retain the FP32
   checkpoint's quality when the recurrent kernel continues to accumulate its
   state in FP32?

The current checkpoint is
`m-a-p/1.3B-100B-GatedDeltaNet-pure`, snapshot
`930ed6ae4ac629c86cb9855bb3dcb0a0974a29aa`. It is a third-party Hugging Face
checkpoint: the authors' NVlabs repository does not publish pretrained model
weights. The comparison is therefore a method-aligned reproduction, not a
byte-identical rerun of an official paper checkpoint.

The source snapshot contains 459 FP32 tensors and 1,466,344,000 tensor
elements. The installed FLA model consumes 1,466,343,808 parameters; 24
legacy `attn.D` tensors (192 scalars total) are present in the checkpoint but
are not consumed by the current model class. Both precision evaluations load
the same tensor names, architecture, tokenizer, prompts, datasets, and
decoding settings.

## Acceptance policy

These thresholds were fixed before inspecting the full Table 2 or Table 5
results.

For the FP32 checkpoint versus the paper:

- Table 2: macro-average no more than 3 percentage points below the paper;
  report any individual cell more than 10 points below.
- Table 3: eight-task accuracy average no more than 1 point below the paper,
  and each perplexity no more than 5% worse.
- Table 5: macro-average no more than 1 point below the paper; report every
  per-task delta.

For BF16-mixed versus this measured FP32 checkpoint:

- Table 2: macro-average drop no more than 2 points and no cell drop greater
  than 5 points.
- Table 3: accuracy-average drop no more than 0.5 point and perplexity increase
  no more than 1%.
- Table 5: macro-average drop no more than 0.5 point and no task drop greater
  than 2 points.

Passing the BF16 gates establishes conversion parity; it does not erase a
failure of the underlying FP32 checkpoint to reproduce the paper.

## Evaluation protocol

All inference runs use one NVIDIA A100 80 GB PCIe GPU with PyTorch 2.7.1
and CUDA 12.6, Transformers 4.51.3, datasets 3.6.0, FLA 0.4.2, and
lm-evaluation-harness 0.4.12.dev0. Repository HEAD is
`704da8f666c91458d4f7b733c26de64e206c634c`.

### Table 2: RULER S-NIAH

The local tasks wrap the exact generators and scorer from NVIDIA RULER commit
`c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a`: S1 uses a repeated-word
haystack and numeric value, S2 uses an essay haystack and numeric value, and
S3 uses an essay haystack and UUID value. Seed 42, greedy decoding, zero-shot
prompting, 128 maximum new tokens, and 500 samples per cell are used. S1 and
S2 cover 1K/2K/4K/8K; S3 covers 1K/2K/4K, matching the 11 paper cells.
The executable official RULER configuration is treated as authoritative for
S3 because it conflicts with a word-based description in the paper appendix.

### Table 3: short-context language modeling and commonsense

The lm-evaluation-harness tasks are WikiText-2, LAMBADA OpenAI, PIQA,
HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge, SocialIQA, and BoolQ, all
zero-shot. The paper's metric mapping is retained: raw accuracy for LAMBADA,
PIQA, WinoGrande, ARC-Easy, SocialIQA, and BoolQ; length-normalized accuracy
for HellaSwag and ARC-Challenge. The accuracy average is the arithmetic mean
over those eight accuracy values. WikiText and LAMBADA also report word
perplexity. This suite consumes token logits/log-likelihoods rather than only
the final argmax token.

### Table 5: LongBench v1

The 14 English LongBench-v1 tasks use dataset revision
`5e628be450b7e67fb7ae6e201bd6d8f7056f7672` and the prompt templates,
generation limits, middle-truncation rule, and scorers from THUDM LongBench
commit `2e00731f8d0bff23dc4325161044d0ed8af94c1e`. Input is capped at 4,096
tokens and all 3,350 test examples are evaluated with greedy decoding. Task
metrics are QA F1, Rouge-L, classification accuracy, or code edit similarity
as defined by LongBench. Table 2 and Table 5 require generated token sequences
but do not require exporting complete logits.

Batching was admitted only after output-equivalence checks. RULER batch 16
matched batch 1 byte-for-byte for the prompt, target, raw response, and scored
response. LongBench batch 2 and batch 16 matched sequential generation
byte-for-byte on the checked variable-length NarrativeQA examples.

## Results

All three tables are complete for FP32 and for both BF16 arms at the full
reference sample counts.

### The two BF16 arms

Two BF16 configurations were measured. They share **one** converted
checkpoint — the same files, with only the load dtype differing — so any
difference between them comes purely from the precision of the activations.

| Arm | Weights | Activations and math | Recurrent state | Purpose |
|---|---|---|---|---|
| **BF16 A** | BF16 | BF16 | FP32 | The pre-registered arm in this document's scope section. |
| **BF16 B** | BF16 | FP32 | FP32 | The configuration the FPGA would implement: BF16 in memory, FP32 arithmetic. |

Arm A is the stricter of the two and bounds Arm B from below, since it rounds
strictly more quantities. Arm B is the arm that decides the accelerator
question, because the decode kernel's HBM traffic is dominated by weights
(5.195 GB/token of the FP32 blob) while activations are BRAM-resident and
contribute almost nothing; only weight precision buys bandwidth.

The FP32 recurrent state is not a configuration choice. FLA's
`fused_recurrent` kernel allocates its accumulator as `tl.float32` and returns
an FP32 final state regardless of model dtype
(`fla/ops/gated_delta_rule/fused_recurrent.py:87`, `:183`), so both arms
already satisfy the FP32-state condition in this document's scope.

### Conversion method

`scripts/convert_gdn_checkpoint_bf16.py` applies `tensor.to(torch.bfloat16)`
to every floating-point tensor and nothing else numerically: no scaling, no
calibration, no per-channel treatment, no outlier handling. That is sufficient
because BF16 shares FP32's 8-bit exponent, so the cast cannot overflow or
flush values to zero; only significand precision falls from 24 bits to 8. The
script preserves non-float tensors, shapes, and metadata, copies the tokenizer
files, sets `config.json` `torch_dtype`, rewrites the shard index, and records
a SHA-256 of every input and output shard.

The cast was verified to be round-to-nearest rather than truncation: the
largest relative change measured across all 459 tensors is 3.891e-3, just
inside the round-to-nearest bound of 2^-8 = 3.906e-3, where truncation would
permit up to 2^-7. An independent re-run of the conversion reproduced both
shard SHA-256 values exactly.

### BF16 acceptance results

Both arms pass every pre-registered BF16 limit; Arm B passes by a factor of
two to four more margin than Arm A.

| Limit | Threshold | BF16 A | BF16 B |
|---|---|---:|---:|
| Table 2 macro drop | ≤ 2.0 | -0.31 | **-0.09** |
| Table 2 worst cell drop | ≤ 5.0 | -1.8 | **-1.2** |
| Table 3 accuracy-average drop | ≤ 0.5 | -0.06 | **-0.03** |
| Table 3 perplexity increase | ≤ 1% | +0.10% | **+0.00%** |
| Table 5 macro drop (as run) | ≤ 0.5 | -0.13 | **-0.04** |
| Table 5 worst task drop (as run) | ≤ 2.0 | -0.83 | **-0.20** |
| Table 5 macro drop (first-line) | ≤ 0.5 | -0.13 | **-0.07** |
| Table 5 worst task drop (first-line) | ≤ 2.0 | -0.83 | **-0.51** |

BF16 verdict: **pass** for both arms. Under Arm B, 8 of the 11 Table 2 cells,
the WikiText perplexity, and 4 of the 14 Table 5 tasks are unchanged from FP32
to the recorded precision, and the largest single movement across all 35 cells
is 1.2 points.

**No cell is genuinely improved by BF16.** Several cells show small positive
deltas, but a paired per-sample comparison of the Arm A and FP32 RULER sample
logs shows that only 37 of 5,500 answers changed at all: 10 wrong→right and 27
right→wrong, a net loss of 17 answers that accounts exactly for the -0.31 macro
delta. A 10-versus-27 split has probability 0.0038 under a symmetric-noise
null, so the direction of the change is real and negative. The positive cells
are single samples crossing a decision boundary — Table 2's "+0.2" is one
sample of 500, Table 3's HellaSwag "+0.02" is two questions of 10,042, and
ARC-Challenge's "+0.09" is one question of 1,172. Any delta below roughly 1/n
of a task should be read as one or two samples moving, not as a gain.

### Table 2: RULER S-NIAH results

All 11 paper cells were evaluated at the full 500 samples per cell: S1 and S2
contribute 2,000 samples each over 1K/2K/4K/8K, S3 contributes 1,500 over
1K/2K/4K. Values are percentages and deltas are percentage points.

| Cell | Paper | FP32 | FP32 - paper | BF16 A | BF16 B |
|---|---:|---:|---:|---:|---:|
| S1 1K | 98.4 | 100.0 | +1.6 | 100.0 | 100.0 |
| S1 2K | 88.4 | 100.0 | +11.6 | 100.0 | 100.0 |
| S1 4K | 91.4 | 100.0 | +8.6 | 100.0 | 100.0 |
| S1 8K | 91.8 | 97.4 | +5.6 | 97.6 | 97.4 |
| S2 1K | 100.0 | 100.0 | +0.0 | 100.0 | 100.0 |
| S2 2K | 99.8 | 100.0 | +0.2 | 100.0 | 100.0 |
| S2 4K | 92.2 | 85.6 | -6.6 | 85.4 | 85.6 |
| S2 8K | 29.6 | 40.0 | +10.4 | 39.2 | 40.0 |
| S3 1K | 86.6 | 82.0 | -4.6 | 80.2 | 81.8 |
| S3 2K | 84.2 | 81.6 | -2.6 | 81.4 | 80.4 |
| S3 4K | 27.6 | 53.2 | +25.6 | 52.6 | 53.6 |
| **Macro average** | **80.91** | **85.44** | **+4.53** | **85.13** | **85.35** |

FP32 Table 2 verdict: **pass**. The macro average is 4.53 points above the
paper, against a gate of no more than 3 points below. The worst single cell is
S2 4K at 6.6 points below, inside the 10-point individual-cell reporting
trigger, so no cell requires separate reporting.

The largest deltas are not attributable to scorer leniency. The upstream
`string_match_all` scorer accepts a case-insensitive substring anywhere in the
128 generated tokens, and the tasks run with an empty `until` list, so a
rambling continuation could in principle earn credit on a later guess.
Re-scoring every stored sample under first-line-only truncation reproduces all
11 cells exactly, so the model either retrieves the needle in its first line or
not at all.

Two measured properties of the harness bound how closely these cells can track
the paper. Tokenized with the model's own tokenizer, prompts reach 0.70 to 0.98
of their nominal length budget (S2 and S3 at 2K are the shortest, near 1,430 and
1,456 tokens), so a cell labelled 4K presents roughly 3,350 tokens. Separately,
the paper's own S1 row rises from 88.4 at 2K to 91.4 at 4K and 91.8 at 8K, which
single-needle retrieval accuracy should not do; the measured FP32 row is
monotone at 100.0/100.0/100.0/97.4.

Source results: `fp32/table2/s12/**/results_2026-08-17T17-41-25.826846.json`
(S1, S2) and `fp32/table2/s3/**/results_2026-08-17T18-01-16.184470.json` (S3),
both under `/home/yaoz0b/gdn_precision_eval_20260817/`.

### Table 3: short-context results

Accuracy values and their deltas use percentage points. For perplexity, lower
is better; the relative delta is included because it is the acceptance metric.

| Metric | Paper | FP32 | FP32 - paper | FP32 relative delta | BF16 A | BF16 B |
|---|---:|---:|---:|---:|---:|---:|
| WikiText PPL | 16.42 | 16.82 | +0.40 | +2.46% | 16.84 | 16.82 |
| LAMBADA PPL | 12.17 | 9.72 | -2.45 | -20.13% | 9.70 | 9.70 |
| LAMBADA accuracy | 46.65 | 51.81 | +5.16 | +11.07% | 51.48 | 51.80 |
| PIQA | 72.25 | 73.78 | +1.53 | +2.11% | 73.72 | 73.72 |
| HellaSwag | 55.76 | 60.13 | +4.37 | +7.83% | 60.15 | 60.14 |
| WinoGrande | 57.45 | 61.88 | +4.43 | +7.71% | 61.80 | 61.88 |
| ARC-Easy | 71.21 | 72.31 | +1.10 | +1.54% | 72.22 | 72.22 |
| ARC-Challenge | 38.39 | 40.78 | +2.39 | +6.24% | 40.87 | 40.70 |
| SocialIQA | 40.63 | 42.37 | +1.74 | +4.29% | 42.32 | 42.48 |
| BoolQ | 60.24 | 61.68 | +1.44 | +2.39% | 61.68 | 61.56 |
| **Accuracy average** | **55.32** | **58.09** | **+2.77** | **+5.01%** | **58.03** | **58.06** |

FP32 Table 3 verdict: **pass**. The accuracy average is 2.77 points above
the paper. WikiText perplexity is 2.46% worse, within the 5% gate, while
LAMBADA perplexity is 20.13% better.

**The size of the FP32-versus-paper gap is sensitive to the metric mapping and
should not be quoted as +2.77 without this caveat.** Two of the ten rows use
length-normalised accuracy, per this document's protocol section. Both metrics
were recorded, and they differ substantially:

| Task | Paper | As reported (`acc_norm`) | Plain `acc` |
|---|---:|---:|---:|
| HellaSwag | 55.76 | 60.13 | 46.30 |
| ARC-Challenge | 38.39 | 40.78 | 37.63 |

Scoring all eight accuracy rows with plain `acc` moves the accuracy average
from 58.09 to 55.97, and the gap over the paper from **+2.77 to +0.65**. The
normalised mapping is the more likely reading — 55.76 is a typical
length-normalised HellaSwag result at this scale and is the common reporting
convention — but ARC-Challenge argues the other way, since the paper's 38.39
is closer to the plain 37.63 than to the normalised 40.78. The paper's harness
version is also unknown, and lm-eval's multiple-choice task definitions have
changed across releases. The defensible statement is therefore that this
checkpoint is somewhat stronger than the paper's, by an amount between roughly
+0.7 and +2.8 points, not that it reproduces the paper.

None of this affects the BF16 conclusion, which compares this checkpoint
against itself through an identical harness, so the mapping cancels.

Table 3 is the most robust of the three tables and should be the primary
BF16 gate. Every task is scored by `multiple_choice`, `loglikelihood`, or
`loglikelihood_rolling` with `generation_kwargs: None`, so no text is
generated and no answer-truncation rule can affect the score. The recorded
run confirms `dtype: float32` (the `float16` default in
`run_gdn_table3_eval.sh` was overridden), `limit: None`, and every task at
its exact reference count. The model is above the paper on all ten rows; a
sign test over ten independent tasks gives p = 0.002, so the gap is
systematic rather than sampling noise, though only HellaSwag, LAMBADA
accuracy, and WinoGrande clear significance individually.

### Table 5: LongBench v1 results

All 14 English tasks completed over the full 3,350 examples, wrapper exit
code 0. Two FP32 columns are recorded because answer truncation is an open
protocol question, described below; the six affected tasks are marked `*`
and the other eight are identical between the columns.

| Task | Metric | Paper | FP32 as-run | Δ | FP32 first-line | Δ | BF16 A as-run | BF16 B as-run |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| narrativeqa `*` | QA F1 | 14.1 | 2.51 | -11.59 | 14.98 | +0.88 | 2.52 | 2.52 |
| qasper `*` | QA F1 | 14.0 | 7.12 | -6.88 | 15.72 | +1.72 | 6.71 | 7.09 |
| multifieldqa_en `*` | QA F1 | 23.3 | 13.58 | -9.72 | 25.56 | +2.26 | 13.45 | 13.58 |
| hotpotqa `*` | QA F1 | 13.7 | 7.05 | -6.65 | 15.99 | +2.29 | 6.88 | 6.98 |
| 2wikimqa `*` | QA F1 | 14.4 | 8.87 | -5.53 | 16.21 | +1.81 | 8.97 | 8.70 |
| musique `*` | QA F1 | 5.8 | 3.16 | -2.64 | 5.47 | -0.33 | 3.16 | 3.12 |
| gov_report | Rouge-L | 7.5 | 8.98 | +1.48 | 8.98 | +1.48 | 8.86 | 8.99 |
| qmsum | Rouge-L | 16.4 | 18.47 | +2.07 | 18.47 | +2.07 | 18.52 | 18.42 |
| multi_news | Rouge-L | 7.9 | 12.68 | +4.78 | 12.68 | +4.78 | 12.70 | 12.71 |
| trec | classification | 30.0 | 36.50 | +6.50 | 36.50 | +6.50 | 36.00 | 36.50 |
| triviaqa | QA F1 | 22.4 | 26.27 | +3.87 | 26.27 | +3.87 | 25.44 | 26.21 |
| samsum | Rouge-L | 23.0 | 30.15 | +7.15 | 30.15 | +7.15 | 30.04 | 29.95 |
| lcc | edit similarity | 18.7 | 18.46 | -0.24 | 18.46 | -0.24 | 18.50 | 18.45 |
| repobench-p | edit similarity | 22.1 | 18.06 | -4.04 | 18.06 | -4.04 | 18.31 | 18.12 |
| **Macro average** | | **16.66** | **15.13** | **-1.53** | **18.82** | **+2.16** | **15.01** | **15.09** |

FP32 Table 5 verdict as run: **fail**. The macro average is 1.53 points
below the paper against a gate of no more than 1 point below. Note that
`scripts/summarize_gdn_quality_eval.py` reports this gap as -1.47, because it
compares against the paper's own printed average of 16.6 rather than the 16.66
mean of the fourteen rounded per-task values used here. The verdict is the same
either way. Under
first-line answer truncation the same predictions give +2.16 and would
pass. The verdict is recorded as a failure because the as-run configuration
is the one that was executed; adopting truncation is a protocol decision
that has not been taken.

Only the as-run BF16 columns are tabulated above, to keep the table readable.
Under first-line truncation the BF16 macro averages are 18.69 (Arm A) and
18.75 (Arm B), against the FP32 18.82 — drops of 0.13 and 0.07, both inside
the 0.5-point BF16 limit, matching the acceptance table above.

The deficit is confined to the six QA tasks scored by F1, which are the six
tasks that apply no answer truncation. Their predictions are far longer
than their references — narrativeqa 83 words against 4, qasper 78 against
8, hotpotqa 20 against 2 — and F1 penalises precision, so a correct answer
embedded in a long continuation scores near zero. The summarisation,
classification, and code tasks are unaffected and 8 of them are above the
paper.

The decisive evidence that this is a protocol difference rather than a
model deficiency does not depend on the truncation experiment. At an
83-word median prediction against 4-word references, narrativeqa F1 cannot
exceed 9.2 even with perfect recall, while the paper reports 14.1. That
number is unreachable under this generation setting for any model, so the
paper's Table 5 must have produced shorter answers.

The configuration itself is faithful to THUDM LongBench, which newline-stops
only trec, triviaqa, samsum, lsht, lcc, and repobench-p. This harness
reproduces those six stops at scoring time — `first_line_only` for trec,
triviaqa, and samsum, and the `code_similarity` scorer's own first
non-comment line for lcc and repobench-p — and correctly leaves the QA tasks
untruncated. No change has been made to the configuration.

Under truncation, 11 of 14 tasks match or beat the paper and the only
material remaining deficit is repobench-p at 4.04 below.

### BF16 recurrent state (follow-up, Table 3 only)

*Measured 2026-08-20, after the three arms above.*

The arms above all keep the recurrent state in FP32 — that is not a
configuration choice but a property of FLA's kernel, which allocates its
accumulator and its returned final state as `torch.float32` regardless of model
dtype (`fla/ops/gated_delta_rule/fused_recurrent.py:87`, `:183`). Measured
directly: with `dtype=bfloat16` the cache holds **24 float32 tensors of shape
(1, 8, 256, 256)** — the 24 layers of recurrent state — while the *convolution*
state (72 tensors of shape (1, 2048, 4)) does become BF16. The document's
FP32-state claim therefore holds for the recurrent state; the conv-state
detail was previously unrecorded.

A separate question is whether the recurrent state could itself be BF16, which
would halve it from 48 MiB to 24 MiB and bring it within the U55C's 33.8 MiB of
URAM. That was tested by patching the kernel to round the accumulator to BF16 at
the end of **every token**, modelling hardware that computes in FP32 and stores
the state in BF16 memory; the output is taken from the unrounded state, matching
a design that computes from registers and rounds only on the store. Weights BF16,
activations FP32 (Arm B), so the state is the only new variable.

| Metric | FP32 | BF16 weights (Arm B) | BF16 weights + BF16 state | state vs FP32 |
|---|---:|---:|---:|---:|
| WikiText PPL | 16.82 | 16.82 | 16.83 | +0.01% |
| LAMBADA PPL | 9.72 | 9.70 | 9.69 | -0.32% |
| LAMBADA accuracy | 51.81 | 51.80 | 51.85 | +0.04 |
| PIQA | 73.78 | 73.72 | 73.67 | -0.11 |
| HellaSwag | 60.13 | 60.14 | 60.17 | +0.04 |
| WinoGrande | 61.88 | 61.88 | 61.88 | +0.00 |
| ARC-Easy | 72.31 | 72.22 | 72.22 | -0.08 |
| ARC-Challenge | 40.78 | 40.70 | 40.70 | -0.09 |
| SocialIQA | 42.37 | 42.48 | 42.48 | +0.10 |
| BoolQ | 61.68 | 61.56 | 61.59 | -0.09 |
| **Accuracy average** | **58.09** | **58.06** | **58.07** | **-0.02** |

**Verdict: pass, and effectively free.** The accuracy average falls 0.02 points
against a 0.5-point limit, and is marginally *better* than BF16 weights alone.
This is a stronger result than expected: the state is a running accumulator, so
rounding error compounds at every token, unlike weights which are rounded once.

**Tables 2 and 5 were then run with the same patch** (2026-08-20), against the
Arm B reference so the state is the only variable.

**Table 2 (RULER), BF16 state vs BF16 weights only:**

| Cell | wt only | wt + BF16 state | state effect |
|---|---:|---:|---:|
| S1 1K / 2K / 4K | 100.0 | 100.0 | +0.0 |
| S1 8K | 97.4 | **100.0** | **+2.6** |
| S2 1K / 2K | 100.0 | 100.0 | +0.0 |
| S2 4K | 85.6 | **91.6** | **+6.0** |
| S2 8K | 40.0 | **44.0** | **+4.0** |
| S3 1K | 81.8 | 80.8 | -1.0 |
| S3 2K | 80.4 | **82.8** | +2.4 |
| S3 4K | 53.6 | **56.6** | +3.0 |
| **Average** | **85.35** | **86.89** | **+1.55** |

**Table 5 (LongBench, answers cut at the first line):** average 18.75 to 18.73,
a change of **-0.02**. All six long-document QA tasks improved (+0.10 to +0.32);
the four largest losses are the two multi-document summarisation tasks
(multi_news -0.35, qmsum -0.19) and the two few-shot tasks (trec -1.50,
triviaqa -0.99).

**So BF16 recurrent state is strongly positive on pure needle retrieval and
neutral everywhere else.** Every pre-registered limit passes on all three tables.

**Harness noise, measured.** The RULER evaluation is *not* bit-reproducible: six
runs of an identical configuration at S2 4K gave 85.6, 85.0, 85.0, 85.0, 85.0,
85.0 - mean 85.10, standard deviation 0.22, full spread 0.6 (one sample in 500).
The BF16-state result of 91.6 is **29 standard deviations above that mean and
outside the entire observed range**, so the effect is not run-to-run variation.
The likely source is Triton selecting different autotune configurations per
process, which changes floating-point reduction order.

**This noise band applies to every Table 2 number in this document.** In
particular the BF16 Arm A macro figure of -0.31 lies *inside* it and must not be
read as a real degradation. Differences at or below ~0.6 points on Table 2 are
not resolvable without repeat runs.

**Mechanism.** The effect is consistent with *swamping* (stagnation), the
classical low-precision phenomenon in which adding a small value to a large
accumulator under round-to-nearest discards it entirely - the same effect that
makes stochastic rounding necessary for low-precision training. The recurrent
state accumulates one strong needle contribution against thousands of weak
distractor contributions; in BF16 each distractor term falls below the rounding
threshold relative to the accumulated signal and never accumulates, whereas in
FP32 it does. That predicts a task-dependent effect - retrieval gains,
many-small-contribution tasks do not - which is what Table 5 shows.

A literature check found the mechanism itself is textbook, but no published
precision ablation on a linear-attention recurrent state, and prevailing practice
is the opposite (keep the state in FP32 because error compounds). The KV-cache
quantisation literature reports retrieval as the *most* quantisation-sensitive
task; a recurrent state inverts that, plausibly because an attention cache is
re-read exactly while a recurrent state is accumulated.

**Boundaries of this result.** The patch was temporary and the kernel was
restored and verified byte-identical afterwards. Every BF16-state measurement
above uses **Arm B** (BF16 weights, FP32 activations). A GEMV-only mixed mode -
BF16 operands with FP32 accumulation and FP32 activations elsewhere - is a
different contract with **no quality data**, and would need its own run before
being adopted in hardware. Results are one seed and one checkpoint.

### All three in BF16 together (2026-08-23)

Every arm above varies **one** precision against a common reference, so the
configuration the accelerator would actually implement — weights, activations
and recurrent state all BF16 — had no data. This run supplies it, using the same
runner and the same converted checkpoint, `DTYPE=bfloat16` for the activations
and the same per-token state-rounding patch.

| Metric | FP32 | Arm A (W+act) | Arm B (W only) | B + BF16 state | **All BF16** |
|---|---:|---:|---:|---:|---:|
| Table 2 macro | 85.44 | 85.13 | 85.35 | 86.89 | **86.44** |
| Table 3 accuracy avg | 55.97 | 55.87 | 55.95 | 55.95 | **56.02** |
| Table 3 WikiText PPL | 16.824 | 16.841 | 16.824 | 16.825 | **16.841** |
| Table 5 macro (as run) | 15.13 | 15.01 | 15.09 | 15.01 | **15.18** |
| Table 5 macro (first line) | — | — | 18.75 | 18.73 | **18.88** |

#### Full five-arm tables

Regenerated from the stored artifacts of all five arms through one extraction,
which reproduces this document's existing FP32 / BF16 A / BF16 B figures exactly
(Table 2 85.44 / 85.13 / 85.35, Table 3 58.09 / 58.03 / 58.06, Table 5 first-line
18.82 / 18.69 / 18.75) — so the new column is on the same footing as the old ones.

**Table 2 — RULER S-NIAH (accuracy %)**

| Cell | Paper | FP32 | BF16 A | BF16 B | B+state | ALL BF16 | ALL - FP32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| S1 1K | 98.4 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | +0.0 |
| S1 2K | 88.4 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | +0.0 |
| S1 4K | 91.4 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | +0.0 |
| S1 8K | 91.8 | 97.4 | 97.6 | 97.4 | 100.0 | 100.0 | **+2.6** |
| S2 1K | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | +0.0 |
| S2 2K | 99.8 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | +0.0 |
| S2 4K | 92.2 | 85.6 | 85.4 | 85.6 | 91.6 | 90.6 | **+5.0** |
| S2 8K | 29.6 | 40.0 | 39.2 | 40.0 | 44.0 | 43.4 | **+3.4** |
| S3 1K | 86.6 | 82.0 | 80.2 | 81.8 | 80.8 | 78.2 | **-3.8** |
| S3 2K | 84.2 | 81.6 | 81.4 | 80.4 | 82.8 | 83.2 | +1.6 |
| S3 4K | 27.6 | 53.2 | 52.6 | 53.6 | 56.6 | 55.4 | +2.2 |
| **Macro average** | **80.91** | **85.44** | **85.13** | **85.35** | **86.89** | **86.44** | **+1.00** |

Every gain is at long context; the only loss, S3 1K, is the shortest, and is the
one cell where the all-BF16 arm is the weakest of the five.

**Table 3 — short-context and commonsense**

| Task | Paper | FP32 | BF16 A | BF16 B | B+state | ALL BF16 | ALL - FP32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| LAMBADA accuracy | 46.65 | 51.81 | 51.48 | 51.80 | 51.85 | 51.58 | -0.23 |
| PIQA | 72.25 | 73.78 | 73.72 | 73.72 | 73.67 | 73.72 | -0.05 |
| HellaSwag | 55.76 | 60.13 | 60.15 | 60.14 | 60.17 | 60.16 | +0.03 |
| WinoGrande | 57.45 | 61.88 | 61.80 | 61.88 | 61.88 | 62.35 | +0.47 |
| ARC-Easy | 71.21 | 72.31 | 72.22 | 72.22 | 72.22 | 72.47 | +0.17 |
| ARC-Challenge | 38.39 | 40.78 | 40.87 | 40.70 | 40.70 | 40.96 | +0.17 |
| SocialIQA | 40.63 | 42.37 | 42.32 | 42.48 | 42.48 | 42.48 | +0.10 |
| BoolQ | 60.24 | 61.68 | 61.68 | 61.56 | 61.59 | 61.74 | +0.06 |
| **Accuracy average** | **55.32** | **58.09** | **58.03** | **58.06** | **58.07** | **58.18** | **+0.09** |
| WikiText word PPL *(lower better)* | 16.42 | 16.824 | 16.841 | 16.824 | 16.825 | 16.841 | +0.105% |
| LAMBADA PPL *(lower better)* | 12.17 | 9.720 | 9.704 | 9.701 | 9.689 | 9.687 | -0.334% |

**Table 5 — LongBench v1, as run**

| Task | Paper | FP32 | BF16 A | BF16 B | B+state | ALL BF16 | ALL - FP32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| narrativeqa | 14.10 | 2.51 | 2.52 | 2.52 | 2.54 | 2.56 | +0.05 |
| qasper | 14.00 | 7.12 | 6.71 | 7.09 | 6.99 | 6.85 | -0.27 |
| multifieldqa_en | 23.30 | 13.58 | 13.45 | 13.58 | 13.82 | 13.93 | +0.35 |
| hotpotqa | 13.70 | 7.05 | 6.88 | 6.98 | 7.02 | 6.89 | -0.16 |
| 2wikimqa | 14.40 | 8.87 | 8.97 | 8.70 | 8.81 | 9.01 | +0.14 |
| musique | 5.80 | 3.16 | 3.16 | 3.12 | 3.32 | 3.34 | +0.18 |
| gov_report | 7.50 | 8.98 | 8.86 | 8.99 | 9.28 | 8.85 | -0.13 |
| qmsum | 16.40 | 18.47 | 18.52 | 18.42 | 18.24 | 18.02 | -0.45 |
| multi_news | 7.90 | 12.68 | 12.70 | 12.71 | 12.35 | 13.14 | +0.46 |
| trec | 30.00 | 36.50 | 36.00 | 36.50 | 35.00 | 36.50 | +0.00 |
| triviaqa | 22.40 | 26.27 | 25.44 | 26.21 | 25.22 | 25.58 | -0.69 |
| samsum | 23.00 | 30.15 | 30.04 | 29.95 | 30.53 | 30.77 | +0.62 |
| lcc | 18.70 | 18.46 | 18.50 | 18.45 | 18.83 | 18.78 | +0.32 |
| repobench-p | 22.10 | 18.06 | 18.31 | 18.12 | 18.23 | 18.32 | +0.26 |
| **Macro average** | **16.66** | **15.13** | **15.01** | **15.09** | **15.01** | **15.18** | **+0.05** |

**Table 5 — first line only.** Only the six long-document QA tasks change; the
other eight already emit a single line, so their rows are identical to as-run.

| Task | Paper | FP32 | BF16 A | BF16 B | B+state | ALL BF16 | ALL - FP32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| narrativeqa | 14.10 | 14.98 | 15.21 | 14.98 | 15.27 | 15.00 | +0.02 |
| qasper | 14.00 | 15.72 | 15.12 | 15.21 | 15.44 | 15.51 | -0.21 |
| multifieldqa_en | 23.30 | 25.56 | 25.13 | 25.46 | 25.64 | 26.23 | +0.67 |
| hotpotqa | 13.70 | 15.99 | 15.59 | 15.93 | 16.17 | 15.67 | -0.32 |
| 2wikimqa | 14.40 | 16.21 | 16.69 | 16.21 | 16.53 | 16.36 | +0.15 |
| musique | 5.80 | 5.47 | 5.57 | 5.42 | 5.52 | 5.62 | +0.15 |
| **Macro average** | **16.66** | **18.82** | **18.69** | **18.75** | **18.73** | **18.88** | **+0.06** |

#### Do not read the ranking as a result

The all-BF16 arm has the highest Table 3 average and the highest Table 5 macro of
all five arms. **That is not evidence it is better.** Against per-task sampling
error, every Table 3 difference is negligible:

| Task | FP32 | ALL BF16 | diff | stderr | diff/se |
|---|---:|---:|---:|---:|---:|
| LAMBADA accuracy | 51.81 | 51.58 | -0.23 | 0.70 | -0.33 |
| PIQA | 73.78 | 73.72 | -0.05 | 1.03 | -0.05 |
| HellaSwag | 60.13 | 60.16 | +0.03 | 0.49 | +0.06 |
| WinoGrande | 61.88 | 62.35 | +0.47 | 1.37 | +0.35 |
| ARC-Easy | 72.31 | 72.47 | +0.17 | 0.92 | +0.18 |
| ARC-Challenge | 40.78 | 40.96 | +0.17 | 1.44 | +0.12 |
| SocialIQA | 42.37 | 42.48 | +0.10 | 1.12 | +0.09 |
| BoolQ | 61.68 | 61.74 | +0.06 | 0.85 | +0.07 |
| **Average** | **58.09** | **58.18** | **+0.09** | **0.36** | **+0.25** |

Every task is under 0.35 standard errors and the average is 0.25. With five arms
all sitting inside noise on Tables 3 and 5, **one of them has to rank first** —
that is what a null result looks like with five samples, and it is the most
likely explanation of the all-BF16 arm's lead there.

The defensible statement is therefore narrower, and splits by quantity rather
than by arm:

- **BF16 costs nothing measurable** in weights, activations, or state, on any of
  the three tables.
- **BF16 recurrent state helps long-context retrieval**, by +1.3 to +1.5 macro
  with per-cell moves of +2.6 to +6.0 — several times the band measured on one of
  those very cells, so this one clears noise.
- **BF16 activations alone are marginally negative**: Arm A trails Arm B on all
  three tables (85.13 vs 85.35, 58.03 vs 58.06, 18.69 vs 18.75). Each gap is
  sub-noise, but the sign is consistent across every table, which is the expected
  direction.

The first point is what matters for hardware: the exact 8x8 multiply and the
halved activation storage are available without paying for them. The second is a
bonus pointing the same way. Neither needs the stronger reading.

Note also that the stderr column above is *sampling* error only. Run-to-run
variation was measured on a single Table 2 cell (about +/-0.6) and never on Tables
3 or 5, so these figures are a floor on the uncertainty, not all of it.


**Every pre-registered limit passes**, and three of the figures land on the
positive side of FP32: Table 2 macro **+1.00**, Table 3 accuracy **+0.09**,
Table 5 macro **+0.05**, WikiText perplexity **+0.10%** (limit 1%), worst Table 2
cell **-3.8** at S3 1K (limit 5.0). S3 1K is the one place this arm is the
weakest of the five and is worth watching.

Only the Table 2 figure is large enough to mean anything. The Table 3 and Table 5
margins are far inside sampling error and must not be read as the all-BF16 arm
being better — see *Do not read the ranking as a result* below, which quantifies
this.

**The state gain survives BF16 activations.** That was the open question, since
both changes act on the same accumulator and could have cancelled:

| Baseline | + BF16 state | gain |
|---|---:|---:|
| Arm B (FP32 activations) | 85.35 → 86.89 | **+1.54** |
| Arm A (BF16 activations) | 85.13 → 86.44 | **+1.31** |

Slightly attenuated, clearly present, and concentrated in the same cells: S1 8K
97.4 → 100.0, S2 4K 85.6 → 90.6, S2 8K 40.0 → 43.4. All-BF16 and B+state differ
by 0.45, which is inside the measured harness band, so on Table 2 the two are not
distinguishable — the activation precision costs nothing once the state is BF16.

**The patch was verified to apply, not assumed.** The earlier state experiment's
patch was never saved and had to be rebuilt. Because Triton keys its JIT cache on
kernel source, a rounding flag read from the environment can silently reuse a
kernel compiled without it — producing a "no difference" result that reads like a
finding. The rounding is therefore edited into the source, the Triton cache is
cleared, and the job gates twice on measured state contents: unpatched control
**0.000014** BF16-exact (so the state really is FP32), patched **1.000000** across
all 24 layer tensors. The kernel was restored from an EXIT trap and confirmed at
its original sha256 `752c117a…`.

**Caveats.** One seed, one checkpoint. The harness band of about ±0.6 was measured
on a single Table 2 cell, not on the macro average, so treat sub-point macro
differences as unresolved rather than as ties. Table 5's absolute values remain
depressed by the answer-length artifact documented in its own section; the
first-line column is the like-for-like comparison. This is a GPU study of a
numerical contract — it says nothing about whether the contract is routable, and
the Iter62 build that implemented packed BF16 weights failed placement-and-route
at congestion level 7.


**Why it does not unblock the accelerator.** Quality is no longer the obstacle,
but the physical one stands: 24 MiB of BF16 state is **683 URAM of 960**, while
each SLR holds only **320** and the recurrent islands are pblock-pinned to SLR2.
The state cannot sit in the SLR that consumes it, so the recurrent step would
reach across SLRs on 512-bit paths. Iter59 already measured a weaker version of
exactly that — 600 URAM moved on chip made congestion worse in all four
directions and `route_design` refused at level 7. One layer alone is 28 URAM and
fits easily, but leaving the other 23 in HBM keeps the traffic. The prize is
also bounded: state traffic is 1.9% of per-token bytes, and the latency upside
from the recurrent critical path is on the order of 10-15%.

### Native BF16 product rounded before FP32 accumulation (2026-08-27)

This follow-up isolates the numerical behavior offered by a compact native HLS
BF16 multiplier. It uses the same all-BF16 operands, transient boundaries,
convolution tails, and per-token BF16 recurrent-state persistence as the prior
all-BF16 arm, but changes every HBM-backed dense product from an exact
BF16-by-BF16 product represented in FP32 to:

```text
BF16 weight * BF16 activation -> RNE-rounded BF16 product -> FP32 reduction
```

The emulator patched all 193 dense modules (the eight projections in each of
24 layers plus the LM head), retained four FP32 partial banks and the balanced
16-product reduction trees, and emitted FP32 logits. The arithmetic preflight
matched an independent bit-level reference at every one of 6,823 tested FP32
outputs. Slurm job 1213 then completed the full suite on an A100 80-GB in
09:33:29 with exit code zero. All reference counts are present: 5,500 Table 2
samples, every Table 3 task at its full harness count, and all 3,350 Table 5
examples.

In the tables below, **Exact-product BF16** is the 2026-08-23 all-BF16 arm and
**Native-product BF16** is the new rounded-product arm. Deltas are against the
measured FP32 checkpoint and against the exact-product arm, not inferred from
the paper.

**Table 2 — RULER S-NIAH (accuracy %)**

| Cell | Paper | FP32 | Exact-product BF16 | Native-product BF16 | Native - FP32 | Native - exact |
|---|---:|---:|---:|---:|---:|---:|
| S1 1K | 98.4 | 100.0 | 100.0 | 100.0 | +0.0 | +0.0 |
| S1 2K | 88.4 | 100.0 | 100.0 | 100.0 | +0.0 | +0.0 |
| S1 4K | 91.4 | 100.0 | 100.0 | 100.0 | +0.0 | +0.0 |
| S1 8K | 91.8 | 97.4 | 100.0 | 100.0 | +2.6 | +0.0 |
| S2 1K | 100.0 | 100.0 | 100.0 | 100.0 | +0.0 | +0.0 |
| S2 2K | 99.8 | 100.0 | 100.0 | 100.0 | +0.0 | +0.0 |
| S2 4K | 92.2 | 85.6 | 90.6 | 91.2 | +5.6 | +0.6 |
| S2 8K | 29.6 | 40.0 | 43.4 | 44.2 | +4.2 | +0.8 |
| S3 1K | 86.6 | 82.0 | 78.2 | 80.4 | -1.6 | +2.2 |
| S3 2K | 84.2 | 81.6 | 83.2 | 83.2 | +1.6 | +0.0 |
| S3 4K | 27.6 | 53.2 | 55.4 | 56.6 | +3.4 | +1.2 |
| **Macro average** | **80.91** | **85.44** | **86.44** | **86.87** | **+1.44** | **+0.44** |

**Table 3 — short-context and commonsense**

| Metric | Paper | FP32 | Exact-product BF16 | Native-product BF16 | Native - FP32 | Native - exact |
|---|---:|---:|---:|---:|---:|---:|
| WikiText word PPL *(lower better)* | 16.42 | 16.824 | 16.841 | 16.827 | +0.003 | -0.014 |
| LAMBADA PPL *(lower better)* | 12.17 | 9.720 | 9.687 | 9.693 | -0.028 | +0.006 |
| LAMBADA accuracy | 46.65 | 51.81 | 51.58 | 51.80 | -0.02 | +0.21 |
| PIQA | 72.25 | 73.78 | 73.72 | 73.88 | +0.11 | +0.16 |
| HellaSwag | 55.76 | 60.13 | 60.16 | 60.17 | +0.04 | +0.01 |
| WinoGrande | 57.45 | 61.88 | 62.35 | 62.27 | +0.39 | -0.08 |
| ARC-Easy | 71.21 | 72.31 | 72.47 | 72.31 | +0.00 | -0.17 |
| ARC-Challenge | 38.39 | 40.78 | 40.96 | 40.78 | +0.00 | -0.17 |
| SocialIQA | 40.63 | 42.37 | 42.48 | 42.32 | -0.05 | -0.15 |
| BoolQ | 60.24 | 61.68 | 61.74 | 61.47 | -0.21 | -0.28 |
| **Accuracy average** | **55.32** | **58.09** | **58.18** | **58.13** | **+0.03** | **-0.06** |

**Table 5 — LongBench v1, as run**

| Task | Paper | FP32 | Exact-product BF16 | Native-product BF16 | Native - FP32 | Native - exact |
|---|---:|---:|---:|---:|---:|---:|
| narrativeqa | 14.10 | 2.51 | 2.56 | 2.55 | +0.03 | -0.02 |
| qasper | 14.00 | 7.12 | 6.85 | 6.80 | -0.31 | -0.05 |
| multifieldqa_en | 23.30 | 13.58 | 13.93 | 13.77 | +0.20 | -0.16 |
| hotpotqa | 13.70 | 7.05 | 6.89 | 6.95 | -0.10 | +0.06 |
| 2wikimqa | 14.40 | 8.87 | 9.01 | 9.16 | +0.28 | +0.15 |
| musique | 5.80 | 3.16 | 3.34 | 3.48 | +0.32 | +0.14 |
| gov_report | 7.50 | 8.98 | 8.85 | 8.88 | -0.10 | +0.03 |
| qmsum | 16.40 | 18.47 | 18.02 | 18.29 | -0.18 | +0.27 |
| multi_news | 7.90 | 12.68 | 13.14 | 12.49 | -0.20 | -0.66 |
| trec | 30.00 | 36.50 | 36.50 | 35.50 | -1.00 | -1.00 |
| triviaqa | 22.40 | 26.27 | 25.58 | 25.28 | -0.98 | -0.30 |
| samsum | 23.00 | 30.15 | 30.77 | 30.86 | +0.71 | +0.10 |
| lcc | 18.70 | 18.46 | 18.78 | 19.02 | +0.57 | +0.24 |
| repobench-p | 22.10 | 18.06 | 18.32 | 18.26 | +0.19 | -0.06 |
| **Macro average** | **16.60** | **15.13** | **15.18** | **15.09** | **-0.04** | **-0.09** |

The known LongBench answer-length artifact remains: rescoring the six QA tasks
at the first generated line raises the native-product macro from 15.09 to
**18.88**, versus 18.82 for FP32 and 18.88 for exact-product BF16. The raw
paper comparison therefore remains unsuitable for judging checkpoint quality;
the precision comparison is paired and uses an identical harness.

**Verdict: quality pass.** Native-product BF16 passes every pre-registered gate:
Table 2 is +1.44 points versus FP32 (worst cell -1.6), Table 3 accuracy is +0.03
with only +0.019% WikiText perplexity, and raw Table 5 is -0.04 with a worst-task
drop of -1.00. It is statistically indistinguishable in aggregate quality from
the exact-product arm; the apparent Table 2 improvement must not be treated as
a model-quality gain because RULER has a measured run-to-run band.

The numerical contract is nevertheless observably different. Against FP32,
105 of 5,500 RULER answers changed (92 wrong-to-right and 13 right-to-wrong),
and only 1,935 of 3,350 LongBench generations (57.8%) were byte-identical.
Against exact-product BF16, 38 RULER answers changed and 2,371 LongBench
generations (70.8%) were byte-identical. A native-product FPGA must therefore
use a newly generated product-rounded all-BF16 logit/trajectory golden; passing
aggregate quality does not authorize comparison against the old exact-product
golden.

Raw results are under
`/home/yaoz0b/gdn_precision_eval_20260827/native_bf16_product/`; the Slurm log is
`c_impl/diagnostics/bf16_native_product_quality_20260827/slurm-1213.out`.
The committed aggregate results, artifact hashes, code map, and reproduction
instructions are in `c_impl/results_native_bf16_product/README.md`.

### FP32 summary across the three tables

Removing the two cells whose construction is disputed, the three tables
agree on the size of the gap. Table 2 without the S3 4K cell is +2.42,
Table 3 is +2.77, and Table 5 under truncation is +2.16. Three independent
task families converging near +2 to +3 points is consistent with a
third-party checkpoint trained on different data being modestly stronger
than the paper's model. The raw spread of +4.53 / +2.77 / -1.53 is
dominated by harness differences in both directions and should not be
quoted as a reproduction of the paper's numbers.

## On-hardware result (WikiText, BF16 weights + FP32 math)

*Measured 2026-08-20 on the U55C with the Iter61 image. This is the first
benchmark number produced by the accelerator itself rather than by a GPU.*

**Configuration.** `scripts/export_gdn_c.py` forces FP32 storage
(`.float()` at the tensor write), so exporting the BF16 checkpoint yields a
`.gdnw` blob holding **BF16-rounded values in FP32 words** -- precisely
"BF16 weights, FP32 arithmetic", and it needs **no kernel change**. Verified on
the blob: 100% of values carry zero low-16 mantissa bits, worst relative change
3.8908e-3 against the BF16 round-to-nearest bound of 3.906e-3.

**Method.** WikiText-2 is `loglikelihood_rolling` -- sequential scoring with no
prefill and no generation -- which is what a decode-only accelerator does
natively. `host.cpp`'s rolling scorer was prefill-era (one `run_forward` per
window, N hidden rows back) and was rewritten to walk each window one token at
a time, resetting the recurrent state per window and taking log-probabilities
from **the kernel's own on-chip LM head** via the Iter61 logit stream.

**Bridge validation.** Scoring example 0 with **FP32** weights reproduced the
committed Python golden: **-3017.65073425** on card against **-3017.65085554**
on GPU, a difference of 1.2e-4 over 1,512 sequential forward passes (4e-8
relative).

**Result.** 62 documents, 183 windows, **360,389 forward passes**, 4.3 h at a
sustained **42.182 ms/token**.

| Configuration | word perplexity |
|---|---:|
| **FPGA, BF16 weights + FP32 math** | **16.824190767** |
| GPU, BF16 weights + FP32 math (Arm B) | 16.824191512 |
| **FPGA - GPU (matched reference)** | **-0.0000007  (-0.000004%)** |
| GPU, FP32 weights (context only) | 16.823770819 |
| FPGA vs FP32 GPU | +0.000420  (+0.0025%) |

**Compare against the matched reference, not the FP32 golden.** The +0.0025%
gap against FP32 is the BF16 weight cast, exactly as this document's GPU study
measured (+0.0025% on the same metric); attributing it to the hardware would be
wrong. Against the configuration the FPGA actually runs, the hardware
difference is **7e-7 perplexity**.

Two cross-checks worth keeping: the committed `.gdnreq` golden (16.823771361)
and lm-eval's FP32 WikiText (16.823770819) agree to 5e-7 despite different
harnesses and windowing; and byte perplexity and bits-per-byte track the same
way (1.695336 vs 1.695328, 0.761571 vs 0.761565).

**What this establishes and what it does not.** It establishes that the
decode-only sequential scoring path is correct at scale, that the Iter61 logit
stream is trustworthy over 360,389 tokens rather than the 64 the decode gate
covers, and that BF16 weights cost what the GPU predicted. It is **one row of
Table 3**. The other eight tasks need per-candidate context replay, and Tables 2
and 5 need a generation path that does not yet exist; at 42 ms/token the full
three tables are an estimated 35-50 h of card time.

## Reproduction and raw evidence

The evaluation launchers are:

- `scripts/run_gdn_table2_eval.sh`
- `scripts/run_gdn_table3_eval.sh`
- `scripts/run_gdn_longbench_eval.py`
- `scripts/convert_gdn_checkpoint_bf16.py`
- `scripts/summarize_gdn_quality_eval.py` — renders all three tables and the
  acceptance verdicts from the stored artifacts
- `scripts/run_gdn_precision_eval_arm.sh` — runs one complete arm (Tables 2, 3,
  and 5) for a given `MODEL_ID` and `DTYPE`; the three arms below differ only in
  those two values
- `scripts/verify_bf16_conversion.py` — pre-flight check on a converted
  checkpoint: counts, dtypes, untouched non-float tensors, and the
  round-to-nearest bound
- `scripts/analyze_gdn_eval_artifacts.py` — the first-line Table 5 rescoring and
  the paired per-sample flip counts quoted in the sections above

Raw model outputs, lm-eval result JSON, manifests, wrapper logs, PIDs, and exit
markers are intentionally outside the Git repository under
`/home/yaoz0b/gdn_precision_eval_20260817/`. Generated checkpoints and result
artifacts are not committed.

### Rendering the tables

FP32 versus BF16 Arm A, the pre-registered comparison:

```bash
python scripts/summarize_gdn_quality_eval.py /home/yaoz0b/gdn_precision_eval_20260817
```

FP32 versus BF16 Arm B. The summarizer requires the two arms to be named
`fp32/` and `bf16/`, so Arm B is presented through a temporary symlink root:

```bash
R=/home/yaoz0b/gdn_precision_eval_20260817; d=$(mktemp -d); ln -s $R/fp32 $d/fp32; ln -s $R/bf16_weights_only $d/bf16; python scripts/summarize_gdn_quality_eval.py $d
```

The summarizer validates every Table 3 sample count against
`TABLE3_EXPECTED_COUNTS` and every Table 5 task name before rendering, so a
truncated or partial run fails loudly rather than reporting a silently wrong
average. It reports the as-run Table 5 numbers only; the first-line-truncation
variant and the paired flip counts come from:

```bash
R=/home/yaoz0b/gdn_precision_eval_20260817
python scripts/analyze_gdn_eval_artifacts.py --table5-truncated   $R/fp32
python scripts/analyze_gdn_eval_artifacts.py --pair-table2        $R/fp32 $R/bf16
python scripts/analyze_gdn_eval_artifacts.py --pair-table5-preds  $R/fp32 $R/bf16_weights_only
```

### Verification performed on these results

Beyond the acceptance gates, the following were checked directly from the
stored artifacts rather than assumed:

- **Loaded precision**, from each Table 5 manifest's `parameter_elements_by_dtype`
  rather than the requested setting: FP32 holds 1,466,343,808 float32 elements;
  Arm A holds 1,466,343,616 bfloat16 plus 192 float32; Arm B holds
  1,466,343,808 float32, confirming the BF16 file was upcast on load.
- **Comparability**: all three arms ran at identical batch sizes (4 / 16 / 16),
  no sample limit, `attn_mode=fused_recurrent`, the same PyTorch build, and the
  same 4,096-token input cap.
- **Sample counts**: identical across arms — all nine Table 3 tasks at their
  reference counts, Table 2 at 2,000/2,000/1,500, Table 5 at 3,350.
- **Reproducibility**: re-running PIQA on FP32 from scratch returned 73.7758
  accuracy and 74.1023 normalised accuracy, matching the recorded values to
  four decimals.
- **Per-answer agreement**: of the 3,350 generated Table 5 answers, 92.9% are
  byte-identical between FP32 and Arm B, and 75.8% between FP32 and Arm A.

That last figure carries a consequence for the accelerator. BF16 preserves
*quality* but not *output identity*: roughly 7% of long-generation answers
differ under Arm B. The kernel's exact-match decode gate therefore cannot be
retained against the current FP32 golden if weights move to BF16 — the golden
must be regenerated from the BF16 weights and the gate re-based on it.

Known gaps in this audit: the `model_index_blob` provenance field is empty in
both BF16 Table 5 manifests but populated for FP32, and the reproducibility
re-run covered a scoring-only task, not batched generation.

### Arm layout on disk

| Arm | Directory | Load dtype |
|---|---|---|
| FP32 | `fp32/` | `float32` on the original checkpoint |
| BF16 A | `bf16/` | `bfloat16` on `checkpoint_bf16_mixed/` |
| BF16 B | `bf16_weights_only/` | `float32` on `checkpoint_bf16_mixed/` |

Both BF16 arms load the same converted checkpoint; only the dtype differs.
Wrapper logs, PIDs, and exit markers for each arm are under
`<arm>/full_evaluation/`. All three arms exited 0.
