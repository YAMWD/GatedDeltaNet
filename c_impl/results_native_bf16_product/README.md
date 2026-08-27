# Native-BF16 Product Quality Evaluation

This directory preserves the complete aggregate evidence for the 2026-08-27
GPU evaluation of the numerical contract proposed for a compact FPGA-native
BF16 multiplier. The full Tables 2, 3, and 5 suite passed its pre-registered
quality limits.

## Numerical contract

The tested dense operation is deliberately different from ordinary BF16 Tensor
Core GEMM:

```text
BF16 weight x BF16 activation
    -> round each scalar product RNE to BF16
    -> widen the rounded product exactly to FP32
    -> balanced 16-product FP32 trees
    -> four FP32 partial banks
    -> (bank0 + bank1) + (bank2 + bank3)
```

The rest of the evaluated model contract is:

| Boundary or operation | Representation |
|---|---|
| Dense weights | BF16 |
| Transient operator boundaries | BF16 |
| Dense product | BF16, RNE after every scalar multiply |
| GEMV reduction and accumulation | FP32 |
| Normalization, convolution, residual, recurrent arithmetic | FP32 |
| Persistent recurrent state | BF16, rounded after every token |
| Persistent convolution tail | BF16 |
| LM-head output | Full 32,000-element FP32 logits |

Exactly 193 HBM-backed dense modules are patched: q/k/v/g/output and the three
MLP projections in each of 24 layers, plus the LM head. Tiny attention `a`/`b`
projections are intentionally not patched. Dense outputs other than the LM head
are rounded back to BF16. The LM head remains FP32 so evaluation tasks consuming
log-likelihoods see the same interface expected from the accelerator.

The prior `exact_product_bf16` comparison arm has the same BF16 boundaries and
state persistence, but feeds the exact BF16-by-BF16 product into FP32
accumulation. The new `native_product_bf16` arm adds the product-rounding point.

## Experiment

The evaluated checkpoint is the converted BF16 form of
`m-a-p/1.3B-100B-GatedDeltaNet-pure`, loaded from
`/home/yaoz0b/gdn_precision_eval_20260817/checkpoint_bf16_mixed`. The run used
the same prompts, seeds, scorers, sample counts, 4,096-token input cap, fused
recurrent attention mode, and greedy decoding as the existing FP32 and
exact-product all-BF16 arms.

Slurm job 1213 ran on one NVIDIA A100 80-GB PCIe GPU on `acclnode01`. It
completed with exit code zero in 09:33:29. Before starting the long suite, the
launcher required:

1. bit-exact agreement between the Triton emulator and an independent
   four-bank reference;
2. generated PTX containing BF16 conversion operations and no `mma.sync`;
3. proof that the control recurrent state is FP32 and the patched state is
   BF16-exact after every token;
4. exactly 193 patched dense modules; and
5. finite full-model FP32 logits from a smoke forward.

The arithmetic gate covered 23 shapes and 6,823 FP32 outputs with zero bit
mismatches. All 512 outputs in a separate discriminator differed from ordinary
exact-product GEMM, proving the new rounding point was active.

## Headline results

| Metric | Paper | FP32 | Exact-product BF16 | Native-product BF16 | Native - FP32 |
|---|---:|---:|---:|---:|---:|
| Table 2 RULER macro | 80.91 | 85.44 | 86.44 | **86.87** | +1.44 |
| Table 3 metric-mapped accuracy | 55.32 | 58.09 | 58.18 | **58.13** | +0.03 |
| Table 3 WikiText PPL | 16.42 | 16.824 | 16.841 | **16.827** | +0.003 |
| Table 3 LAMBADA PPL | 12.17 | 9.720 | 9.687 | **9.693** | -0.028 |
| Table 5 macro, as run | 16.60 | 15.13 | 15.18 | **15.09** | -0.04 |
| Table 5 macro, first-line rescore | 16.66 | 18.82 | 18.88 | **18.88** | +0.06 |

All full per-cell and per-task values are in `comparison_summary.json` and the
original aggregate result JSON files in this directory. Sample counts are:

- Table 2: 5,500 total, 500 per cell;
- Table 3: every task at the full lm-eval reference count; and
- Table 5: all 3,350 examples across 14 tasks.

The pre-registered BF16-versus-FP32 limits all pass. The worst Table 2 cell
movement is -1.6 points, Table 3 accuracy changes by +0.03 points, WikiText PPL
changes by +0.019%, Table 5 changes by -0.04 points, and its worst task movement
is -1.00 point.

The result establishes quality compatibility, not output identity. Only
1,935/3,350 LongBench generations are byte-identical to FP32, and 2,371/3,350
are identical to the exact-product BF16 arm. An FPGA implementation of this
contract therefore needs its own product-rounded full-logit and trajectory
golden.

The low raw Table 5 macro is the already documented answer-length artifact:
six QA tasks often generate a correct first line followed by extra text that
reduces F1 precision. First-line rescoring raises the native-product macro from
15.09 to 18.88. Precision comparisons remain valid because all arms use the
same harness.

## Files

| File | Purpose |
|---|---|
| `comparison_summary.json` | Machine-readable paper/FP32/exact/native comparison, counts, paired-output analysis, and verdict. |
| `arithmetic_manifest.json` | Exact arithmetic contract and all 193 patched module names. |
| `table2_s12_results.json` | Original aggregate lm-eval output for RULER S1 and S2. |
| `table2_s3_results.json` | Original aggregate lm-eval output for RULER S3. |
| `table3_results.json` | Original aggregate lm-eval output for all Table 3 tasks. |
| `table5_results.json` | Original LongBench per-task scores and counts. |
| `table5_manifest.json` | Dataset, model, protocol, dependency, and arithmetic metadata. |
| `verify_results.py` | Dependency-free consistency gate for the committed summary and aggregate JSON files. |
| `evaluated_code_sha256.txt` | Hashes proving which source files produced the results. |
| `raw_artifact_sha256.txt` | Hashes of external aggregate and per-sample artifacts. |

The corresponding code is:

- `scripts/gdn_native_bf16_product.py`: Triton emulator, independent reference,
  and exact dense-module patch;
- `scripts/test_gdn_native_bf16_product.py`: arithmetic, generated-PTX, and
  benchmark gate;
- `scripts/verify_gdn_native_bf16_product_model.py`: full-model patch and
  FP32-logit smoke gate;
- `scripts/fla_lm_eval.py`: lm-eval integration;
- `scripts/run_gdn_longbench_eval.py`: LongBench integration and manifest;
- `scripts/run_gdn_native_bf16_product_eval.slurm`: complete gated A100 run;
- `scripts/gdn_bf16_state_patch.sh`: temporary per-token BF16-state patch with
  restoration; and
- `scripts/run_gdn_precision_eval_arm.sh`: common Tables 2/3/5 driver.

## Reproduce the full evaluation

The launcher contains the exact repository, checkpoint, output, environment,
and diagnostics paths used by job 1213. Submit it from `acclhead1`:

```bash
cd /home/yaoz0b/GatedDeltaNet
PATH=/opt/slurm/current/bin:$PATH sbatch scripts/run_gdn_native_bf16_product_eval.slurm
```

The job writes generated outputs outside Git under:

```text
/home/yaoz0b/gdn_precision_eval_20260827/native_bf16_product/
```

and its persistent live logs under:

```text
/home/yaoz0b/GatedDeltaNet/c_impl/diagnostics/bf16_native_product_quality_20260827/
```

The launcher temporarily patches the installed FLA recurrent kernel to round
persistent state after every token. An EXIT trap restores the original source
and verifies its SHA-256. Do not run two jobs sharing the same Python environment
concurrently, because they would race on that temporary site-package patch.

## Run only the arithmetic gate

Inside an A100 Slurm allocation with the repository environment available:

```bash
cd /home/yaoz0b/GatedDeltaNet
export PYTHONPATH="$PWD/scripts:${PYTHONPATH:-}"
.micromamba/envs/gdn-hf/bin/python scripts/test_gdn_native_bf16_product.py --benchmark
```

This must report zero exact mismatches, at least one ordinary-GEMM difference,
BF16 operations in PTX, and zero `mma.sync` instructions.

## Re-render and inspect

The committed summary is directly readable with:

```bash
python3 -m json.tool c_impl/results_native_bf16_product/comparison_summary.json
```

Verify the committed aggregate files and all native-product cells with:

```bash
python3 c_impl/results_native_bf16_product/verify_results.py
```

When the external FP32 and exact-product arms are present, render the standard
FP32-versus-native tables through the existing summarizer:

```bash
root=$(mktemp -d /tmp/gdn-native-summary.XXXXXX)
ln -s /home/yaoz0b/gdn_precision_eval_20260817/fp32 "$root/fp32"
ln -s /home/yaoz0b/gdn_precision_eval_20260827/native_bf16_product "$root/bf16"
.micromamba/envs/gdn-hf/bin/python scripts/summarize_gdn_quality_eval.py "$root"
```

Supporting first-line and paired-output analyses are:

```bash
.micromamba/envs/gdn-hf/bin/python scripts/analyze_gdn_eval_artifacts.py \
  --table5-truncated /home/yaoz0b/gdn_precision_eval_20260827/native_bf16_product
.micromamba/envs/gdn-hf/bin/python scripts/analyze_gdn_eval_artifacts.py \
  --pair-table2 /home/yaoz0b/gdn_precision_eval_20260817/fp32 \
  /home/yaoz0b/gdn_precision_eval_20260827/native_bf16_product
.micromamba/envs/gdn-hf/bin/python scripts/analyze_gdn_eval_artifacts.py \
  --pair-table5-preds /home/yaoz0b/gdn_precision_eval_20260817/bf16_all \
  /home/yaoz0b/gdn_precision_eval_20260827/native_bf16_product
```

## Artifact policy

The aggregate JSON evidence is committed here. Checkpoints, generated text,
per-sample RULER logs, LongBench prediction JSONL, wrapper logs, and Python
environments are generated artifacts and remain outside Git. Their exact
locations and SHA-256 values are preserved in `raw_artifact_sha256.txt`; this
keeps the commit reviewable without discarding auditability.
