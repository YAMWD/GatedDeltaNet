#!/usr/bin/env python3
"""Check a native or hardware decode result against the cached GPU golden."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def _summarize_ms(values: list[float]) -> dict[str, float] | None:
    """median/mean of a flat list of per-step millisecond timings."""
    vals = [float(v) for v in values]
    if not vals:
        return None
    return {"median_ms": statistics.median(vals), "mean_ms": statistics.mean(vals)}


def compare_decode(golden_path: Path, c_path: Path) -> dict:
    """Compare a decode result file (C native or HW) against the decode golden.

    Schemas (FROZEN):
      golden example: index, prompt_ids, golden_traj[N], per_step_argmax[N],
                      per_step_logprob[N]   (golden_traj == per_step_argmax)
      C/HW example:   index, gen_traj[N], tf_argmax[N], per_step_tpot_ms[N],
                      and (HW only) kernel_ms[N]
      C native root:  optional logits_parity summary covering every pre-argmax
                      logit produced during the decode

    The golden may cover more examples / more positions than the C/HW file
    (e.g. golden N=64, E=12 vs a fast C run N=6, E=1). We compare only the
    overlap: examples matched by ``index``, over the first
    ``min(golden_len, c_len)`` positions of each.

    Metrics:
      * exact_traj_match     gen_traj == golden_traj (per example; aggregate =
                             all compared examples match -> overall PASS/FAIL)
      * first_divergence_index  first i where gen_traj[i] != golden_traj[i]
                             (-1 if none); aggregate = min over examples
      * top1_agreement_rate  fraction of positions where tf_argmax[i] ==
                             golden_traj[i]; aggregate = mean over positions
      * tpot_summary         median/mean ms from per_step_tpot_ms (and
                             kernel_ms when present, HW)
      * logits_parity        full-vocabulary comparison against the independent
                             scalar CPU LM head, plus optional exact-reference
                             and captured-argmax checks
    """
    golden = json.loads(golden_path.read_text())
    c = json.loads(c_path.read_text())

    golden_by_index = {int(ex["index"]): ex for ex in golden.get("examples", [])}
    c_examples = c.get("examples", [])

    has_kernel = any("kernel_ms" in ex for ex in c_examples)

    per_example: list[dict] = []
    all_exact = True
    overall_first_div = -1            # min first-divergence across examples (-1 == none)
    top1_hits = 0
    top1_total = 0
    tpot_all: list[float] = []
    kernel_all: list[float] = []
    compared_examples = 0
    skipped: list[int] = []

    for c_ex in c_examples:
        idx = int(c_ex["index"])
        g_ex = golden_by_index.get(idx)
        if g_ex is None:
            skipped.append(idx)
            continue
        compared_examples += 1

        golden_traj = [int(x) for x in g_ex["golden_traj"]]
        gen_traj = [int(x) for x in c_ex["gen_traj"]]
        tf_argmax = [int(x) for x in c_ex["tf_argmax"]]
        n = min(len(golden_traj), len(gen_traj), len(tf_argmax))

        g_slice = golden_traj[:n]
        gen_slice = gen_traj[:n]
        tf_slice = tf_argmax[:n]

        exact = gen_slice == g_slice
        all_exact = all_exact and exact

        first_div = -1
        for i in range(n):
            if gen_slice[i] != g_slice[i]:
                first_div = i
                break
        if first_div != -1:
            overall_first_div = (
                first_div if overall_first_div == -1 else min(overall_first_div, first_div)
            )

        ex_top1_hits = sum(1 for i in range(n) if tf_slice[i] == g_slice[i])
        top1_hits += ex_top1_hits
        top1_total += n

        tpot = [float(v) for v in c_ex.get("per_step_tpot_ms", [])][:n]
        tpot_all.extend(tpot)
        kernel = [float(v) for v in c_ex.get("kernel_ms", [])][:n]
        kernel_all.extend(kernel)

        per_example.append(
            {
                "index": idx,
                "compared_positions": n,
                "exact_traj_match": exact,
                "first_divergence_index": first_div,
                "top1_agreement_rate": (ex_top1_hits / n) if n else 0.0,
                "tpot_summary": _summarize_ms(tpot),
                "kernel_summary": _summarize_ms(kernel) if has_kernel else None,
            }
        )

    logits_parity = c.get("logits_parity")
    logits_parity_pass = None
    if logits_parity is not None:
        checked_steps = int(logits_parity.get("checked_steps", 0))
        compared_values = int(logits_parity.get("compared_values", 0))
        if checked_steps > 0 and compared_values > 0:
            exact_required = bool(
                logits_parity.get("exact_reference_required", True)
            )
            logits_parity_pass = bool(
                int(logits_parity.get("cpu_tolerance_failures", 0)) == 0
                and int(logits_parity.get("nonfinite_mismatches", 0)) == 0
                and (
                    not exact_required
                    or int(logits_parity.get("exact_reference_mismatches", 0)) == 0
                )
                and int(logits_parity.get("argmax_mismatches", 0)) == 0
            )

    aggregate = {
        "exact_traj_match": bool(all_exact and compared_examples > 0),
        "first_divergence_index": overall_first_div,
        "top1_agreement_rate": (top1_hits / top1_total) if top1_total else 0.0,
        "tpot_summary": _summarize_ms(tpot_all),
        "kernel_summary": _summarize_ms(kernel_all) if has_kernel else None,
        "compared_examples": compared_examples,
        "compared_positions_total": top1_total,
        "logits_parity_pass": logits_parity_pass,
    }

    return {
        "kind": int(c.get("kind", golden.get("kind", 2))),
        "golden_path": str(golden_path),
        "c_path": str(c_path),
        "golden_decode_len": int(golden.get("decode_len", 0)),
        "c_decode_len": int(c.get("decode_len", 0)),
        "golden_num_examples": int(golden.get("num_examples", len(golden_by_index))),
        "c_num_examples": int(c.get("num_examples", len(c_examples))),
        "has_kernel_ms": has_kernel,
        "skipped_c_examples_not_in_golden": skipped,
        "logits_parity": logits_parity,
        "aggregate": aggregate,
        "examples": per_example,
    }


def _fmt_ms(summary: dict | None) -> str:
    if not summary:
        return "n/a"
    return f"median={summary['median_ms']:.3f} mean={summary['mean_ms']:.3f}"


def print_decode_report(report: dict) -> None:
    agg = report["aggregate"]
    print("=" * 70)
    print("GatedDeltaNet decode-correctness report")
    print("=" * 70)
    print(f"  golden : {report['golden_path']}")
    print(f"  candidate : {report['c_path']}")
    print(
        f"  golden  decode_len={report['golden_decode_len']} "
        f"num_examples={report['golden_num_examples']}"
    )
    print(
        f"  candidate decode_len={report['c_decode_len']} "
        f"num_examples={report['c_num_examples']}  "
        f"(kernel_ms present: {report['has_kernel_ms']})"
    )
    if report["skipped_c_examples_not_in_golden"]:
        print(
            f"  WARNING: candidate example indices not found in golden, skipped: "
            f"{report['skipped_c_examples_not_in_golden']}"
        )
    print("-" * 70)
    print(
        f"  compared {agg['compared_examples']} example(s), "
        f"{agg['compared_positions_total']} position(s) total"
    )
    print("  per-example:")
    for ex in report["examples"]:
        status = "MATCH" if ex["exact_traj_match"] else "DIVERGE"
        print(
            f"    idx={ex['index']:>3}  n={ex['compared_positions']:>3}  "
            f"exact={status:<7}  first_div={ex['first_divergence_index']:>3}  "
            f"top1={ex['top1_agreement_rate'] * 100:6.2f}%  "
            f"tpot[{_fmt_ms(ex['tpot_summary'])}]"
            + (f"  kernel[{_fmt_ms(ex['kernel_summary'])}]" if report["has_kernel_ms"] else "")
        )
    print("-" * 70)
    print("  AGGREGATE:")
    print(f"    exact_traj_match      : {agg['exact_traj_match']}")
    print(f"    first_divergence_index: {agg['first_divergence_index']}  (-1 = no divergence)")
    print(f"    top1_agreement_rate   : {agg['top1_agreement_rate'] * 100:.2f}%")
    print(f"    tpot_summary (ms/tok) : {_fmt_ms(agg['tpot_summary'])}")
    if report["has_kernel_ms"]:
        print(f"    kernel_summary (ms/tok): {_fmt_ms(agg['kernel_summary'])}")
    logits = report["logits_parity"]
    if logits is not None:
        print("    full_logits_parity:")
        print(
            f"      checked_steps={int(logits.get('checked_steps', 0))} "
            f"values={int(logits.get('compared_values', 0))} "
            f"max_abs={float(logits.get('max_abs_error', 0.0)):.9g} "
            f"max_rel={float(logits.get('max_rel_error', 0.0)):.9g}"
        )
        print(
            f"      cpu_tol_fail={int(logits.get('cpu_tolerance_failures', 0))} "
            f"nonfinite_mismatch={int(logits.get('nonfinite_mismatches', 0))} "
            f"exact_ref_mismatch={int(logits.get('exact_reference_mismatches', 0))} "
            f"exact_required={bool(logits.get('exact_reference_required', True))} "
            f"argmax_mismatch={int(logits.get('argmax_mismatches', 0))} "
            f"pass={agg['logits_parity_pass']}"
        )
    print(
        "    NOTE: decode-from-state timings are one O(1) recurrent decode call "
        "per generated token; the seed entry has no kernel invocation."
    )
    print("-" * 70)
    passed = bool(
        agg["exact_traj_match"]
        and (agg["logits_parity_pass"] is not False)
    )
    banner = "PASS" if passed else "FAIL"
    gate = "exact trajectory"
    if logits is not None:
        gate += " + full pre-argmax logits"
    print(f"  RESULT: {banner}  (gates on {gate})")
    print("=" * 70)


def run_decode(args: argparse.Namespace) -> int:
    report = compare_decode(args.golden, args.c)
    if args.json:
        text = json.dumps(report, indent=2)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text)
            print(f"wrote {args.output}")
        else:
            print(text)
    else:
        print_decode_report(report)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2))
            print(f"wrote {args.output}")
    aggregate = report["aggregate"]
    passed = bool(
        aggregate["exact_traj_match"]
        and aggregate["logits_parity_pass"] is not False
    )
    return 0 if passed else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--decode",
        action="store_true",
        help="accepted for command compatibility; decode is the only mode",
    )
    parser.add_argument(
        "--golden", type=Path, required=True, help="golden decode JSON"
    )
    parser.add_argument(
        "--c", type=Path, required=True, help="native/HW decode JSON to check"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the machine-readable JSON report instead of the text report",
    )
    args = parser.parse_args()

    sys.exit(run_decode(args))


if __name__ == "__main__":
    main()
