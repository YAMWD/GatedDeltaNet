#!/usr/bin/env python3
"""Windowed drift trend between two GDNLOG1 logit dumps.

A worst-step aggregate is a maximum: it cannot distinguish a bounded
perturbation from one that grows with token index. This reports NRMSE, cosine,
max-abs and argmax agreement per fixed-size window of steps, plus a
least-squares slope over the window series, so "bounded" versus "growing" is
a measurement rather than an impression.
"""

import argparse
import struct
import sys


def read_log(path):
    with open(path, "rb") as handle:
        magic = handle.read(8)
        if magic[:7] != b"GDNLOG1":
            raise SystemExit(f"{path}: bad magic {magic!r}")
        version, vocab, steps = struct.unpack("<3I", handle.read(12))
        payload = handle.read()
    expected = steps * vocab * 4
    if len(payload) < expected:
        steps = len(payload) // (vocab * 4)
        print(f"note: {path} truncated; using {steps} complete steps")
    return vocab, steps, payload


def step_floats(payload, vocab, index):
    start = index * vocab * 4
    return struct.unpack_from(f"<{vocab}f", payload, start)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--window", type=int, default=64)
    args = parser.parse_args()

    vocab_r, steps_r, ref = read_log(args.reference)
    vocab_c, steps_c, cand = read_log(args.candidate)
    if vocab_r != vocab_c:
        raise SystemExit("vocab mismatch")
    steps = min(steps_r, steps_c)
    print(f"comparing {steps} steps x {vocab_r} logits, "
          f"window={args.window}")

    rows = []
    first_argmax_div = -1
    for start in range(0, steps, args.window):
        stop = min(start + args.window, steps)
        err_sq = ref_sq = dot = cand_sq = 0.0
        max_abs = 0.0
        argmax_bad = 0
        exact_bad = 0
        for index in range(start, stop):
            a = step_floats(ref, vocab_r, index)
            b = step_floats(cand, vocab_c, index)
            best_a = best_b = 0
            for v in range(vocab_r):
                av, bv = a[v], b[v]
                d = bv - av
                err_sq += d * d
                ref_sq += av * av
                cand_sq += bv * bv
                dot += av * bv
                if abs(d) > max_abs:
                    max_abs = abs(d)
                if d != 0.0:
                    exact_bad += 1
                if av > a[best_a]:
                    best_a = v
                if bv > b[best_b]:
                    best_b = v
            if best_a != best_b:
                argmax_bad += 1
                if first_argmax_div < 0:
                    first_argmax_div = index
        nrmse = (err_sq / ref_sq) ** 0.5 if ref_sq > 0 else float("inf")
        cosine = dot / ((ref_sq * cand_sq) ** 0.5) if ref_sq and cand_sq else -1.0
        rows.append((start, stop, nrmse, cosine, max_abs, argmax_bad, exact_bad))
        print(f"  steps {start:4d}-{stop-1:4d}  nrmse={nrmse:.9f}  "
              f"cosine={cosine:.12f}  max_abs={max_abs:.6g}  "
              f"argmax_mismatch={argmax_bad}  differing_logits={exact_bad}")

    # Trend is only meaningful BEFORE the trajectories fork: once free-running
    # decode picks a different token, later steps compare unrelated sequences
    # and NRMSE saturates at an "unrelated" level that must not be read as
    # drift growth. Restrict the trend to windows entirely before the fork.
    if first_argmax_div >= 0:
        usable = [r for r in rows if r[1] <= first_argmax_div]
        print(f"\ntrajectories fork at step {first_argmax_div}; "
              f"{len(usable)} of {len(rows)} windows precede it and are "
              f"comparable (later windows compare different sequences)")
        rows = usable

    if len(rows) >= 2:
        xs = [r[0] for r in rows]
        ys = [r[2] for r in rows]
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        var = sum((x - mx) ** 2 for x in xs)
        slope = (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var
                 if var else 0.0)
        first, last = ys[0], ys[-1]
        ratio = (last / first) if first > 0 else float("inf")
        print(f"\nNRMSE first window {first:.9f} -> last window {last:.9f} "
              f"(ratio {ratio:.3f})")
        print(f"least-squares slope per step: {slope:.3e}")
        verdict = "GROWING" if (ratio > 1.5 and slope > 0) else "BOUNDED"
        print(f"TREND_VERDICT={verdict}")
    print(f"FIRST_ARGMAX_DIVERGENCE={first_argmax_div}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
