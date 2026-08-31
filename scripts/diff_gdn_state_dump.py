#!/usr/bin/env python3
"""Diff two GDNSDMP1 persistent-state dumps (gdn_eval / host.exe --dump-state).

Reports, per region (four recurrent-state port stripes + the conv-tail
region), the number of differing 64-byte beats, the first differing beats
with hex context, and simple pattern classification (head/tail clustering,
even spread) to localize the on-card state round-trip defect.
"""

import argparse
import struct
import sys

BEAT = 64


def read_dump(path):
    with open(path, "rb") as handle:
        magic = handle.read(8)
        if magic != b"GDNSDMP1":
            raise SystemExit(f"{path}: bad magic {magic!r}")
        stripe_beats, port_count, conv_beats, _ = struct.unpack(
            "<4I", handle.read(16))
        regions = {}
        for port in range(port_count):
            regions[f"port{28 + port}"] = handle.read(stripe_beats * BEAT)
        regions["conv"] = handle.read(conv_beats * BEAT)
        tail = handle.read()
        if tail:
            raise SystemExit(f"{path}: {len(tail)} trailing bytes")
    return regions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dump_a")
    parser.add_argument("dump_b")
    parser.add_argument("--show", type=int, default=4,
                        help="differing beats to print per region")
    args = parser.parse_args()

    a = read_dump(args.dump_a)
    b = read_dump(args.dump_b)
    if a.keys() != b.keys():
        raise SystemExit("region sets differ between dumps")

    total_diff = 0
    for name in a:
        ra, rb = a[name], b[name]
        if len(ra) != len(rb):
            raise SystemExit(f"{name}: region sizes differ")
        beats = len(ra) // BEAT
        diffs = [i for i in range(beats)
                 if ra[i * BEAT:(i + 1) * BEAT] != rb[i * BEAT:(i + 1) * BEAT]]
        total_diff += len(diffs)
        if not diffs:
            print(f"{name:7s} beats={beats:7d} differing=0  IDENTICAL")
            continue
        first, last = diffs[0], diffs[-1]
        spread = (last - first + 1)
        density = len(diffs) / spread if spread else 1.0
        print(f"{name:7s} beats={beats:7d} differing={len(diffs):7d} "
              f"first={first} last={last} span={spread} density={density:.3f}")
        for i in diffs[:args.show]:
            wa = ra[i * BEAT:(i + 1) * BEAT]
            wb = rb[i * BEAT:(i + 1) * BEAT]
            lanes = [f"lane{j}:{wa[j*4:j*4+4].hex()}!={wb[j*4:j*4+4].hex()}"
                     for j in range(16)
                     if wa[j * 4:j * 4 + 4] != wb[j * 4:j * 4 + 4]][:4]
            print(f"    beat {i}: {' '.join(lanes)}")

    print(f"TOTAL differing beats: {total_diff}")
    return 0 if total_diff == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
