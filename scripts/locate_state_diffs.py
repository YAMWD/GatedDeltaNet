#!/usr/bin/env python3
"""Decode differing lanes of two GDNSDMP1 state dumps into model coordinates.

The scatter layout (gdn_scatter_recurrent_state) is, per state row:
    beat_in_port = row * 2 + pair
    global_v     = (port >> 1) * 128 + pair * 64 + subhalf * 32
                   + (port & 1) * 16 + lane
    row          = layer * (GDN_HEADS * GDN_DK) + head * GDN_DK + k

Clustering tells us where the upstream perturbation lives: by head or layer
=> a per-head scalar (decay/beta); by v-column => delta; by k-row => k_j;
uniform => a per-element effect.
"""

import argparse
import struct
import sys
from collections import Counter

BEAT = 64
HEADS = 8
DK = 256


def read_dump(path):
    with open(path, "rb") as handle:
        if handle.read(8) != b"GDNSDMP1":
            raise SystemExit(f"{path}: bad magic")
        stripe_beats, ports, conv_beats, _ = struct.unpack("<4I", handle.read(16))
        stripes = [handle.read(stripe_beats * BEAT) for _ in range(ports)]
        conv = handle.read(conv_beats * BEAT)
    return stripe_beats, ports, stripes, conv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dump_a")
    parser.add_argument("dump_b")
    parser.add_argument("--first-port", type=int, default=28)
    args = parser.parse_args()

    beats_a, ports, sa, _ = read_dump(args.dump_a)
    beats_b, ports_b, sb, _ = read_dump(args.dump_b)
    if (beats_a, ports) != (beats_b, ports_b):
        raise SystemExit("dump geometries differ")

    coords = []
    for p in range(ports):
        port = args.first_port + p
        for beat in range(beats_a):
            base = beat * BEAT
            wa = sa[p][base:base + BEAT]
            wb = sb[p][base:base + BEAT]
            if wa == wb:
                continue
            for lane16 in range(32):
                va = wa[lane16 * 2:lane16 * 2 + 2]
                vb = wb[lane16 * 2:lane16 * 2 + 2]
                if va == vb:
                    continue
                row, pair = divmod(beat, 2)
                subhalf, lane = divmod(lane16, 16)
                v = ((port >> 1) * 128 + pair * 64 + subhalf * 32
                     + (port & 1) * 16 + lane)
                layer, rem = divmod(row, HEADS * DK)
                head, k = divmod(rem, DK)
                ia = int.from_bytes(va, "little")
                ib = int.from_bytes(vb, "little")
                coords.append((layer, head, k, v, ia, ib))

    if not coords:
        print("no differing lanes")
        return 0

    print(f"differing lanes: {len(coords)}")
    ulp = Counter(abs(a - b) for *_, a, b in coords)
    print(f"|delta| in BF16 ULP: {dict(sorted(ulp.items()))}")

    for label, index in (("layer", 0), ("head", 1)):
        hist = Counter(c[index] for c in coords)
        print(f"\nby {label} ({len(hist)} distinct of "
              f"{24 if index == 0 else HEADS}):")
        print("  " + " ".join(f"{key}:{count}"
                              for key, count in sorted(hist.items())))

    for label, index, size in (("k row", 2, DK), ("v col", 3, DK)):
        hist = Counter(c[index] for c in coords)
        repeats = sum(1 for count in hist.values() if count > 1)
        print(f"\nby {label}: {len(hist)} distinct of {size}, "
              f"{repeats} value(s) hit more than once "
              f"(uniform-random expectation ~"
              f"{len(coords) * (len(coords) - 1) / (2 * size):.1f})")

    print("\nfirst 8 lanes (layer, head, k, v, a_bits, b_bits):")
    for c in coords[:8]:
        print(f"  layer={c[0]:2d} head={c[1]} k={c[2]:3d} v={c[3]:3d} "
              f"0x{c[4]:04x} vs 0x{c[5]:04x}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
