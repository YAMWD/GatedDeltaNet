#!/usr/bin/env bash
# Iter84: build the SLR2-region cosim sources from the CURRENT c_impl source into <work dir>.
# The only difference from the tree is GDN_COMPILED_WEIGHT_SHARD_BEATS (1,366,528 -> 16,384:
# one QKVG call's weights, the layer-0 state stripe and the probe tail fit a 20,992-beat port),
# asserted by a diff. Then: cd <work dir> && vitis_hls -f cosim.tcl (Vitis 2024.2).
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); c_impl=$(cd "$here/.." && pwd)
out="${1:?usage: make_cosim.sh <work dir>}"; mkdir -p "$out"
sed -e 's/#define GDN_COMPILED_WEIGHT_SHARD_BEATS 1366528u/#define GDN_COMPILED_WEIGHT_SHARD_BEATS 16384u/' "$c_impl/gdn_model.h" > "$out/gdn_model.h"
cp "$c_impl/gdn_model.cpp" "$out/gdn_model.cpp"
cp "$c_impl/hls_gdn_forward.tcl" "$here/slr2_region_cosim_tb.cpp" "$here/cosim.tcl" "$out/"
d=$(diff "$c_impl/gdn_model.h" "$out/gdn_model.h" | command grep -c '^[<>]' || true)
[ "$d" = 2 ] || { echo "header diff is not exactly the shard constant ($d lines)"; exit 2; }
cmp -s "$c_impl/gdn_model.cpp" "$out/gdn_model.cpp" || { echo "cpp copy differs"; exit 2; }
sha256sum "$c_impl/gdn_model.cpp" "$c_impl/gdn_model.h" > "$out/source.sha256"
echo "cosim sources ready in $out ($(cut -c1-16 "$out/source.sha256" | tr '\n' ' '))"
