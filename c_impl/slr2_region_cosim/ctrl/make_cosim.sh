#!/usr/bin/env bash
# Iter86: build the control-region cosim sources (unchanged tree copies) into <work dir>.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); c_impl=$(cd "$here/../.." && pwd)
out="${1:?usage: make_cosim.sh <work dir>}"; mkdir -p "$out"
cp "$c_impl/gdn_model.cpp" "$c_impl/gdn_model.h" "$c_impl/hls_gdn_forward.tcl" "$here/ctrl_region_cosim_tb.cpp" "$here/cosim.tcl" "$out/"
sha256sum "$c_impl/gdn_model.cpp" "$c_impl/gdn_model.h" > "$out/source.sha256"
echo "ctrl cosim sources ready in $out ($(cut -c1-16 "$out/source.sha256" | tr '\n' ' '))"
