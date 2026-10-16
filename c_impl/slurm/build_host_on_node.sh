#!/bin/bash
# Build the XRT host of a frozen submission on the node that will run it.
#   usage: build_host_on_node.sh <DIAG dir> <output host.exe> [HOST_DEFINES]
# The three host sources come from the submission's source_snapshot.tar and are
# compiled with its own reproduction.Makefile, so the binary matches the frozen
# source exactly (checked against source_hashes.txt) and links against this
# node's libstdc++ (a host built on acclnode01 needs GLIBCXX_3.4.30, which
# harrier lacks: card job 7778, Iter84).  Iter88.
set -euo pipefail
DIAG="${1:?DIAG dir}"; OUT="${2:?output host path}"; DEFINES="${3:-}"
SRC="$DIAG/host_src"
mkdir -p "$SRC"
tar -xf "$DIAG/source_snapshot.tar" -C "$SRC" host.cpp gdn_model.cpp gdn_model.h
for f in host.cpp gdn_model.cpp gdn_model.h; do
    want="$(grep -E " $f\$" "$DIAG/source_hashes.txt" | cut -d' ' -f1)"
    have="$(sha256sum "$SRC/$f" | cut -d' ' -f1)"
    [ -n "$want" ] && [ "$want" = "$have" ] || { echo "FATAL: $f in the snapshot does not match source_hashes.txt" >&2; exit 2; }
done
rm -f "$OUT"
make -s -C "$SRC" -f "$DIAG/reproduction.Makefile" host HOST_BIN="$OUT" HOST_DEFINES="$DEFINES"
test -x "$OUT"
echo "host built on $(hostname -s) from the frozen snapshot: $(sha256sum "$OUT" | cut -c1-16)  defines='${DEFINES}'"
