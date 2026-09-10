#!/bin/bash
#SBATCH --job-name=iter75f_package
#SBATCH --partition=build
#SBATCH --constraint=vivado2024.2
#SBATCH --exclude=acclnode04,acclnode05,harrier
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --chdir=/home/yaoz0b/GatedDeltaNet/c_impl/diagnostics/iter75f_package
#SBATCH --output=/home/yaoz0b/GatedDeltaNet/c_impl/diagnostics/iter75f_package/package-%j.log
set -o pipefail
D=/home/yaoz0b/GatedDeltaNet/c_impl/diagnostics/iter75f_package
C=/home/yaoz0b/GatedDeltaNet/c_impl
export PK_DCP="$C/diagnostics/iter75f_ce_locality/reports-3955/closed_f150.dcp"
export PK_OUT="$D/out-$SLURM_JOB_ID"
DONOR="$C/diagnostics/iter75b_recover/gdn_forward_f142.xclbin"   # same source (ca263d7e), 142.5 MHz metadata
trap 'rc=$?; printf "%s\n" "$rc" > "$D/package.exit"; echo "PACKAGE_EXIT=$rc"' EXIT
mkdir -p "$PK_OUT"
echo "node=$(hostname) job=$SLURM_JOB_ID start=$(date -u +%FT%TZ)"
df -h /tmp "$D" | sed 's/^/df: /'
sha256sum "$PK_DCP" "$DONOR" "$D/package_closed_f150.tcl" | tee "$PK_OUT/input_hashes.txt"
set +u; source /tools/Xilinx/Vitis/2024.2/settings64.sh; source /opt/xilinx/xrt/setup.sh; set -u
vivado -mode batch -nolog -nojournal -source "$D/package_closed_f150.tcl" 2>&1 | tee "$PK_OUT/vivado.log"
vstat=${PIPESTATUS[0]}; echo "vivado exit=$vstat"
[ "$vstat" -eq 0 ] || exit "$vstat"
BIT="$PK_OUT/ulp_partial.bit"; XU=/opt/xilinx/xrt/bin/xclbinutil
test -s "$BIT" || { echo "FATAL: no bitstream"; exit 2; }
echo "=== patch DATA_CLK to 150 in the donor's clock topology"
$XU --input "$DONOR" --dump-section CLOCK_FREQ_TOPOLOGY:JSON:"$PK_OUT/clk.json" --force >/dev/null 2>&1
python3 - "$PK_OUT/clk.json" "$PK_OUT/clk150.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); n=0
for e in d["clock_freq_topology"]["m_clock_freq"]:
    if e.get("m_name")=="DATA_CLK": print("DATA_CLK",e["m_freq_Mhz"],"-> 150"); e["m_freq_Mhz"]="150"; n+=1
assert n==1, f"expected one DATA_CLK, found {n}"
json.dump(d,open(sys.argv[2],"w"),indent=4)
PY
$XU --input "$DONOR" --replace-section BITSTREAM:RAW:"$BIT" \
    --replace-section CLOCK_FREQ_TOPOLOGY:JSON:"$PK_OUT/clk150.json" \
    --output "$PK_OUT/gdn_forward_f150.xclbin" --force 2>&1 | tail -3
test -s "$PK_OUT/gdn_forward_f150.xclbin" || { echo "FATAL: assemble failed"; exit 2; }
sha256sum "$PK_OUT/gdn_forward_f150.xclbin" | tee "$PK_OUT/gdn_forward_f150.sha256"
echo "=== verify the assembled image reports 150"
$XU --input "$PK_OUT/gdn_forward_f150.xclbin" --info 2>/dev/null | grep -iE 'Kernels|Content|Frequency|UUID \(xclbin\)' | head -8
echo "PACKAGE_OK finish=$(date -u +%FT%TZ)"
