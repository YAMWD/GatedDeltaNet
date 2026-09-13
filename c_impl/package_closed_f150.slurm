#!/bin/bash
#SBATCH --job-name=package_closed_f150
#SBATCH --partition=build
#SBATCH --constraint=vivado2024.2
#SBATCH --exclude=acclnode04,acclnode05,harrier
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --chdir=/home/yaoz0b/GatedDeltaNet/c_impl
#SBATCH --output=/home/yaoz0b/GatedDeltaNet/c_impl/diagnostics/package_closed_f150/package-%j.log
# Package a timing-closed checkpoint into a loadable 150 MHz image.
#   inputs (--export): PK_DCP  = the closed checkpoint (required)
#                      PK_DONOR = an XCLBIN built from the SAME kernel source,
#                                 whose metadata is reused (required)
#   Slurm opens --output before the script runs, so create the log directory
#   first:
#     mkdir -p c_impl/diagnostics/package_closed_f150 && \
#     sbatch --export=ALL,PK_DCP=/abs/closed.dcp,PK_DONOR=/abs/donor.xclbin c_impl/package_closed_f150.slurm
set -o pipefail
C=/home/yaoz0b/GatedDeltaNet/c_impl
D=$C/diagnostics/package_closed_f150
export PK_DCP="${PK_DCP:?set PK_DCP to the timing-closed checkpoint}"
export PK_OUT="$D/out-$SLURM_JOB_ID"
DONOR="${PK_DONOR:?set PK_DONOR to an XCLBIN built from the same kernel source}"
trap 'rc=$?; printf "%s\n" "$rc" > "$D/package.exit"; echo "PACKAGE_EXIT=$rc"' EXIT
mkdir -p "$PK_OUT"
echo "node=$(hostname) job=$SLURM_JOB_ID start=$(date -u +%FT%TZ)"
df -h /tmp "$D" | sed 's/^/df: /'
test -s "$PK_DCP" || { echo "FATAL: checkpoint missing: $PK_DCP"; exit 2; }
test -s "$DONOR" || { echo "FATAL: donor image missing: $DONOR"; exit 2; }
sha256sum "$PK_DCP" "$DONOR" "$C/package_closed_f150.tcl" | tee "$PK_OUT/input_hashes.txt"
set +u; source /tools/Xilinx/Vitis/2024.2/settings64.sh; source /opt/xilinx/xrt/setup.sh; set -u
vivado -mode batch -nolog -nojournal -source "$C/package_closed_f150.tcl" 2>&1 | tee "$PK_OUT/vivado.log"
vstat=${PIPESTATUS[0]}; echo "vivado exit=$vstat"
[ "$vstat" -eq 0 ] || exit "$vstat"
BIT="$PK_OUT/ulp_partial.bit"; XU=/opt/xilinx/xrt/bin/xclbinutil
test -s "$BIT" || { echo "FATAL: no bitstream"; exit 2; }
echo "=== patch DATA_CLK to 150 in the donor's clock topology"
$XU --input "$DONOR" --dump-section CLOCK_FREQ_TOPOLOGY:JSON:"$PK_OUT/clk.json" --force >/dev/null 2>&1 \
    || { echo "FATAL: cannot read the donor's clock topology"; exit 2; }
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
[ "${PIPESTATUS[0]}" -eq 0 ] && test -s "$PK_OUT/gdn_forward_f150.xclbin" || { echo "FATAL: assemble failed"; exit 2; }
sha256sum "$PK_OUT/gdn_forward_f150.xclbin" | tee "$PK_OUT/gdn_forward_f150.sha256"
echo "=== verify by reading the assembled image's clock topology back"
$XU --input "$PK_OUT/gdn_forward_f150.xclbin" --dump-section CLOCK_FREQ_TOPOLOGY:JSON:"$PK_OUT/clk_check.json" --force >/dev/null 2>&1 \
    || { echo "FATAL: cannot read back the assembled image"; exit 2; }
python3 - "$PK_OUT/clk_check.json" <<'PY' || { echo "FATAL: assembled image does not report DATA_CLK 150"; exit 2; }
import json, sys
d = json.load(open(sys.argv[1]))['clock_freq_topology']['m_clock_freq']
c = [float(e['m_freq_Mhz']) for e in d if e['m_type'] == 'DATA']
assert c == [150.0], f"DATA clocks read back: {c}"
print('EXACT_CLOCK_PASS', c)
PY
$XU --input "$PK_OUT/gdn_forward_f150.xclbin" --info 2>/dev/null | grep -iE 'Kernels|Frequency|UUID \(xclbin\)' | head -6
echo "PACKAGE_OK finish=$(date -u +%FT%TZ)"
