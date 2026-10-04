#!/usr/bin/env python3
"""Fail closed on the Iter66 integrated-HLS gates before Vivado linking.

The build tree can contain several copies of the HLS reports.  Select the
complete solution report directory (the one with the largest csynth report
set), then validate architecture, II, timing, arithmetic structure, and the
pre-registered local-resource comparison.  This intentionally has no global
LUT-percentage gate; physical feasibility is judged later from the routed
candidate.

Iter79 (2026-10-04): the production image is three kernels compiled from one
source -- gdn_forward_p (control, ports 0/1/18-23), gdn_k_slr0 (ports 2-17) and
gdn_k_slr2 (ports 24-31, recurrence, state writers).  The gate selects one report
directory per kernel and applies the structural checks to their union: 32 masters
across the three tops, one GEMV region instance per kernel, 16 clusters, 27
ordinary readers, four state owners.  A build tree that holds only the retired
monolith (gdn_forward) is still accepted, for A/B builds.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


REFERENCE = {
    "four_dots_lut": 18_356,
    "four_dots_ff": 27_557,
    "cluster_lut": 32_304,
    "cluster_ff": 45_716,
    # Iter66e re-baseline: style=frp on gemv32_cl_weight_stream costs +151 LUT
    # per cluster weight loop (probe job 2485: 27,275 -> 27,426, II=1 kept,
    # yes(frp) on all 16 loops) in exchange for eliminating the 5.1K-9.3K-load
    # clock-enable cones present in every Iter66b-66d routing conflict set.
    # Bound set from the measured probe BEFORE any Iter66e hardware result.
    "cluster_weight_lut": 27_426,
    "cluster_weight_ff": 40_322,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("build_dir", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--target-mhz", type=float, default=200.0,
                        help="HLS clock target the kernels must have been synthesized for")
    return parser.parse_args()


PARTITION_TOPS = ("gdn_forward_p", "gdn_k_slr0", "gdn_k_slr2")
REGION_MODULE = {
    "gdn_forward_p": "gdn_gemv_part_ctrl",
    "gdn_k_slr0": "gdn_gemv_part_slr0",
    "gdn_k_slr2": "gdn_gemv_part_slr2",
    "gdn_forward": "gdn_gemv",
}


def xml_text(root: ET.Element, path: str) -> str:
    value = root.findtext(path)
    if value is None:
        raise ValueError(f"missing XML field {path}")
    return value.strip()


def resources(path: Path) -> Dict[str, int]:
    root = ET.parse(path).getroot()
    area = root.find("AreaEstimates/Resources")
    if area is None:
        raise ValueError(f"missing resource summary in {path}")
    result: Dict[str, int] = {}
    for name in ("BRAM_18K", "DSP", "FF", "LUT", "URAM"):
        value = area.findtext(name)
        if value is None:
            raise ValueError(f"missing {name} in {path}")
        result[name] = int(value)
    return result


def select_report_dir_for(build_dir: Path, top: str) -> Path:
    candidates: List[Tuple[int, int, Path]] = []
    for xml in build_dir.rglob(f"{top}_csynth.xml"):
        parent = xml.parent
        count = len(list(parent.glob("*_csynth.xml")))
        candidates.append((count, -len(str(parent)), parent))
    if not candidates:
        raise ValueError(f"no {top}_csynth.xml below {build_dir}")
    return max(candidates)[2]


def select_report_dirs(build_dir: Path) -> Dict[str, Path]:
    """One complete report directory per kernel of the three-kernel image, or the
    retired monolith's single directory when that is all the tree holds."""
    found: Dict[str, Path] = {}
    for top in PARTITION_TOPS:
        try:
            found[top] = select_report_dir_for(build_dir, top)
        except ValueError:
            pass
    if len(found) == len(PARTITION_TOPS):
        return found
    if found:
        raise ValueError(
            f"incomplete three-kernel build: found reports for {sorted(found)} only"
        )
    return {"gdn_forward": select_report_dir_for(build_dir, "gdn_forward")}


def reports(report_dirs: Iterable[Path], pattern: str) -> List[Path]:
    """Every report whose file name matches, across all selected directories.
    Instance-numbered names repeat between kernels (each kernel has its own
    gemv32_cluster2_1); those are distinct modules and are all counted."""
    regex = re.compile(pattern)
    result: List[Path] = []
    for report_dir in report_dirs:
        result.extend(
            path for path in report_dir.glob("*_csynth.xml") if regex.fullmatch(path.name)
        )
    return sorted(result, key=lambda path: (path.name, str(path)))


def pipeline_iis(path: Path) -> List[int]:
    root = ET.parse(path).getroot()
    return [int(node.text) for node in root.findall(".//PipelineII") if node.text]


def scalar_trip_counts(path: Path) -> List[int]:
    root = ET.parse(path).getroot()
    values: List[int] = []
    for node in root.findall(".//TripCount"):
        if node.text and node.text.strip():
            values.append(int(node.text.strip()))
    return values


def main() -> int:
    args = parse_args()
    failures: List[str] = []
    notes: List[str] = []

    try:
        report_dirs = select_report_dirs(args.build_dir.resolve())
    except (OSError, ValueError) as error:
        print(f"XO_GATE_FAIL: {error}", file=sys.stderr)
        return 2
    tops = sorted(report_dirs)
    dirs = [report_dirs[top] for top in tops]
    partitioned = set(tops) == set(PARTITION_TOPS)

    four_xml = None
    for report_dir in dirs:
        candidate = report_dir / "gemv32_four_dots_csynth.xml"
        if candidate.is_file():
            four_xml = candidate
            break
    if four_xml is None:
        # Iter78 step 1b: gemv32_four_dots is inlined into the free-running
        # weight loop, so its II, multiplier and resource checks read that
        # loop's report instead (the unsuffixed cluster's; all sixteen are
        # the same module body).
        loop_reports = reports(
            dirs, r"gemv32_cluster2_Pipeline_gemv32_cl_weight_stream_csynth\.xml")
        if not loop_reports:
            loop_reports = reports(
                dirs, r"gemv32_cluster2_\d+_Pipeline_gemv32_cl_weight_stream_csynth\.xml")
        if loop_reports:
            four_xml = loop_reports[0]
            notes.append(
                "gemv32_four_dots is inlined into the cluster weight loop "
                f"(Iter78 step 1b); its checks were applied to {four_xml.name}"
            )
    if four_xml is None:
        failures.append("no gemv32_four_dots or cluster weight-loop report found")
        four_xml = dirs[0] / "gemv32_four_dots_csynth.xml"
    four_rpt = four_xml.with_suffix(".rpt")
    required_files = [four_xml, four_rpt]
    for top in tops:
        required_files += [report_dirs[top] / f"{top}_csynth.xml",
                           report_dirs[top] / f"{top}_csynth.rpt"]
    for path in required_files:
        if not path.is_file():
            failures.append(f"required report missing: {path}")

    summary: Dict[str, object] = {
        "build_dir": str(args.build_dir.resolve()),
        "tops": tops,
        "partitioned": partitioned,
        "report_dirs": {top: str(path) for top, path in report_dirs.items()},
        "reference": REFERENCE,
    }
    if failures:
        summary["failures"] = failures
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 2

    target_limit_ns = 1000.0 / args.target_mhz + 0.01
    per_top: Dict[str, Dict[str, object]] = {}
    masters_all: set = set()
    region_instances: Dict[str, List[str]] = {}
    for top in tops:
        top_xml = report_dirs[top] / f"{top}_csynth.xml"
        top_rpt = report_dirs[top] / f"{top}_csynth.rpt"
        top_root = ET.parse(top_xml).getroot()
        version = xml_text(top_root, "ReportVersion/Version")
        target_ns = float(xml_text(top_root, "UserAssignments/TargetClockPeriod"))
        estimated_ns = float(
            xml_text(
                top_root,
                "PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod",
            )
        )
        latency = top_root.find("PerformanceEstimates/SummaryOfOverallLatency")
        if latency is None:
            raise ValueError(f"missing top-level latency summary in {top_xml}")
        # A free-running (ap_ctrl_none) blackbox in the dataflow graph gives the
        # top an unbounded latency model and HLS reports 'undef' (Iter77 S2);
        # record it as null -- the architectural checks below do not depend on it.
        def latency_cycles(tag: str):
            text = xml_text(latency, tag)
            return None if text in ("undef", "?", "") else int(text)
        per_top[top] = {
            "hls_version": version,
            "target_ns": target_ns,
            "estimated_ns": estimated_ns,
            "estimated_fmax_mhz": 1000.0 / estimated_ns,
            "resources": resources(top_xml),
            "latency": {
                "best_cycles": latency_cycles("Best-caseLatency"),
                "average_cycles": latency_cycles("Average-caseLatency"),
                "worst_cycles": latency_cycles("Worst-caseLatency"),
            },
        }
        if not version.startswith("2024.2"):
            failures.append(f"{top}: HLS version is {version}, expected 2024.2")
        if target_ns > target_limit_ns:
            failures.append(
                f"{top}: HLS target is {target_ns:.3f} ns, not {args.target_mhz:g} MHz")
        if estimated_ns > target_ns:
            failures.append(
                f"{top}: estimated clock {estimated_ns:.3f} ns misses {target_ns:.3f} ns target"
            )
        masters_all |= {int(port) for port in re.findall(
            r"m_axi_mem_weights_mm(\d+)_", top_xml.read_text(errors="replace"))}
        module = REGION_MODULE[top]
        instances = set()
        for line in top_rpt.read_text(errors="replace").splitlines():
            match = re.search(
                r"\|\s*(grp_" + re.escape(module) + r"_fu_[^| ]+)\s*\|\s*" + re.escape(module) + r"\s*\|",
                line,
            )
            if match:
                instances.add(match.group(1))
        region_instances[top] = sorted(instances)
        if len(instances) != 1:
            failures.append(
                f"{top}: found {len(instances)} shared {module} instances, expected 1"
            )
    summary["per_top"] = per_top
    masters = sorted(masters_all)
    summary["weight_master_indices"] = masters
    if masters != list(range(32)):
        failures.append(f"weight AXI masters are {masters}, expected exactly 0..31")
    summary["shared_gemv_region_instances"] = region_instances

    clusters = reports(dirs, r"gemv32_cluster2(?:_\d+)?_csynth\.xml")
    cluster_weight = reports(
        dirs,
        r"gemv32_cluster2(?:_\d+)?_Pipeline_gemv32_cl_weight_stream_csynth\.xml",
    )
    summary["cluster_report_count"] = len(clusters)
    summary["cluster_weight_loop_count"] = len(cluster_weight)
    if len(clusters) != 16:
        failures.append(f"found {len(clusters)} clusters, expected 16")
    if len(cluster_weight) != 16:
        failures.append(
            f"found {len(cluster_weight)} cluster weight loops, expected 16"
        )

    bad_cluster_iis: Dict[str, List[int]] = {}
    for path in cluster_weight:
        iis = pipeline_iis(path)
        if not iis or any(ii != 1 for ii in iis):
            bad_cluster_iis[path.name] = iis
    summary["bad_cluster_iis"] = bad_cluster_iis
    if bad_cluster_iis:
        failures.append(f"cluster weight-loop II failures: {bad_cluster_iis}")

    # Port 0 is combined with the activation loader, ports 1..27 use the
    # ordinary reader, and 28..31 use the full-window state-owning reader.
    # Vitis HLS 2022.2 emits a separate report for each ordinary MM2S loop;
    # 2024.2 folds that loop into the templated function's `_s` report.  Select
    # one representation, then apply the same PipelineII=1 gate.  Do not count
    # both if a future release happens to emit both forms.
    ordinary_top = reports(
        dirs, r"gemv32_mm2s_\d+_s_csynth\.xml"
    )
    ordinary_pipeline = reports(
        dirs,
        r"gemv32_mm2s_\d+_Pipeline_gemv32_mm2s_loop_csynth\.xml",
    )
    # Iter78 step 4: with style=frp the ordinary loop is outlined into its own
    # `_Pipeline_gemv32_mm2s_loop` module and the `_s` report carries no II,
    # so prefer the loop reports whenever all 27 exist.
    ordinary_reports = (
        ordinary_pipeline if len(ordinary_pipeline) == 27 else ordinary_top
    )
    # Port 0's combined x/w0 loader: the monolith's gemv32_load_x_and_w0, or the
    # three-kernel control kernel's gemv32_load_x_bcast (x to the local chain
    # and the two remote kernels; same two pipelined loops).
    port0_reports = reports(
        dirs,
        r"gemv32_load_x_(?:and_w0_Pipeline_gemv32_(?:lx|w0)|bcast_Pipeline_gemv32_(?:lx|w0)_bcast)_csynth\.xml",
    )
    state_owner_reports = reports(
        dirs,
        r"gemv32_mm2s_with_state_\d+_Pipeline_gemv32_state_owner_"
        r"(?:weight|weight_only|prefetch)_csynth\.xml",
    )
    reader_reports = port0_reports + ordinary_reports + state_owner_reports
    bad_reader_iis: Dict[str, List[int]] = {}
    for path in reader_reports:
        iis = pipeline_iis(path)
        if not iis or any(ii != 1 for ii in iis):
            bad_reader_iis[path.name] = iis
    ordinary_count = len(ordinary_reports)
    owner_weight_count = sum(
        path.name.endswith("state_owner_weight_csynth.xml") for path in reader_reports
    )
    owner_weight_only_count = sum(
        path.name.endswith("state_owner_weight_only_csynth.xml")
        for path in reader_reports
    )
    owner_prefetch_count = sum(
        path.name.endswith("state_owner_prefetch_csynth.xml") for path in reader_reports
    )
    port0_count = 1 if len(port0_reports) == 2 else len(port0_reports)
    reader_counts = {
        "port0_combined": port0_count,
        "ordinary_ports_1_27": ordinary_count,
        "state_owner_weight": owner_weight_count,
        "state_owner_weight_only": owner_weight_only_count,
        "state_owner_prefetch": owner_prefetch_count,
    }
    summary["reader_loop_counts"] = reader_counts
    summary["bad_reader_iis"] = bad_reader_iis
    expected_reader_counts = {
        "port0_combined": 1,
        "ordinary_ports_1_27": 27,
        "state_owner_weight": 4,
        "state_owner_weight_only": 4,
        "state_owner_prefetch": 4,
    }
    if reader_counts != expected_reader_counts:
        failures.append(
            f"reader loop structure is {reader_counts}, expected {expected_reader_counts}"
        )
    if bad_reader_iis:
        failures.append(f"MM2S/state reader II failures: {bad_reader_iis}")

    prefetch_trips = {
        path.name: scalar_trip_counts(path)
        for path in state_owner_reports
        if path.name.endswith("state_owner_prefetch_csynth.xml")
    }
    qkvg_weight_trips = {
        path.name: scalar_trip_counts(path)
        for path in state_owner_reports
        if path.name.endswith("state_owner_weight_csynth.xml")
    }
    summary["state_prefetch_trip_counts"] = prefetch_trips
    summary["qkvg_weight_trip_counts"] = qkvg_weight_trips
    if len(prefetch_trips) != 4 or any(value != [512] for value in prefetch_trips.values()):
        failures.append(
            f"state prefetch is not exactly 512 beats/head/port: {prefetch_trips}"
        )
    if len(qkvg_weight_trips) != 4 or any(
        value != [2048] for value in qkvg_weight_trips.values()
    ):
        failures.append(
            f"QKVG state-owner weight phase is not exactly 2048 beats: {qkvg_weight_trips}"
        )

    gemv_rpt = (
        report_dirs["gdn_k_slr2"] / "gdn_gemv_part_slr2_csynth.rpt"
        if partitioned else report_dirs["gdn_forward"] / "gdn_gemv_csynth.rpt"
    )
    state_fifo_layout: Dict[str, Dict[str, int]] = {}
    if gemv_rpt.is_file():
        for line in gemv_rpt.read_text(errors="replace").splitlines():
            if not re.search(r"\|\s*state_stream[0-3]_U\s*\|", line):
                continue
            fields = [field.strip() for field in line.split("|") if field.strip()]
            if len(fields) >= 8:
                state_fifo_layout[fields[0]] = {
                    "depth": int(fields[5]),
                    "bits": int(fields[6]),
                }
    summary["state_fifo_layout"] = state_fifo_layout
    expected_fifos = {f"state_stream{index}_U" for index in range(4)}
    if set(state_fifo_layout) != expected_fifos or any(
        value != {"depth": 4096, "bits": 512}
        for value in state_fifo_layout.values()
    ):
        failures.append(
            "full-window state FIFO layout is not four 4096x512 queues: "
            f"{state_fifo_layout}"
        )

    four_root = ET.parse(four_xml).getroot()
    four_iis = [
        int(node.text)
        for node in four_root.findall(".//PipelineInitiationInterval")
        if node.text
    ]
    if not four_iis:
        four_iis = [
            int(node.text)
            for node in four_root.findall(".//PipelineII")
            if node.text
        ]
    summary["four_dots_iis"] = four_iis
    if not four_iis or any(ii != 1 for ii in four_iis):
        failures.append(f"gemv32_four_dots is not II=1: {four_iis}")

    four_text = four_rpt.read_text(errors="replace")
    native_instances = set(
        re.findall(r"\b(floatingpoint_mul_16\w*_U\d+)\b", four_text)
    )
    fp32_mul_instances = set(re.findall(r"\b(fmul_32\w*_U\d+)\b", four_text))
    summary["native_bf16_multiplier_instances"] = len(native_instances)
    summary["fp32_multiplier_instances"] = len(fp32_mul_instances)
    if len(native_instances) != 64:
        failures.append(
            f"found {len(native_instances)} native BF16 multipliers, expected 64"
        )
    if fp32_mul_instances:
        failures.append(
            f"found {len(fp32_mul_instances)} FP32 multipliers in gemv32_four_dots"
        )

    four_resources = resources(four_xml)
    cluster_values = [resources(path) for path in clusters]
    weight_values = [resources(path) for path in cluster_weight]

    def maxima(values: Iterable[Dict[str, int]]) -> Dict[str, int]:
        values = list(values)
        if not values:
            return {name: -1 for name in ("BRAM_18K", "DSP", "FF", "LUT", "URAM")}
        return {name: max(value[name] for value in values) for name in values[0]}

    cluster_max = maxima(cluster_values)
    weight_max = maxima(weight_values)
    summary["four_dots_resources"] = four_resources
    summary["cluster_resource_max"] = cluster_max
    summary["cluster_weight_resource_max"] = weight_max

    if cluster_max["LUT"] > REFERENCE["cluster_lut"]:
        failures.append(
            f"cluster LUT grew {REFERENCE['cluster_lut']} -> {cluster_max['LUT']}"
        )
    if cluster_max["FF"] > REFERENCE["cluster_ff"]:
        failures.append(
            f"cluster FF grew {REFERENCE['cluster_ff']} -> {cluster_max['FF']}"
        )
    if weight_max["LUT"] > REFERENCE["cluster_weight_lut"]:
        failures.append(
            "cluster weight-loop LUT grew "
            f"{REFERENCE['cluster_weight_lut']} -> {weight_max['LUT']}"
        )
    if weight_max["FF"] > REFERENCE["cluster_weight_ff"]:
        failures.append(
            "cluster weight-loop FF grew "
            f"{REFERENCE['cluster_weight_ff']} -> {weight_max['FF']}"
        )

    improvements = {
        "four_dots_lut": four_resources["LUT"]
        <= int(REFERENCE["four_dots_lut"] * 0.90),
        "cluster_ff": cluster_max["FF"] <= int(REFERENCE["cluster_ff"] * 0.90),
    }
    summary["ten_percent_local_improvements"] = improvements
    notes.append(
        "ap_ce load count is a post-synthesis physical metric; it is recorded "
        "after placement and is not used to rescue a failed XO resource gate"
    )
    if not any(improvements.values()):
        failures.append(
            "neither gemv32_four_dots LUT nor cluster FF improved by at least 10%"
        )

    summary["notes"] = notes
    summary["failures"] = failures
    summary["verdict"] = "PASS" if not failures else "FAIL"
    payload = json.dumps(summary, indent=2, sort_keys=True)
    print(payload)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload + "\n")
    if failures:
        print(f"XO_GATE_FAIL: {len(failures)} gate(s) failed", file=sys.stderr)
        return 1
    print("XO_GATE_PASS: integrated native-BF16 HLS candidate is authorized for link")
    return 0


if __name__ == "__main__":
    sys.exit(main())
