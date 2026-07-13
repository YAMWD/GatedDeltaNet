# Physical SLR assignment for the mono-kernel eight-cluster GEMV.
#
# The first 3/3/2 floorplan forced every AXI adapter beside its compute cluster.
# That put gmem30 in SLR2 while its HMSS endpoint remained in SLR0, producing a
# direct 512-bit SLR0->SLR2 path and 98.9% routing-dominated critical delay.
# Leave adapters unconstrained so the platform can retain their natural SLR0/1
# placement. MM2S loaders form the registered boundary: loaders for SLR2 compute
# remain in SLR1 and feed BRAM streams placed with the destination clusters.

proc required_cell {pattern} {
    set cells [get_cells -hierarchical -quiet $pattern]
    if {[llength $cells] == 0} {
        error "pblock_gemv_full: required pattern matched 0 cells: $pattern"
    }
    return $cells
}

foreach slr {SLR0 SLR1 SLR2} {
    create_pblock pb_gemv_full_$slr
    resize_pblock pb_gemv_full_$slr -add $slr
    set_property CONTAIN_ROUTING 0 [get_pblocks pb_gemv_full_$slr]
}

proc assign_cells_to_slr {slr cells} {
    if {[llength $cells] > 0} {
        set_property USER_SLR_ASSIGNMENT $slr $cells
        add_cells_to_pblock pb_gemv_full_$slr $cells -quiet
    }
}

proc loader_cells_for_port {port} {
    if {$port == 0} {
        return [required_cell "*loader0_sys_U0"]
    }
    if {$port == 1} {
        return [required_cell "*mm2s_loader_U0"]
    }
    set suffix [expr {$port - 1}]
    return [required_cell "*mm2s_loader_${suffix}_U0"]
}

set cluster_names {
    cluster4_sys_U0
    cluster4_sys_31_U0
    cluster4_sys_32_U0
    cluster4_sys_33_U0
    cluster4_sys_34_U0
    cluster4_sys_35_U0
    cluster4_sys_36_U0
    cluster4_sys_37_U0
}

# Preserve activation-ripple order while balancing compute against the platform:
# two clusters in SLR0, three in SLR1, and three in SLR2. The failed route used
# 92%/86%/59% of CLBs in SLR0/1/2; this moves one complete cluster to SLR2.
set cluster_slrs {SLR0 SLR0 SLR1 SLR1 SLR1 SLR2 SLR2 SLR2}

for {set c 0} {$c < 8} {incr c} {
    set slr [lindex $cluster_slrs $c]
    set loader_slr [expr {$c < 2 ? "SLR0" : "SLR1"}]
    set cluster [lindex $cluster_names $c]
    assign_cells_to_slr $slr [required_cell "*${cluster}"]
    assign_cells_to_slr $slr [required_cell "*xr_${c}_U"]
    assign_cells_to_slr $slr [required_cell "*ys_${c}_U"]

    for {set local 0} {$local < 4} {incr local} {
        set port [expr {$c * 4 + $local}]
        assign_cells_to_slr $loader_slr [loader_cells_for_port $port]
        assign_cells_to_slr $slr [required_cell "*ws_${port}_U"]
    }
    puts "pblock_gemv_full: cluster $c -> $slr, loaders -> $loader_slr"
}

# Merge each SLR locally. The final collector sees three BRAM-backed streams
# instead of eight cross-coupled cluster FIFOs.
assign_cells_to_slr SLR0 [required_cell "*collector_slr0_U0"]
assign_cells_to_slr SLR0 [required_cell "*slr0_result_U"]
assign_cells_to_slr SLR1 [required_cell "*collector_slr1_U0"]
assign_cells_to_slr SLR1 [required_cell "*slr1_result_U"]
assign_cells_to_slr SLR2 [required_cell "*collector_slr2_U0"]
assign_cells_to_slr SLR2 [required_cell "*slr2_result_U"]
assign_cells_to_slr SLR1 [required_cell "*collector_final_U0"]
assign_cells_to_slr SLR1 [required_cell "*result_U"]
assign_cells_to_slr SLR0 [required_cell "*s2mm_U0"]
assign_cells_to_slr SLR2 [required_cell "*drain_U0"]
assign_cells_to_slr SLR2 [required_cell "*xr_8_U"]

puts "pblock_gemv_full: applied ordered 2/3/3 full-SLR pblock floorplan"

# Manual equivalents of the congestion-only QoR suggestions. Vivado 2022.1
# reports RQS_CONG-9 and RQS_CONG-3_1 for this DFX checkpoint but creates no
# queryable suggestion objects, so write_qor_suggestions cannot emit an RQS.
set cluster_primitives [get_cells -hierarchical -quiet -filter {
    IS_PRIMITIVE && NAME =~ *gemv_full_1/inst/cluster4_sys*
}]
if {[llength $cluster_primitives] > 0} {
    if {[catch {set_property EQUIVALENT_DRIVER_OPT MERGE $cluster_primitives} message]} {
        puts "pblock_gemv_full: EQUIVALENT_DRIVER_OPT not applied: $message"
    } else {
        puts "pblock_gemv_full: applied EQUIVALENT_DRIVER_OPT=MERGE to [llength $cluster_primitives] cluster primitives"
    }
}

set soft_hlutnm_cells [get_cells -hierarchical -quiet -filter {
    NAME =~ *gemv_full_1/inst/cluster4_sys* && SOFT_HLUTNM != ""
}]
if {[llength $soft_hlutnm_cells] > 0} {
    reset_property SOFT_HLUTNM $soft_hlutnm_cells
    puts "pblock_gemv_full: reset SOFT_HLUTNM on [llength $soft_hlutnm_cells] cluster cells"
}
