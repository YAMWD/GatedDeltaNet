# Iter56 surgical 150 MHz floorplan.
#
# Iter55d proved that hard-containing all sixteen clusters, their BRAM FIFOs,
# local collectors, auxiliary actors, and recurrence in a 6/7/3 partition was
# over-constrained: post-place SLR0--SLR1 connectivity reached 95.83%, outer
# SLR BRAM exceeded 92%, and individual SLL columns reached 179% demand.
# Preserve only the placements with prior routing evidence:
#
#   * the complete recurrent wrapper may use all of SLR2;
#   * clusters 8 and 10 stay in SLR1, with cluster 10's local input FIFOs;
#   * the three explicit collector-boundary relays and final collector meet in
#     SLR1, with the relay registers marked for dedicated SLL placement;
#   * every other cluster, FIFO, local collector, and auxiliary actor is free.

proc f150_one {pattern role} {
    set cells [get_cells -hierarchical -quiet -filter "NAME =~ $pattern"]
    if {[llength $cells] != 1} {
        error "iter56 floorplan: expected one $role matching '$pattern', matched [llength $cells]"
    }
    return [lindex $cells 0]
}

proc f150_many {pattern expected role} {
    set cells [get_cells -hierarchical -quiet -filter "NAME =~ $pattern"]
    if {[llength $cells] != $expected} {
        error "iter56 floorplan: expected $expected $role matching '$pattern', matched [llength $cells]"
    }
    return $cells
}

proc f150_slr_range {slr_name} {
    set slr [get_slrs -quiet -filter "NAME == $slr_name"]
    if {[llength $slr] != 1} {
        error "iter56 floorplan: expected one $slr_name"
    }
    set xs {}
    set ys {}
    foreach cr [get_clock_regions -quiet -of_objects $slr] {
        if {[regexp {X([0-9]+)Y([0-9]+)$} [get_property NAME $cr] -> x y]} {
            lappend xs $x
            lappend ys $y
        }
    }
    if {[llength $xs] == 0} {
        error "iter56 floorplan: no clock regions found for $slr_name"
    }
    return [format "CLOCKREGION_X%dY%d:CLOCKREGION_X%dY%d" \
        [tcl::mathfunc::min {*}$xs] [tcl::mathfunc::min {*}$ys] \
        [tcl::mathfunc::max {*}$xs] [tcl::mathfunc::max {*}$ys]]
}

proc f150_make_pblock {name range cell_names slr_name} {
    foreach cell_name $cell_names {
        set resolved [get_cells -quiet $cell_name]
        if {[llength $resolved] != 1} {
            error "iter56 floorplan: pblock $name expected one exact cell '$cell_name', resolved [llength $resolved]"
        }
    }
    set old [get_pblocks -quiet $name]
    if {[llength $old] > 0} {
        delete_pblocks $old
    }
    create_pblock $name
    resize_pblock [get_pblocks $name] -add $range
    set_property IS_SOFT false [get_pblocks $name]
    set_property CONTAIN_ROUTING false [get_pblocks $name]
    foreach cell_name $cell_names {
        set resolved [get_cells -quiet $cell_name]
        add_cells_to_pblock [get_pblocks $name] $resolved
        set_property USER_SLR_ASSIGNMENT $slr_name $resolved
    }
    set roots [get_cells -quiet -of_objects [get_pblocks $name]]
    if {[llength $roots] != [llength $cell_names]} {
        error "iter56 floorplan: pblock $name contains [llength $roots] of [llength $cell_names] requested roots"
    }
    puts "GDN_ITER56_PBLOCK name=$name slr=$slr_name range=$range cells=[llength $roots]"
}

set slr1_range [f150_slr_range SLR1]
set slr2_range [f150_slr_range SLR2]

set recurrent [f150_one \
    "*/grp_gdn_gemv_fu_*/gdn_recurrent_attention_islands_U0" \
    "two-island recurrent wrapper"]
f150_make_pblock pb_iter56_recurrent_slr2 $slr2_range \
    [list $recurrent] SLR2

set cluster8 [f150_one \
    "*/grp_gdn_gemv_fu_*/gemv32_cluster2_8_U0" "cluster 8"]
f150_make_pblock pb_iter56_cluster8_slr1 $slr1_range \
    [list $cluster8] SLR1

set cluster10_cone [list \
    [f150_one "*/grp_gdn_gemv_fu_*/gemv32_cluster2_10_U0" "cluster 10"] \
    [f150_one "*/grp_gdn_gemv_fu_*/ws_20_U" "ws20 FIFO"] \
    [f150_one "*/grp_gdn_gemv_fu_*/ws_21_U" "ws21 FIFO"] \
    [f150_one "*/grp_gdn_gemv_fu_*/xr_10_U" "xr10 FIFO"]]
f150_make_pblock pb_iter56_cluster10_slr1 $slr1_range \
    $cluster10_cone SLR1

# The three relay actors are explicit sequential boundaries added in Iter56.
# Put their receiving registers and the final collector in SLR1. The local
# collectors and their source FIFOs remain unconstrained, so SSI_SpreadSLLs can
# distribute their endpoints instead of concentrating an entire hierarchy.
set relay_roots [f150_many \
    "*/grp_gdn_gemv_fu_*/gemv32_boundary_relay*_U0" 3 \
    "collector-boundary relays"]
set relay_names {}
set relay_registers 0
foreach relay $relay_roots {
    set relay_name [get_property NAME $relay]
    lappend relay_names $relay_name
    set sequential [get_cells -hierarchical -quiet -filter \
        "NAME =~ $relay_name/* && IS_SEQUENTIAL"]
    if {[llength $sequential] == 0} {
        error "iter56 floorplan: relay $relay_name contains no sequential leaves"
    }
    set_property USER_SLL_REG TRUE $sequential
    incr relay_registers [llength $sequential]
}
set final_collector [f150_one \
    "*/grp_gdn_gemv_fu_*/gemv32_collect_final_U0" "final collector"]
lappend relay_names [get_property NAME $final_collector]
f150_make_pblock pb_iter56_result_boundary_slr1 $slr1_range \
    $relay_names SLR1

# Preserve the demonstrated clock-region-local reset replication. Iter55d
# proved this property applies successfully; reset was not its route failure.
set reset_net [get_nets -hierarchical -quiet -regexp \
    {^level0_i/ulp/gdn_forward_1/inst/ap_rst_n_inv$}]
if {[llength $reset_net] != 1} {
    error "iter56 floorplan: expected one flattened kernel reset net"
}
set_property MAX_FANOUT_MODE CLOCK_REGION $reset_net
set_property FORCE_MAX_FANOUT 32 $reset_net

puts "GDN_ITER56_DONE collector_cut=4/6/6 recurrent=full_slr2 clusters=8,10_slr1 relay_regs=$relay_registers reset_fanout=32 broad_pblocks=0"
