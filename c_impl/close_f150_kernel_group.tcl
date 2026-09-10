source [file join [file dirname [info script]] close_f150_common.tcl]

proc cell_location {cell} {
    if {[llength $cell] != 1} {error "Expected one primitive, got $cell"}
    return [list [get_property NAME $cell] [get_property REF_NAME $cell] \
        [get_property LOC $cell] [get_property BEL $cell] \
        [join [get_slrs -quiet -of_objects $cell] ,] \
        [join [get_clock_regions -quiet -of_objects $cell] ,] \
        [get_property PBLOCK $cell]]
}

proc inspect_ce_cone {stage required} {
    global out
    set name {level0_i/ulp/gdn_forward_1/inst/grp_gdn_gemv_fu_1054/p_read25_c_U/addr[4]_i_1__7}
    set cell [get_cells -quiet $name]
    if {[llength $cell] != 1} {
        if {$required} {error "Missing exact input CE LUT: $name"}
        note "$stage original CE LUT changed; consult final timing report"
        return
    }
    if {![string match LUT* [get_property REF_NAME $cell]]} {error "CE target is not a LUT"}
    file mkdir $out/$stage
    set f [open $out/$stage/ce_connectivity.tsv w]
    puts $f "role\tpin\tnet\tcell\tref\tloc\tbel\tslr\tclock_region\tpblock"
    puts $f [join [concat [list target NA NA] [cell_location $cell]] \t]
    # Record every input's immediate driver; do not walk the whole fanin cone.
    foreach pin [get_pins -quiet -of_objects $cell -filter {DIRECTION == IN}] {
        set nets [get_nets -quiet -of_objects $pin -top_net_of_hierarchical_group]
        foreach net $nets {
            set drivers [get_pins -quiet -leaf -of_objects $net -filter {DIRECTION == OUT}]
            foreach driver $drivers {
                set dc [get_cells -quiet -of_objects $driver]
                puts $f [join [concat [list input_driver $pin $net] [cell_location $dc]] \t]
            }
        }
    }
    set output [get_pins -quiet -of_objects $cell -filter {DIRECTION == OUT}]
    set net [get_nets -quiet -top_net_of_hierarchical_group -of_objects $output]
    if {[llength $net] != 1} {error "Expected one CE output net"}
    set loads [get_pins -quiet -leaf -of_objects $net -filter {DIRECTION == IN}]
    if {[llength $loads] > 16} {error "Unexpected CE fanout; refusing an unbounded inspection"}
    foreach pin $loads {
        puts $f [join [concat [list output_load $pin $net] \
            [cell_location [get_cells -quiet -of_objects $pin]]] \t]
    }
    close $f
    report_timing -to $loads -delay_type max -max_paths 16 -nworst 1 \
        -input_pins -file $out/$stage/all_ce_load_setup.rpt
    report_timing -to $loads -delay_type min -max_paths 16 -nworst 1 \
        -input_pins -file $out/$stage/all_ce_load_hold.rpt
    note "$stage CE LUT and all[llength $loads] loads recorded"
}

note "open verified Iter75e after_explore checkpoint"
open_checkpoint $::env(REPAIR_DCP)
set baseline [inspect input]
if {![dict get $baseline legal] || abs([dict get $baseline wns]+0.001)>0.0005} {
    error "Input legality/timing differs from the recorded Iter75e result"
}
inspect_ce_cone input 1
set groups [get_path_groups -quiet clk_kernel_00_unbuffered_net]
if {[llength $groups] != 1} {error "Expected one kernel path group"}
note "one focused post-route pass: placement/routing/critical-cell, kernel group only"
# Supported UltraScale+ post-route operations. No directive, unroute, retime,
# global clock optimization, forced replication, or timing-constraint changes.
phys_opt_design -placement_opt -routing_opt -critical_cell_opt -path_groups $groups
write_checkpoint $out/after_focused.dcp
set candidate [inspect after_focused]
inspect_ce_cone after_focused 0

note "final route, setup/hold, bus skew, DRC and regional evidence"
report_utilization -slr -file $out/utilization_slr.rpt
report_design_analysis -congestion -file $out/congestion.rpt
report_bus_skew -file $out/bus_skew.rpt
report_drc -file $out/drc.rpt
set drc_errors [get_drc_violations -quiet -filter {SEVERITY == Error}]
set f [open $out/drc_errors.txt w]; puts $f $drc_errors; close $f
if {![dict get $candidate pass] || [llength $drc_errors]} {
    note "REPAIR_NOT_CLOSED candidate_wns=[dict get $candidate wns]; CE evidence ready for step2"
    error "Timing/route/DRC gate failed; preserve baseline, no packaging"
}
write_checkpoint $out/closed_f150.dcp
set f [open $out/physical_gate.pass w]
puts $f "Three-clock setup/hold,route,DRC PASS; bus-skew review and packaging/on-card still required"
close $f
note "PHYSICAL_GATE_PASS: ready for bus-skew review, packaging and on-card validation"
