# Fresh implementation only: replicate eight measured control/address drivers
# before placement. No reset rewiring, clock change, or floorplan relaxation.
set iter75d_dir [file dirname [file normalize [info script]]]
source [file join $iter75d_dir apply_iter66e_unpair.tcl]

set root level0_i/ulp/gdn_forward_1/inst
set gemv $root/grp_gdn_gemv_fu_1054
set rms $root/grp_gdn_rmsnorm_rows_bf16_fu_1415/grp_gdn_rmsnorm_rows_bf16_Pipeline_rms_load_w_fu_108
set onorm $root/grp_gdn_output_norm_and_gate_fu_1497/grp_gdn_output_norm_and_gate_Pipeline_onorm_load_w_fu_200
# Resolve the reset at its primitive Q, not a post-route lopt alias.
set specs [list \
    [list reset cell {level0_i/ulp/proc_sys_reset_kernel_slr0/U0/ACTIVE_LOW_PR_OUT_DFF[0].FDRE_PER_N} 32] \
    [list gemv_launch cell $root/grp_gdn_gemv_fu_1054_ap_start_reg_reg 8] \
    [list top_state92 net $root/ap_CS_fsm_state92 8] \
    [list rms_addr7 net ${rms}/zext_ln1178_2_fu_347_p1\[7\] 32] \
    [list rms_addr8 net ${rms}/zext_ln1178_2_fu_347_p1\[8\] 32] \
    [list onorm_addr7 net ${onorm}/zext_ln2284_2_fu_347_p1\[7\] 32] \
    [list onorm_addr8 net ${onorm}/zext_ln2284_2_fu_347_p1\[8\] 32] \
    [list store_state5 net $gemv/gemv32_store_or_qkvg_conv_stream_U0/grp_gemv32_store_fu_178/ap_CS_fsm_state5 128]]
set fp [open iter75d_control_targets.tsv w]
puts $fp "target\tdriver\tnet\tflat_pins_before\tforce_max_fanout"
set seen [dict create]
foreach spec $specs {
    lassign $spec role kind name limit
    if {$kind eq "cell"} {
        set driver [get_cells -quiet $name]
    } else {
        set net [get_nets -quiet $name]
        if {[llength $net] != 1} {error "ITER75D: expected one $role net: $name"}
        set driver [get_cells -quiet -of_objects [get_pins -quiet -leaf \
            -of_objects $net -filter {DIRECTION == OUT}]]
    }
    if {[llength $driver] != 1 || [get_property REF_NAME $driver] ne "FDRE"} {
        error "ITER75D: $role must resolve to exactly one FDRE: $driver"
    }
    set dn [get_property NAME $driver]
    if {[dict exists $seen $dn]} {error "ITER75D: duplicate driver $dn"}
    dict set seen $dn 1
    set q [get_pins -quiet -of_objects $driver -filter {REF_PIN_NAME == Q}]
    set net [get_nets -quiet -top_net_of_hierarchical_group -of_objects $q]
    if {[llength $net] != 1} {error "ITER75D: ambiguous $role Q net"}
    set_property FORCE_MAX_FANOUT $limit $net
    if {[get_property FORCE_MAX_FANOUT $net] != $limit} {
        error "ITER75D: fanout property was not retained for $role"
    }
    puts $fp "$role\t$dn\t[get_property NAME $net]\t[get_property FLAT_PIN_COUNT $net]\t$limit"
    puts "ITER75D_TARGET role=$role driver=$dn force_max_fanout=$limit"
}
close $fp
puts "ITER75D_CONTROL_FANOUT_DONE targets=[dict size $seen]"
