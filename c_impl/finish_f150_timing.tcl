# Called AFTER the normal post-route AggressiveExplore pass. Reproduce the
# measured sequence, not an untested replacement of AggressiveExplore.
set gdn_finish_f150_script_dir [file dirname [file normalize [info script]]]
# Progress notes in plain Tcl: stdout (-> impl_1/runme.log) plus, when the build
# exports GDN_SHARED_LOG_DIR, an append-only live file on shared storage.
# Iter76 draw 2 (build 3963) died at this hook's first statement because the
# vpl run context has no `redirect` command -- use only commands that have run
# in a real Vivado session on this design.
proc gdn_f150_note {msg} {
    puts $msg
    if {[info exists ::env(GDN_SHARED_LOG_DIR)]} {
        catch {
            file mkdir $::env(GDN_SHARED_LOG_DIR)
            set fh [open [file join $::env(GDN_SHARED_LOG_DIR) physical_finish.live.log] a]
            puts $fh "[clock format [clock seconds] -format %Y-%m-%dT%H:%M:%S] $msg"
            close $fh
        }
    }
}
proc gdn_finish_f150 {dir} {
    set clk [get_clocks -quiet clk_kernel_00_unbuffered_net]
    if {[llength $clk] != 1 || abs([get_property PERIOD $clk]-6.667)>0.002} {
        error "GDN_F150: expected the unscaled 150MHz kernel clock"
    }
    write_checkpoint -force gdn_f150_after_aggressive.dcp
    set p [get_timing_paths -quiet -from $clk -to $clk -delay_type max -max_paths 1]
    if {![llength $p]} {error "GDN_F150: no kernel setup paths"}
    gdn_f150_note "GDN_F150_WNS stage=after_aggressive kernel_setup=[get_property SLACK $p]"
    if {[get_property SLACK $p]<0} {
        gdn_f150_note "GDN_F150_STAGE additional post-route Explore"
        phys_opt_design -directive Explore
        write_checkpoint -force gdn_f150_after_explore.dcp
        set p [get_timing_paths -quiet -from $clk -to $clk -delay_type max -max_paths 1]
        gdn_f150_note "GDN_F150_WNS stage=after_explore kernel_setup=[get_property SLACK $p]"
        if {[get_property SLACK $p]<0} {
            set groups [get_path_groups -quiet clk_kernel_00_unbuffered_net]
            if {[llength $groups]!=1} {error "GDN_F150: missing kernel path group"}
            gdn_f150_note "GDN_F150_STAGE focused kernel placement/routing/critical-cell optimization"
            phys_opt_design -placement_opt -routing_opt -critical_cell_opt -path_groups $groups
            write_checkpoint -force gdn_f150_after_focused.dcp
            set p [get_timing_paths -quiet -from $clk -to $clk -delay_type max -max_paths 1]
            gdn_f150_note "GDN_F150_WNS stage=after_focused kernel_setup=[get_property SLACK $p]"
        }
    }
    # Preserve the final candidate before a gate can abort the normal run.
    write_checkpoint -force gdn_f150_final_candidate.dcp
    gdn_f150_note "GDN_F150_STAGE exact-clock gate, route status, DRC, bus skew"
    source [file join $dir check_iter75d_final_timing.tcl]
    set route [report_route_status -return_string]
    foreach {key label} {total {routable nets} fully {fully routed nets} errors {nets with routing errors}} {
        if {![regexp [format {# of %s\.+\s*:\s*([0-9]+)} $label] $route -> count]} {
            error "GDN_F150: route-status field missing: $label"
        }
        set counts($key) $count
    }
    if {$counts(errors)!=0 || $counts(total)==0 || $counts(total)!=$counts(fully)} {
        error "GDN_F150: route is not legal and complete"
    }
    report_drc -file [file join $gdn_final_qor_dir drc.rpt]
    if {[llength [get_drc_violations -quiet -filter {SEVERITY == Error}]]} {
        error "GDN_F150: DRC errors remain"
    }
    set skew_file [file join $gdn_final_qor_dir bus_skew.rpt]
    if {![file exists $skew_file] || [file exists [file join $gdn_final_qor_dir bus_skew_error.txt]]} {
        error "GDN_F150: missing successful bus-skew report"
    }
    set f [open $skew_file r]
    set skew [read $f]
    close $f
    if {[string first "VIOLATED" $skew] >= 0} {error "GDN_F150: bus-skew violation"}
    gdn_f150_note "GDN_F150_FINAL_PASS: exact-clock setup/hold and route/DRC passed"
}

# Run directly; record the abort reason in the live log, then re-raise so the
# vpl run fails closed exactly as before.
if {[catch {gdn_finish_f150 $gdn_finish_f150_script_dir} gdn_f150_err]} {
    gdn_f150_note "GDN_F150_ABORT: $gdn_f150_err"
    error $gdn_f150_err
}
