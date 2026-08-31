# Final post-route evidence for the production hardware flow.  Keep this hook
# report-only: v++ owns the implementation pass/fail decision, while these
# reports preserve the exact routed candidate needed to diagnose either result.

set gdn_final_qor_dir [file join [pwd] gdn_final_qor]
file mkdir $gdn_final_qor_dir

report_route_status \
    -file [file join $gdn_final_qor_dir route_status.rpt]
report_timing_summary -delay_type min_max -report_unconstrained \
    -max_paths 20 -input_pins \
    -file [file join $gdn_final_qor_dir timing_summary.rpt]
report_utilization -slr \
    -file [file join $gdn_final_qor_dir utilization_slr.rpt]
report_design_analysis -congestion \
    -file [file join $gdn_final_qor_dir congestion.rpt]

# The congestion report includes Vivado's SLR Net Crossing Reporting table.
# QoR suggestions are captured as text evidence only; the production build
# never reads or depends on a binary RQS file.
if {[catch {
    report_qor_suggestions \
        -file [file join $gdn_final_qor_dir qor_suggestions.rpt]
} gdn_qor_error]} {
    set qor_log [open [file join $gdn_final_qor_dir qor_suggestions_error.txt] w]
    puts $qor_log $gdn_qor_error
    close $qor_log
    puts "GDN_FINAL_QOR_WARN report_qor_suggestions failed: $gdn_qor_error"
}

# set_bus_skew is present in the platform shell.  Preserve its dedicated
# report, but do not turn absence of a skew object into a Tcl-hook failure.
if {[catch {
    report_bus_skew -file [file join $gdn_final_qor_dir bus_skew.rpt]
} gdn_bus_skew_error]} {
    set bus_skew_log [open [file join $gdn_final_qor_dir bus_skew_error.txt] w]
    puts $bus_skew_log $gdn_bus_skew_error
    close $bus_skew_log
    puts "GDN_FINAL_QOR_WARN report_bus_skew failed: $gdn_bus_skew_error"
}

set clock_log [open [file join $gdn_final_qor_dir clock_slacks.tsv] w]
puts $clock_log "clock\tsetup_wns_ns\thold_whs_ns"
foreach clock_pattern {DATA_CLK dma_ip_axi_aclk_1} {
    set clocks [get_clocks -quiet -filter "NAME =~ *${clock_pattern}*"]
    if {[llength $clocks] == 0} {
        puts $clock_log "$clock_pattern\tMISSING\tMISSING"
        puts "GDN_FINAL_QOR_WARN clock=$clock_pattern matched=0"
        continue
    }

    set setup_paths [get_timing_paths -quiet -from $clocks -to $clocks \
        -delay_type max -max_paths 1]
    set hold_paths [get_timing_paths -quiet -from $clocks -to $clocks \
        -delay_type min -max_paths 1]
    set setup_wns "NO_PATH"
    set hold_whs "NO_PATH"
    if {[llength $setup_paths] > 0} {
        set setup_wns [get_property SLACK [lindex $setup_paths 0]]
    }
    if {[llength $hold_paths] > 0} {
        set hold_whs [get_property SLACK [lindex $hold_paths 0]]
    }
    puts $clock_log "$clock_pattern\t$setup_wns\t$hold_whs"
    puts "GDN_FINAL_QOR_CLOCK clock=$clock_pattern setup_wns=$setup_wns hold_whs=$hold_whs matched=[llength $clocks]"
}
close $clock_log

puts "GDN_FINAL_QOR_DONE directory=$gdn_final_qor_dir"
