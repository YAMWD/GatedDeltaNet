set checkpoint [lindex $argv 0]
set output_dir [lindex $argv 1]

if {$checkpoint eq "" || $output_dir eq ""} {
    error "usage: vivado -mode batch -source analyze_routed_checkpoint.tcl -tclargs <checkpoint> <output_dir>"
}

set checkpoint [file normalize $checkpoint]
set output_dir [file normalize $output_dir]
file mkdir $output_dir

if {![file exists $checkpoint]} {
    error "checkpoint does not exist: $checkpoint"
}

puts "Opening checkpoint: $checkpoint"
puts "Writing reports to: $output_dir"
open_checkpoint $checkpoint

set failed 0

proc run_report {name body} {
    global failed
    puts "REPORT_START $name"
    if {[catch {uplevel 1 $body} message options]} {
        puts "REPORT_FAILED $name: $message"
        puts [dict get $options -errorinfo]
        set failed 1
    } else {
        puts "REPORT_COMPLETE $name"
    }
}

run_report route_status {
    report_route_status -show_all \
        -file [file join $output_dir route_status.rpt]
}

run_report congestion {
    report_design_analysis -congestion \
        -file [file join $output_dir congestion.rpt]
}

run_report complexity {
    report_design_analysis -complexity \
        -file [file join $output_dir complexity.rpt]
}

run_report utilization_hierarchical {
    report_utilization -hierarchical -hierarchical_depth 6 \
        -file [file join $output_dir utilization_hierarchical.rpt]
}

run_report utilization_slr {
    report_utilization -slr \
        -file [file join $output_dir utilization_slr.rpt]
}

run_report timing_summary {
    report_timing_summary -delay_type min_max -report_unconstrained \
        -check_timing_verbose -max_paths 100 -input_pins \
        -file [file join $output_dir timing_summary.rpt]
}

run_report high_fanout {
    report_high_fanout_nets -timing -max_nets 200 \
        -file [file join $output_dir high_fanout.rpt]
}

run_report qor_suggestions {
    report_qor_suggestions \
        -file [file join $output_dir qor_suggestions.rpt]
}

run_report write_qor_suggestions {
    write_qor_suggestions -force \
        [file join $output_dir qor_suggestions.rqs]
}

close_design
puts "ANALYSIS_COMPLETE failed=$failed"
exit $failed
