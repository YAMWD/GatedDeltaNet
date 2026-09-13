# Bounded continuation of job3751, without unroute or route serialization.
set out $::env(REPAIR_OUT)
file mkdir $out
set_param general.maxThreads 8

proc note {message} {
    puts "ITER75F [clock format [clock seconds] -format %Y-%m-%dT%H:%M:%S] $message"
    flush stdout
    set f [open $::env(REPAIR_STATUS) w]
    puts $f $message
    close $f
}

proc route_counts {text} {
    set result {}
    foreach {key label} {routable {routable nets} fully {fully routed nets} errors {nets with routing errors}} {
        set pattern [format {# of %s\.+\s*:\s*([0-9]+)} $label]
        if {![regexp $pattern $text -> count]} {error "Missing route-status field: $label"}
        dict set result $key $count
    }
    return $result
}

proc inspect {stage} {
    global out
    note "inspect $stage"
    set dir [file join $out $stage]
    file mkdir $dir
    set route [report_route_status -return_string]
    set f [open $dir/route_status.rpt w]; puts $f $route; close $f
    set counts [route_counts $route]
    set legal [expr {[dict get $counts errors] == 0 && [dict get $counts routable] > 0 &&
                     [dict get $counts routable] == [dict get $counts fully]}]
    set pass $legal
    set fp [open $dir/clock_slacks.tsv w]
    puts $fp "clock\tperiod_ns\tsetup_ns\thold_ns"
    foreach {name expected} {clk_kernel_00_unbuffered_net 6.667 dma_ip_axi_aclk_1 4.000 hbm_aclk 2.222} {
        set clk [get_clocks -quiet $name]
        if {[llength $clk] != 1} {error "Expected exactly one clock $name"}
        set period [get_property PERIOD $clk]
        if {abs($period-$expected)>0.002} {error "Wrong clock period: $name=$period"}
        set sp [get_timing_paths -quiet -from $clk -to $clk -delay_type max -max_paths 1]
        set hp [get_timing_paths -quiet -from $clk -to $clk -delay_type min -max_paths 1]
        if {![llength $sp] || ![llength $hp]} {error "Missing timing paths for $name"}
        set ss [get_property SLACK $sp]; set hs [get_property SLACK $hp]
        puts $fp "$name\t$period\t$ss\t$hs"
        puts "ITER75F_CLOCK stage=$stage clock=$name setup=$ss hold=$hs"
        if {$ss<0 || $hs<0} {set pass 0}
        if {$name eq "clk_kernel_00_unbuffered_net"} {set kernel_wns $ss}
    }
    close $fp
    set clk [get_clocks clk_kernel_00_unbuffered_net]
    # Bounded endpoints, never full-design net/ROUTE enumeration.
    set paths [get_timing_paths -quiet -from $clk -to $clk -delay_type max \
        -nworst 1 -max_paths 256 -slack_lesser_than 0]
    set f [open $dir/failing_endpoints.tsv w]
    puts $f "slack\tstartpoint\tendpoint\tstart_loc\tend_loc"
    foreach p $paths {
        set sp [get_property STARTPOINT_PIN $p]
        set ep [get_property ENDPOINT_PIN $p]
        set sc [get_cells -quiet -of_objects $sp]
        set ec [get_cells -quiet -of_objects $ep]
        puts $f "[get_property SLACK $p]\t$sp\t$ep\t[get_property LOC $sc]\t[get_property LOC $ec]"
    }
    close $f
    report_timing -from $clk -to $clk -delay_type max -nworst 1 -max_paths 256 \
        -slack_lesser_than 0 -input_pins -file $dir/failing_paths.rpt
    report_timing_summary -delay_type min_max -max_paths 3 -file $dir/timing_summary.rpt
    set f [open $dir/verdict.tsv w]
    puts $f "route_legal\t$legal\nclocks_and_route_pass\t$pass\nreported_failing_endpoints\t[llength $paths]\nendpoint_cap\t256\npossibly_truncated\t[expr {[llength $paths]>=256}]"
    close $f
    note "$stage route_legal=$legal timing_pass=$pass wns=$kernel_wns endpoints=[llength $paths] cap=256"
    return [dict create pass $pass legal $legal wns $kernel_wns]
}
