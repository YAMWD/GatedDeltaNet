set checkpoint [lindex $argv 0]
set output_file [lindex $argv 1]

if {$checkpoint eq "" || $output_file eq ""} {
    error "usage: vivado -mode batch -source report_hbm_port_slrs.tcl -tclargs <checkpoint> <output-file>"
}

set checkpoint [file normalize $checkpoint]
set output_file [file normalize $output_file]
open_checkpoint $checkpoint

set output [open $output_file w]
puts $output "port dominant_slr slr0_primitives slr1_primitives slr2_primitives total_primitives"

for {set port 0} {$port < 32} {incr port} {
    set pattern "*gemv_full_1/inst/gmem${port}_m_axi_U/*"
    set primitives [get_cells -hierarchical -quiet -filter "IS_PRIMITIVE && NAME =~ $pattern"]
    set counts [dict create SLR0 0 SLR1 0 SLR2 0]

    foreach primitive $primitives {
        set sites [get_sites -quiet -of_objects $primitive]
        if {[llength $sites] == 0} {
            continue
        }
        set slrs [get_slrs -quiet -of_objects $sites]
        if {[llength $slrs] == 1 && [dict exists $counts $slrs]} {
            dict incr counts $slrs
        }
    }

    set dominant SLR0
    foreach slr {SLR1 SLR2} {
        if {[dict get $counts $slr] > [dict get $counts $dominant]} {
            set dominant $slr
        }
    }
    puts $output [format "%2d %4s %7d %7d %7d %7d" \
        $port $dominant \
        [dict get $counts SLR0] [dict get $counts SLR1] \
        [dict get $counts SLR2] [llength $primitives]]
}

close $output
close_design
exit
