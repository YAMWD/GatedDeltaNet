set checkpoint [lindex $argv 0]
set floorplan [lindex $argv 1]

if {$checkpoint eq "" || $floorplan eq ""} {
    error "usage: vivado -mode batch -source validate_gemv_floorplan.tcl -tclargs <checkpoint> <floorplan>"
}

open_checkpoint [file normalize $checkpoint]
foreach name {pb_gemv_full_SLR0 pb_gemv_full_SLR1 pb_gemv_full_SLR2} {
    set pblock [get_pblocks -quiet $name]
    if {[llength $pblock] > 0} {
        delete_pblocks $pblock
    }
}

source [file normalize $floorplan]
puts "FLOORPLAN_VALIDATION_COMPLETE"
close_design
exit
