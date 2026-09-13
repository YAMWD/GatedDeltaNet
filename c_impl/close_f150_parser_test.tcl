# Pure-tclsh self-test run by close_f150_kernel_group.slurm before Vivado starts:
# the route-status parser must read a real report shape and reject an incomplete
# one, and the closure pass must be complete Tcl that still carries the focused
# kernel-path-group command. No Vivado command is executed here.
set ::env(REPAIR_OUT) [pwd]
set ::env(REPAIR_STATUS) [file join [pwd] parser_test.phase]
proc set_param {args} {}
source [file join [file dirname [info script]] close_f150_common.tcl]
set counts [route_counts {# of routable nets..... : 1592228 :
    # of fully routed nets..... : 1592228 :
    # of nets with routing errors..... : 0 :}]
if {[dict get $counts errors] != 0 || [dict get $counts fully] != 1592228 || [dict get $counts routable] != 1592228} {
    error "Route parse mismatch: $counts"
}
if {![catch {route_counts {incomplete report}}]} {error "Incomplete report accepted"}
set f [open [file join [file dirname [info script]] close_f150_kernel_group.tcl]]
set code [read $f]; close $f
if {![info complete $code]} {error "Incomplete Tcl in close_f150_kernel_group.tcl"}
if {![string match {*phys_opt_design -placement_opt -routing_opt -critical_cell_opt -path_groups $groups*} $code]} {
    error "Unexpected physical optimization command"
}
puts "CLOSE_F150_PARSER_AND_SYNTAX_PASS"
