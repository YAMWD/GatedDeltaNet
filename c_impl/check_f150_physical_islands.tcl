# Iter56 post-place report. Structural floorplan violations remain fatal, but
# utilization/SLL thresholds are observations only: Iter54c proved that a
# 98.84%-occupied SLR0 and 88.87% lower-boundary SLL use can still route, so
# placement metrics alone must not prevent route_design from supplying the
# decisive evidence.

proc iter56_check_pblock {name expected_slr} {
    set pb [get_pblocks -quiet $name]
    if {[llength $pb] != 1} {
        error "iter56 gate: missing pblock $name"
    }
    set roots [get_cells -quiet -of_objects $pb -include_replicated_objects]
    set placed 0
    set outside 0
    foreach root $roots {
        set root_name [get_property NAME $root]
        set leaves [get_cells -hierarchical -quiet -filter \
            "NAME =~ $root_name/* && IS_PRIMITIVE"]
        if {[get_property -quiet IS_PRIMITIVE $root]} {
            set leaves [list $root]
        }
        foreach leaf $leaves {
            set loc [get_property -quiet LOC $leaf]
            if {$loc eq ""} { continue }
            incr placed
            set slr [get_slrs -quiet -of_objects [get_sites -quiet $loc]]
            if {[llength $slr] != 1 ||
                [get_property NAME $slr] ne $expected_slr} {
                incr outside
            }
        }
    }
    puts "GDN_ITER56_CHECK pblock=$name expected=$expected_slr roots=[llength $roots] placed=$placed outside=$outside"
    if {$placed == 0 || $outside != 0} {
        error "iter56 gate: $name has $outside of $placed placed cells outside $expected_slr"
    }
}

iter56_check_pblock pb_iter56_recurrent_slr2 SLR2
iter56_check_pblock pb_iter56_cluster8_slr1 SLR1
iter56_check_pblock pb_iter56_cluster10_slr1 SLR1
iter56_check_pblock pb_iter56_result_boundary_slr1 SLR1

set util [report_utilization -slr -return_string]
set lower_pct -1.0
set upper_pct -1.0
set clb_pct {}
set bram_pct {}
set direct_20 -1
set direct_02 -1
set in_matrix 0
foreach line [split $util "\n"] {
    if {[regexp {^2[.] SLR Connectivity Matrix} $line]} {
        set in_matrix 1
        continue
    }
    if {[regexp {^3[.] SLR CLB Logic} $line]} {
        set in_matrix 0
    }
    if {[regexp {^\|\s*SLR2 <-> SLR1\s*\|\s*[0-9]+\s*\|\s*\|\s*[0-9]+\s*\|\s*([0-9.]+)\s*\|} $line -> pct]} {
        set upper_pct $pct
    }
    if {[regexp {^\|\s*SLR1 <-> SLR0\s*\|\s*[0-9]+\s*\|\s*\|\s*[0-9]+\s*\|\s*([0-9.]+)\s*\|} $line -> pct]} {
        set lower_pct $pct
    }
    if {$in_matrix && [regexp {^\|\s*SLR2\s*\|\s*[0-9]+\s*\|\s*[0-9]+\s*\|\s*([0-9]+)\s*\|} $line -> n]} {
        set direct_20 $n
    }
    if {$in_matrix && [regexp {^\|\s*SLR0\s*\|\s*([0-9]+)\s*\|\s*[0-9]+\s*\|\s*[0-9]+\s*\|} $line -> n]} {
        set direct_02 $n
    }
    if {[regexp {^\|\s*CLB\s*\|\s*[0-9]+\s*\|\s*[0-9]+\s*\|\s*[0-9]+\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|} $line -> p0 p1 p2]} {
        set clb_pct [list $p0 $p1 $p2]
    }
    if {[regexp {^\|\s*Block RAM Tile\s*\|\s*[0-9.]+\s*\|\s*[0-9.]+\s*\|\s*[0-9.]+\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|} $line -> p0 p1 p2]} {
        set bram_pct [list $p0 $p1 $p2]
    }
}

if {$lower_pct < 0 || $upper_pct < 0 ||
    [llength $clb_pct] != 3 || [llength $bram_pct] != 3 ||
    $direct_20 < 0 || $direct_02 < 0} {
    error "iter56 gate: could not parse the complete report_utilization -slr result"
}
set direct_total [expr {$direct_20 + $direct_02}]
puts "GDN_ITER56_SLR lower_pct=$lower_pct upper_pct=$upper_pct direct_02=$direct_02 direct_20=$direct_20 direct_total=$direct_total clb=$clb_pct bram=$bram_pct"

set advisory_count 0
if {$lower_pct > 85.0} {
    puts "GDN_ITER56_ADVISORY metric=SLR0_SLR1_CONNECTIVITY value=$lower_pct threshold=85"
    incr advisory_count
}
if {$upper_pct > 65.0} {
    puts "GDN_ITER56_ADVISORY metric=SLR1_SLR2_CONNECTIVITY value=$upper_pct threshold=65"
    incr advisory_count
}
if {$direct_total > 6000} {
    puts "GDN_ITER56_ADVISORY metric=DIRECT_SLR0_SLR2 value=$direct_total threshold=6000"
    incr advisory_count
}
for {set slr 0} {$slr < 3} {incr slr} {
    if {[lindex $clb_pct $slr] > 95.0} {
        puts "GDN_ITER56_ADVISORY metric=SLR${slr}_CLB value=[lindex $clb_pct $slr] threshold=95"
        incr advisory_count
    }
    if {[lindex $bram_pct $slr] > 90.0} {
        puts "GDN_ITER56_ADVISORY metric=SLR${slr}_BRAM value=[lindex $bram_pct $slr] threshold=90"
        incr advisory_count
    }
}

set sll_regs [get_cells -hierarchical -quiet -filter {USER_SLL_REG == 1}]
if {[llength $sll_regs] == 0} {
    error "iter56 gate: no USER_SLL_REG boundary registers survived placement"
}
puts "GDN_ITER56_REPORT_ONLY advisories=$advisory_count sll_regs=[llength $sll_regs] proceeding_to_route"
