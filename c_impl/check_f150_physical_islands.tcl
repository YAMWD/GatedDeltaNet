# Iter56 post-place report. Structural floorplan violations remain fatal, but
# utilization/SLL thresholds are observations only: Iter54c proved that a
# 98.84%-occupied SLR0 and 88.87% lower-boundary SLL use can still route, so
# placement metrics alone must not prevent route_design from supplying the
# decisive evidence.

proc iter56_check_pblock {name expected_slr expected_roots} {
    set pb [get_pblocks -quiet $name]
    if {[llength $pb] != 1} {
        error "iter56 gate: missing pblock $name"
    }
    set roots [get_cells -quiet -of_objects $pb -include_replicated_objects]
    if {[llength $roots] != $expected_roots} {
        error "iter66b gate: pblock $name has [llength $roots] roots, expected $expected_roots"
    }
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

iter56_check_pblock pb_iter56_recurrent_slr2 SLR2 1
iter56_check_pblock pb_iter56_cluster8_slr1 SLR1 1
iter56_check_pblock pb_iter66b_cluster10_slr1 SLR1 4
iter56_check_pblock pb_iter56_result_boundary_slr1 SLR1 4

# Report where clusters 9/10 and their transport hierarchies actually land.
# Cluster 10 plus ws20/ws21/xr10 are checked above; cluster 9 and the remaining
# transports are observations rather than placement constraints.
set gdn_script_dir [file dirname [file normalize [info script]]]
set gdn_place_report_dir [file join $gdn_script_dir diagnostics placement_reports]
file mkdir $gdn_place_report_dir
set gdn_actor_report_path [file join $gdn_place_report_dir actor_slr_distribution.rpt]
set gdn_actor_report [open $gdn_actor_report_path w]

proc iter66b_report_actor_slr {pattern role report_channel} {
    set roots [get_cells -hierarchical -quiet -filter "NAME =~ $pattern"]
    if {[llength $roots] != 1} {
        error "iter66b report: expected one $role matching '$pattern', matched [llength $roots]"
    }
    set root [lindex $roots 0]
    set root_name [get_property NAME $root]
    set leaves [get_cells -hierarchical -quiet -filter \
        "NAME =~ $root_name/* && IS_PRIMITIVE"]
    array set count {SLR0 0 SLR1 0 SLR2 0 OTHER 0 UNPLACED 0}
    foreach leaf $leaves {
        set loc [get_property -quiet LOC $leaf]
        if {$loc eq ""} {
            incr count(UNPLACED)
            continue
        }
        set slr [get_slrs -quiet -of_objects [get_sites -quiet $loc]]
        if {[llength $slr] != 1} {
            incr count(OTHER)
            continue
        }
        set slr_name [get_property NAME $slr]
        if {[info exists count($slr_name)]} {
            incr count($slr_name)
        } else {
            incr count(OTHER)
        }
    }
    set line "GDN_ITER66B_ACTOR role=$role root=$root_name leaves=[llength $leaves] slr0=$count(SLR0) slr1=$count(SLR1) slr2=$count(SLR2) other=$count(OTHER) unplaced=$count(UNPLACED)"
    puts $line
    puts $report_channel $line
}

foreach actor_spec {
    {"*/grp_gdn_gemv_fu_*/gemv32_cluster2_9_U0" cluster9}
    {"*/grp_gdn_gemv_fu_*/gemv32_cluster2_10_U0" cluster10}
    {"*/grp_gdn_gemv_fu_*/ws_18_U" ws18}
    {"*/grp_gdn_gemv_fu_*/ws_19_U" ws19}
    {"*/grp_gdn_gemv_fu_*/ws_20_U" ws20}
    {"*/grp_gdn_gemv_fu_*/ws_21_U" ws21}
    {"*/grp_gdn_gemv_fu_*/xr_9_U" xr9}
    {"*/grp_gdn_gemv_fu_*/xr_10_U" xr10}
    {"*/grp_gdn_gemv_fu_*/ys_9_U" ys9}
    {"*/grp_gdn_gemv_fu_*/ys_10_U" ys10}
    {"*/grp_gdn_gemv_fu_*/gemv32_collect6_16_U0" upper_collector}
} {
    # Observation only: a renamed/dissolved/replicated cell must not kill a
    # 7-32 h link after placement. The structural pblock gates above remain
    # the only fatal checks.
    if {[catch {
        iter66b_report_actor_slr \
            [lindex $actor_spec 0] [lindex $actor_spec 1] $gdn_actor_report
    } gdn_actor_err]} {
        set gdn_actor_warn "GDN_ITER66B_WARN actor report [lindex $actor_spec 1] failed: $gdn_actor_err"
        puts $gdn_actor_warn
        catch {puts $gdn_actor_report $gdn_actor_warn}
    }
}
close $gdn_actor_report

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

# Preserve the exact placement evidence before routing.  Congestion analysis
# is advisory; the structural pblock checks above are the only fatal gates.
report_utilization -slr -file \
    [file join $gdn_place_report_dir utilization_slr.rpt]
report_timing_summary -delay_type min_max -max_paths 20 -file \
    [file join $gdn_place_report_dir timing_summary.rpt]
if {[catch {
    report_design_analysis -congestion -file \
        [file join $gdn_place_report_dir congestion.rpt]
} gdn_congestion_err]} {
    puts "GDN_ITER66B_WARN placement congestion report failed: $gdn_congestion_err"
}

# ---------------------------------------------------------------------------
# Post-place checkpoint.
#
# v++ carries the design in memory across place -> phys_opt -> route and writes
# no intermediate DCP of its own, so an interrupted link (node shutdown, OOM,
# dropped session) loses hours of placement with nothing to resume from. This
# hook already runs after place_design, so writing the checkpoint here costs a
# couple of minutes and makes routing restartable:
#
#   open_checkpoint <this file>
#   route_design -directive NoTimingRelaxation
#   write_checkpoint -force post_route.dcp
#
# Written outside the build tree so a `make clean` or a rebuilt build directory
# does not take it with them.
# ---------------------------------------------------------------------------
# NOTE: @C_IMPL_DIR@ is substituted by the Makefile into the *cfg* file only,
# never inside a sourced Tcl script -- Iter59 wrote its checkpoint into a
# directory literally named "@C_IMPL_DIR@" under impl_1. Derive the path from
# this script's own location instead, which is always absolute.
set gdn_ckpt_dir [file join $gdn_script_dir diagnostics checkpoints]
if {[catch {file mkdir $gdn_ckpt_dir} gdn_ckpt_err]} {
    puts "GDN_CKPT_WARN could not create $gdn_ckpt_dir: $gdn_ckpt_err"
} else {
    set gdn_ckpt_path [file join $gdn_ckpt_dir "post_place.dcp"]
    if {[catch {write_checkpoint -force $gdn_ckpt_path} gdn_ckpt_err]} {
        puts "GDN_CKPT_WARN write_checkpoint failed: $gdn_ckpt_err"
    } else {
        puts "GDN_CKPT_OK post-place checkpoint written to $gdn_ckpt_path"
    }
}
