# Report reset BUFG insertion without constraining it. At 150 MHz Vivado must be
# free to remove, replicate, or relocate this network during placement.

set reset_bufgs [get_cells -hierarchical -quiet -filter {
    NAME =~ *gemv_full_1/inst/ap_rst_n_inv_BUFG* && REF_NAME =~ BUFG*
}]

puts "place_gemv_full: found [llength $reset_bufgs] reset BUFG(s): $reset_bufgs"
