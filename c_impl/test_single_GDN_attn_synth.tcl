# Synth-only run for the gdn_attn_forward target. csim is skipped here
# because the inlined hls::stream-based systolic kernel takes a very long
# time under sequential csim semantics — native parity has already been
# validated through gdn_eval / gdn_attn_test. csynth_design alone gives
# the latency / resource report we need to compare against the tiled
# baseline.
open_project GDN_single_attn_synth
set_top gdn_attn_forward
add_files gdn_model.cpp
add_files gdn_model.h
open_solution -reset "solution_synth" -flow_target vitis
set_part {xcu55c-fsvh2892-2L-e}
create_clock -period 10 -name default
config_export -format ip_catalog -rtl verilog -vivado_clock 10
csynth_design
