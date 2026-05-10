# Synth-only run for the full 24-layer gdn_forward. csim is skipped
# because the inlined hls::stream-based systolic kernel takes very long
# under sequential csim semantics — native parity has already been
# validated through gdn_eval. csynth_design alone gives the latency /
# resource report we need to evaluate whether the whole-model design
# fits on the U55C.
open_project GDN_synth
set_top gdn_forward
add_files gdn_model.cpp
add_files gdn_model.h
open_solution -reset "solution_synth" -flow_target vitis
set_part {xcu55c-fsvh2892-2L-e}
create_clock -period 10 -name default
config_export -format ip_catalog -rtl verilog -vivado_clock 10
config_interface -m_axi_alignment_byte_size 64 -m_axi_latency 64 -m_axi_max_widen_bitwidth 512
csynth_design
