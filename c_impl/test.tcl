open_project GDN
set_top gdn_forward
add_files gdn_model.cpp
add_files gdn_model.h
add_files -tb gdn_eval.cpp
open_solution "solution1" -flow_target vitis
set_part {xcu55c-fsvh2892-2L-e}
create_clock -period 6.667 -name default
source ./hls_gdn_forward.tcl
# Decode-only csim requires a GPU-exported .gdnstate (gitignored/regenerable).
# Supply current BF16-exact weights/state and logits-reference paths explicitly
# before enabling csim or cosim.
csynth_design
exit
