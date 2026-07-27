open_project GDN
set_top gdn_forward
add_files gdn_eval.cpp
add_files gdn_model.cpp
add_files gdn_model.h
add_files -tb gdn_eval.cpp
add_files -tb artifacts
add_files -tb fixtures_parity
add_files -tb fixtures_smoke
open_solution "solution1" -flow_target vitis
set_part {xcu55c-fsvh2892-2L-e}
create_clock -period 6.667 -name default
source ./hls_gdn_forward.tcl
#source "./GDN/solution1/directives.tcl"
# Decode-only csim requires a GPU-exported .gdnstate (gitignored/regenerable).
# Uncomment and ensure fixtures_decode/decode_ex0.gdnstate exists locally before running csim.
# csim_design -argv {artifacts/gdn-1.3b-f32.gdnw fixtures_decode/decode.gdnreq results_decode_c/decode.c.json --decode --decode-from-state fixtures_decode/decode_ex0.gdnstate --decode-len 6}
csynth_design
#cosim_design -argv {artifacts/gdn-1.3b-f32.gdnw fixtures_smoke/mini_mc.gdnreq}
#export_design -format ip_catalog
