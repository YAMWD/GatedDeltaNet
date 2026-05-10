CXX ?= c++
# C++14; -Wno-unknown-pragmas because the Vitis HLS pragmas in the synthesis
# headers are not understood by host gcc.
CXXFLAGS ?= -O3 -std=c++14 -Wall -Wextra -pedantic -Wno-unknown-pragmas

# Vitis HLS install location (provides hls_stream.h, used by gdn_model.cpp's
# inlined systolic kernel and by gdn_matmul_systolic.cpp).
XILINX_HLS_INC ?= /tools/Xilinx/Vitis_HLS/2022.1/include

all: gdn_eval gdn_attn_test gdn_matmul_test

gdn_eval: gdn_eval.cpp gdn_model.cpp gdn_model.h
	$(CXX) $(CXXFLAGS) -I$(XILINX_HLS_INC) -o $@ gdn_eval.cpp gdn_model.cpp -lm

gdn_attn_test: gdn_attn_test.cpp gdn_model.cpp gdn_model.h
	$(CXX) $(CXXFLAGS) -I$(XILINX_HLS_INC) -o $@ gdn_attn_test.cpp gdn_model.cpp -lm

# gdn_matmul_test exercises the standalone systolic top (gdn_matmul_top in
# gdn_matmul_systolic.cpp) — still linked here as a separate TU. The static
# helpers inside gdn_model.cpp do not collide because they're file-local.
gdn_matmul_test: gdn_matmul_test.cpp gdn_matmul_systolic.cpp gdn_model.cpp gdn_model.h
	$(CXX) $(CXXFLAGS) -I$(XILINX_HLS_INC) -o $@ gdn_matmul_test.cpp gdn_matmul_systolic.cpp gdn_model.cpp -lm

clean:
	rm -f gdn_eval gdn_attn_test gdn_matmul_test
