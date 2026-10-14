# hbm_rd: read-only HBM streaming microbenchmark (Iter88 diagnosis). Vitis 2024.2, 300 MHz.
SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -c
CXX          ?= c++
XILINX_VITIS ?= /tools/Xilinx/Vitis/2024.2
XILINX_XRT   ?= /opt/xilinx/xrt
VPP          ?= $(XILINX_VITIS)/bin/v++
PLATFORM     ?= xilinx_u55c_gen3x16_xdma_3_202210_1
FREQ         ?= 300
JOBS         ?= 8
BANKS        ?= 0 12 20 31
NCU          := $(words $(BANKS))
KERNEL       := hbm_rd
BUILD_DIR    := build.hw.$(KERNEL).f$(FREQ)
XO           := $(BUILD_DIR)/$(KERNEL).xo
XCLBIN       := $(BUILD_DIR)/$(KERNEL).xclbin
HOST_BIN     ?= host_hbm_rd.exe
HOST_INC     := -I$(XILINX_VITIS)/include -I$(XILINX_XRT)/include
HOST_LIB     := -L$(XILINX_XRT)/lib -Wl,-rpath,$(XILINX_XRT)/lib -lxrt_core -lxrt_coreutil -lpthread -lrt -ldl -lstdc++
VPP_FLAGS    := -t hw --platform $(PLATFORM) --save-temps --optimize 2 --kernel_frequency $(FREQ)
VPP_LDFLAGS  := --vivado.synth.jobs $(JOBS) --vivado.impl.jobs $(JOBS)
VPP_LDFLAGS  += --connectivity.nk $(KERNEL):$(NCU):$(subst $(eval) ,.,$(foreach i,$(shell seq 1 $(NCU)),$(KERNEL)_$(i)))
VPP_LDFLAGS  += $(foreach i,$(shell seq 1 $(NCU)),--connectivity.sp $(KERNEL)_$(i).in:HBM[$(word $(i),$(BANKS))] --connectivity.sp $(KERNEL)_$(i).out:HBM[$(word $(i),$(BANKS))] --connectivity.slr $(KERNEL)_$(i):SLR0)
VPP_LDFLAGS  += --vivado.prop run.impl_1.STEPS.OPT_DESIGN.TCL.PRE=$(abspath apply_kernel_clock.tcl)
VPP_LDFLAGS  += --vivado.prop run.impl_1.STEPS.ROUTE_DESIGN.TCL.POST=$(abspath report_clocks.tcl)
.PHONY: all xo xclbin host clean
all: xclbin
xo: $(XO)
$(XO): hbm_rd.cpp hls_hbm_rd.tcl
	@mkdir -p $(BUILD_DIR)
	$(VPP) $(VPP_FLAGS) --hls.jobs $(JOBS) --temp_dir $(BUILD_DIR)/_x_compile --report_dir $(BUILD_DIR)/reports_compile \
	    --hls.pre_tcl $(abspath hls_hbm_rd.tcl) -k $(KERNEL) -c -o $@ hbm_rd.cpp 2>&1 | tee $(BUILD_DIR)/$(KERNEL)_compile.log
xclbin: $(XCLBIN)
$(XCLBIN): $(XO) apply_kernel_clock.tcl report_clocks.tcl
	cd $(BUILD_DIR) && GDN_KERNEL_TARGET_MHZ=$(FREQ) $(VPP) -l $(VPP_FLAGS) $(VPP_LDFLAGS) --temp_dir _x_temp --report_dir reports \
	    -o $(KERNEL).xclbin $(abspath $<) 2>&1 | tee $(KERNEL)_link.log
host: $(HOST_BIN)
$(HOST_BIN): host_hbm_rd.cpp
	$(CXX) -O2 -std=c++17 -Wall $(HOST_INC) -o $@ host_hbm_rd.cpp $(HOST_LIB)
clean:
	rm -rf $(BUILD_DIR) $(HOST_BIN)
