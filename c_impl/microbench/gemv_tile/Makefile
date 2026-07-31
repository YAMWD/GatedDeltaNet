# Routed 32-port mono-kernel GEMV microbenchmark for Alveo U55C.

SHELL       := /bin/bash
.SHELLFLAGS := -o pipefail -c

CXX          ?= c++
CXXFLAGS     ?= -O2 -std=c++14 -Wall -Wextra -Wno-unknown-pragmas -Wno-unused-label
XILINX_VITIS ?= /tools/Xilinx/Vitis/2022.1
XILINX_HLS   ?= /tools/Xilinx/Vitis_HLS/2022.1
XILINX_XRT   ?= /opt/xilinx/xrt
VPP          ?= $(XILINX_VITIS)/bin/v++
VIVADO       ?= /tools/Xilinx/Vivado/2022.1/bin/vivado

PLATFORM     ?= xilinx_u55c_gen3x16_xdma_3_202210_1
TARGET       ?= hw
FREQ         ?= 150
JOBS         ?= 8
READ_OUTSTANDING ?= 16
KERNEL       := gemv_full
PORTS        := 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
                16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31

BUILD_DIR    := build.$(TARGET).$(KERNEL).f$(FREQ).o$(READ_OUTSTANDING)
XO           := $(BUILD_DIR)/$(KERNEL).xo
XCLBIN       := $(BUILD_DIR)/$(KERNEL).xclbin
ROUTED_DCP   := $(BUILD_DIR)/_x_temp/link/vivado/vpl/prj/prj.runs/impl_1/level0_wrapper_routed.dcp
HOST_BIN     := host_gemv_full.exe
CSIM_BIN     := csim_full.exe

HOST_INC     := -I$(XILINX_VITIS)/include -I$(XILINX_XRT)/include
HOST_LIB     := -L$(XILINX_XRT)/lib -Wl,-rpath,$(XILINX_XRT)/lib \
                -lxrt_core -lxrt_coreutil -lpthread -lrt -ldl -lstdc++

VPP_FLAGS    := -t $(TARGET) --platform $(PLATFORM) --save-temps -g --optimize 2
VPP_FLAGS    += --hls.jobs $(JOBS) --kernel_frequency $(FREQ)
VPP_CFLAGS   := $(VPP_FLAGS) -D GEMV_READ_OUTSTANDING=$(READ_OUTSTANDING)
VPP_LDFLAGS  := --vivado.synth.jobs $(JOBS) --vivado.impl.jobs $(JOBS)
VPP_LDFLAGS  += --connectivity.nk $(KERNEL):1:$(KERNEL)_1
VPP_LDFLAGS  += $(foreach P,$(PORTS),--connectivity.sp $(KERNEL)_1.w$(P):HBM[$(P)])
VPP_LDFLAGS  += --connectivity.sp $(KERNEL)_1.x:HBM[0]
VPP_LDFLAGS  += --connectivity.sp $(KERNEL)_1.y:HBM[0]
VPP_LDFLAGS  += --vivado.prop run.impl_1.STEPS.OPT_DESIGN.TCL.PRE=$(abspath pblock_gemv_full.tcl)
VPP_LDFLAGS  += --vivado.prop run.impl_1.STEPS.PLACE_DESIGN.TCL.PRE=$(abspath place_gemv_full.tcl)
VPP_LDFLAGS  += --vivado.prop run.impl_1.STEPS.PLACE_DESIGN.ARGS.DIRECTIVE=SSI_SpreadSLLs
VPP_LDFLAGS  += --vivado.prop run.impl_1.STEPS.PHYS_OPT_DESIGN.IS_ENABLED=true
VPP_LDFLAGS  += --vivado.prop run.impl_1.STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE=AggressiveExplore
VPP_LDFLAGS  += --vivado.prop run.impl_1.STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE=Explore

ROWS         ?= 2048
KPACKS       ?= 352
DEVICE       ?= 0
VERIFY       ?= 1
RUN_FREQ     ?= 130.6
TIMED_REPS   ?= 9
WEIGHTS      ?= ../../artifacts/gdn-1.3b-f32.gdnw

.PHONY: all help csim_full host_full xo_full xclbin_full run_full run_layer0 analyze clean distclean

all: csim_full host_full

help:
	@echo "Targets: csim_full host_full xo_full xclbin_full run_full run_layer0 analyze clean"
	@echo "Hardware knobs: TARGET=$(TARGET) FREQ=$(FREQ) READ_OUTSTANDING=$(READ_OUTSTANDING) JOBS=$(JOBS)"
	@echo "Run knobs: ROWS=$(ROWS) KPACKS=$(KPACKS) DEVICE=$(DEVICE) RUN_FREQ=$(RUN_FREQ) TIMED_REPS=$(TIMED_REPS)"

csim_full: $(CSIM_BIN)
	./$(CSIM_BIN) 64 16
	./$(CSIM_BIN) 128 128
	./$(CSIM_BIN) 64 64
	./$(CSIM_BIN) 64 128

$(CSIM_BIN): gemv_full_csim.cpp gemv_full.cpp gemv_full.h
	$(CXX) -O2 -std=c++14 -I$(XILINX_HLS)/include gemv_full_csim.cpp gemv_full.cpp -o $@

host_full: $(HOST_BIN)

$(HOST_BIN): host_gemv_full.cpp gemv_full.h ../../gdn_model.h
	$(CXX) $(CXXFLAGS) $(HOST_INC) -o $@ host_gemv_full.cpp $(HOST_LIB)

xo_full: $(XO)

$(XO): gemv_full.cpp gemv_full.h hls_gemv_full.tcl
	@mkdir -p $(BUILD_DIR)
	$(VPP) $(VPP_CFLAGS) --temp_dir $(BUILD_DIR)/_x_compile \
	    --report_dir $(BUILD_DIR)/reports_compile \
	    --hls.pre_tcl $(abspath hls_gemv_full.tcl) \
	    -k $(KERNEL) -c -o $@ gemv_full.cpp \
	    2>&1 | tee $(BUILD_DIR)/$(KERNEL)_compile.log

xclbin_full: $(XCLBIN)

$(XCLBIN): $(XO) pblock_gemv_full.tcl place_gemv_full.tcl
	cd $(BUILD_DIR) && $(VPP) -l $(VPP_FLAGS) $(VPP_LDFLAGS) \
	    --temp_dir _x_temp --report_dir reports \
	    -o $(KERNEL).xclbin $(abspath $<) \
	    2>&1 | tee $(KERNEL)_link.log

run_full: $(XCLBIN) $(HOST_BIN)
	./$(HOST_BIN) $(XCLBIN) $(KERNEL) $(ROWS) $(KPACKS) $(DEVICE) \
	    $(VERIFY) $(RUN_FREQ) $(TIMED_REPS)

run_layer0: $(XCLBIN) $(HOST_BIN)
	./$(HOST_BIN) $(XCLBIN) $(KERNEL) 64 128 $(DEVICE) \
	    $(VERIFY) $(RUN_FREQ) 101 $(WEIGHTS)

analyze: $(ROUTED_DCP) analyze_routed_checkpoint.tcl
	$(VIVADO) -mode batch -notrace -source analyze_routed_checkpoint.tcl \
	    -tclargs $(ROUTED_DCP) $(BUILD_DIR)/postroute_analysis

clean:
	rm -rf $(BUILD_DIR) $(HOST_BIN) $(CSIM_BIN)

distclean:
	rm -rf build.* _x* .Xil *.exe *.log *.jou *.pb
