# ============================================================================
# GatedDeltaNet c_impl/ — host testbench + v++ build for Alveo U55C.
#
# Builds three classes of artifact:
#   1. The decode-only native testbench (`gdn_eval`).
#   2. The HLS kernel for U55C: gdn_forward.xo (compile) → gdn_forward.xclbin
#      (link). Targets are `xo` and `xclbin`; per-target output goes to
#      build.$(TARGET)/ so sw_emu/hw_emu/hw can coexist.
#   3. The XRT host program: host.exe — loads gdn_forward.xclbin onto the
#      board and runs a fixture through the hardware kernel.
#
# Modeled on cs356-project/matmul/{Makefile,pl/Makefile,host/Makefile},
# flattened into one file because c_impl/ keeps everything in one directory.
# ============================================================================

SHELL          := /bin/bash
.SHELLFLAGS    := -o pipefail -c

# ----------------------------------------------------------------------------
# Common host-toolchain config
# ----------------------------------------------------------------------------
CXX            ?= c++
# C++14 here matches the existing host testbench (gdn_eval etc.).
# -Wno-unknown-pragmas silences the Vitis HLS #pragma directives that the
# host gcc does not understand. -Wno-unused-label silences the HLS loop labels
# (pipeline/unroll targets) that are intentionally unused in the native build.
CXXFLAGS       ?= -O3 -std=c++14 -Wall -Wextra -pedantic -Wno-unknown-pragmas -Wno-unused-label

# Vitis HLS install location — provides hls_stream.h used by the systolic
# matmul kernel in gdn_model.cpp.
XILINX_HLS_INC ?= /tools/Xilinx/Vitis_HLS/2022.1/include


# ----------------------------------------------------------------------------
# Vitis / XRT install paths (override via env or `make XILINX_XRT=...`)
# ----------------------------------------------------------------------------
XILINX_VITIS   := /tools/Xilinx/Vitis/2022.2
XILINX_XRT     ?= /opt/xilinx/xrt
VPP            := $(XILINX_VITIS)/bin/v++
VITIS_SETTINGS := $(XILINX_VITIS)/settings64.sh


# ----------------------------------------------------------------------------
# v++ build knobs
# ----------------------------------------------------------------------------
# TARGET: sw_emu | hw_emu | hw
# The retained timing-friendly island design is synthesized at 150 MHz so HLS
# inserts the required pipeline boundaries, then linked at its demonstrated
# timing-closed 100 MHz operating point. HLS and link clocks remain independently
# overridable so an experiment cannot silently reuse an XO synthesized for a
# different schedule.
PLATFORM       ?= xilinx_u55c_gen3x16_xdma_3_202210_1
TARGET         ?= hw
FREQ           ?= 100
HLS_FREQ       ?= 150
LINK_FREQ      ?= $(FREQ)
JOBS           ?= 32
VPP_DEBUG      ?= 0

KERNEL         := gdn_forward
KERNEL_SRC     := gdn_model.cpp gdn_model.h
HLS_PRE_TCL    := $(abspath ./hls_gdn_forward.tcl)

# Include both clocks so changing only the HLS target cannot silently reuse an
# XO synthesized for a different period.
BUILD_DIR      := build.$(TARGET).gdn32.h$(HLS_FREQ).f$(LINK_FREQ).o$(JOBS)
XO             := $(BUILD_DIR)/$(KERNEL).xo
XCLBIN         := $(BUILD_DIR)/$(KERNEL).xclbin
HW_CFG_TEMPLATE ?= hw_f150_physical_islands.cfg
RESOLVED_HW_CFG := $(BUILD_DIR)/resolved_hw.cfg
HW_CFG          ?= $(RESOLVED_HW_CFG)

XSA            := $(strip $(patsubst %.xpfm,%,$(shell basename $(PLATFORM))))

VPP_FLAGS      := -t $(TARGET) --platform $(PLATFORM) --save-temps
VPP_FLAGS      += --optimize 2
ifeq ($(VPP_DEBUG),1)
VPP_FLAGS      += -g
endif
VPP_FLAGS      += --hls.jobs $(JOBS)
# Compile and link clocks are deliberately separate. The routed decode designs
# synthesize HLS at 150 MHz, then request the independently chosen physical
# DATA_CLK during link (100 MHz for the retained Iter57 image).
VPP_COMPILE_FLAGS := $(VPP_FLAGS) --kernel_frequency $(HLS_FREQ)
VPP_LINK_FLAGS    := $(VPP_FLAGS) --kernel_frequency $(LINK_FREQ)

VPP_LDFLAGS    := --config $(abspath $(HW_CFG))
VPP_LDFLAGS    += --vivado.synth.jobs $(JOBS) --vivado.impl.jobs $(JOBS)
# `-g` and --profile.data invoke Vitis debug/profiling automation. With this
# 32-master DFX design, the empty-monitor path crashes design initialization with
# "HADAFileSet::getSrcOptions() : NULL pointer". Keep both disabled for normal
# hardware builds. Host timing measures TPOT and throughput; set VPP_DEBUG=1
# only for a dedicated diagnostic build with explicit monitor points.
# VPP_LDFLAGS  += --profile.data all:all:all


# ----------------------------------------------------------------------------
# Host-program (XRT) build knobs
# ----------------------------------------------------------------------------
HOST_SRC       := host.cpp
HOST_BIN       := host.exe

HOST_INC       := -I$(XILINX_VITIS)/include -I$(XILINX_XRT)/include
# -Wl,-rpath bakes the XRT lib path into the binary so the runtime linker
# finds libxrt_*.so without needing `source /opt/xilinx/xrt/setup.sh`.
HOST_LIB       := -L$(XILINX_XRT)/lib -Wl,-rpath,$(XILINX_XRT)/lib \
                  -lxrt_core -lxrt_coreutil -lpthread -lrt -ldl -lstdc++


# ----------------------------------------------------------------------------
# On-card validation knobs (consumed by the `run_hw` target)
# ----------------------------------------------------------------------------
#   WEIGHTS   flat .gdnw weight blob (~5.6 GB for GDN-1.3B)
#   HW_DEVICE XRT device index (0 if multiple U55Cs are on the host)
WEIGHTS        ?= artifacts/gdn-1.3b-f32.gdnw
HW_DEVICE      ?= 0
DECODE_FIXTURE ?= fixtures_decode/decode.gdnreq
DECODE_STATE   ?= fixtures_decode/decode_ex0.gdnstate
DECODE_GOLDEN  ?= results_decode_golden/decode.decode.json
RUN_HW_DIR     ?= diagnostics/run_hw_h$(HLS_FREQ)_f$(LINK_FREQ)
ONCARD_DIR     ?= $(RUN_HW_DIR)/on_card


# ----------------------------------------------------------------------------
# Default target
# ----------------------------------------------------------------------------
.PHONY: all host_tb xo xclbin host run_hw clean clean_hw distclean help

all: host_tb

help:
	@echo "Targets:"
	@echo "  host_tb     decode-only native testbench: gdn_eval (default)"
	@echo "  xo          v++ compile  → $(XO)"
	@echo "  xclbin      v++ link     → $(XCLBIN)"
	@echo "  host        XRT host    → $(HOST_BIN)"
	@echo "  run_hw      build the current candidate, then run exact 8/64-token validation"
	@echo "  clean       remove host-tb binaries"
	@echo "  clean_hw    remove build.\$$(TARGET), _x/, v++ logs"
	@echo "  distclean   clean + clean_hw + remove host.exe"
	@echo ""
	@echo "Build knobs (override on cmdline, e.g. make TARGET=hw_emu):"
	@echo "  PLATFORM=$(PLATFORM)"
	@echo "  TARGET=$(TARGET)        (sw_emu | hw_emu | hw)"
	@echo "  FREQ=$(FREQ) MHz"
	@echo "  HLS_FREQ=$(HLS_FREQ) MHz"
	@echo "  LINK_FREQ=$(LINK_FREQ) MHz"
	@echo "  JOBS=$(JOBS)"
	@echo "  VPP_DEBUG=$(VPP_DEBUG)       (0 normally; 1 enables v++ -g)"
	@echo ""
	@echo "run_hw knobs:"
	@echo "  WEIGHTS=$(WEIGHTS)"
	@echo "  DECODE_FIXTURE=$(DECODE_FIXTURE)"
	@echo "  DECODE_STATE=$(DECODE_STATE)"
	@echo "  DECODE_GOLDEN=$(DECODE_GOLDEN)"
	@echo "  HW_DEVICE=$(HW_DEVICE)"
	@echo "  ONCARD_DIR=$(ONCARD_DIR)"


# ----------------------------------------------------------------------------
# C++ host testbenches
# ----------------------------------------------------------------------------
# Decode-only native testbench.
host_tb: gdn_eval

gdn_eval: gdn_eval.cpp gdn_model.cpp gdn_model.h
	$(CXX) $(CXXFLAGS) -I$(XILINX_HLS_INC) -o $@ gdn_eval.cpp gdn_model.cpp -lm

# ----------------------------------------------------------------------------
# v++ compile: gdn_model.cpp → $(KERNEL).xo
# ----------------------------------------------------------------------------
xo: $(XO)

$(RESOLVED_HW_CFG): $(HW_CFG_TEMPLATE) \
                    apply_f150_physical_islands.tcl \
                    apply_iter54_dma_timing.tcl \
                    apply_iter35_dma_w15_fifoaddr_fanout.tcl \
                    check_f150_physical_islands.tcl
	@mkdir -p $(@D)
	@sed 's|@C_IMPL_DIR@|$(CURDIR)|g' $< > $@
	@if grep -q '@C_IMPL_DIR@' $@; then \
	    echo "run_hw: unresolved path placeholder in $@" >&2; \
	    rm -f $@; \
	    exit 1; \
	fi

$(XO): $(KERNEL_SRC) $(HLS_PRE_TCL)
	@mkdir -p $(BUILD_DIR)
	@source '$(VITIS_SETTINGS)'; \
	$(VPP) $(VPP_COMPILE_FLAGS) \
	    --temp_dir $(BUILD_DIR)/_x_compile \
	    --report_dir $(BUILD_DIR)/reports_compile \
	    --hls.pre_tcl $(HLS_PRE_TCL) \
	    -k $(KERNEL) -c -o $@ $(filter %.cpp,$(KERNEL_SRC)) \
	    2>&1 | tee $(BUILD_DIR)/$(KERNEL)_compile.log


# ----------------------------------------------------------------------------
# v++ link: $(KERNEL).xo + hw.cfg → $(KERNEL).xclbin
# ----------------------------------------------------------------------------
xclbin: $(XCLBIN)

$(XCLBIN): $(XO) $(HW_CFG)
	@mkdir -p $(BUILD_DIR)
	@source '$(VITIS_SETTINGS)'; \
	cd $(BUILD_DIR) && $(VPP) -l $(VPP_LINK_FLAGS) $(VPP_LDFLAGS) \
	    --temp_dir _x_temp/ \
	    --report_dir reports/ \
	    -o $(KERNEL).xclbin \
	    $(abspath $<) \
	    2>&1 | tee $(KERNEL)_link.log


# ----------------------------------------------------------------------------
# XRT host program
# ----------------------------------------------------------------------------
host: $(HOST_BIN)

# Links gdn_model.cpp for the host-side weight-shard builder
# (gdn_build_weight_shards / gdn_weight_shard_floats). gdn_model.cpp needs the
# Vitis HLS include for hls_stream.h; -lm for the libm calls it pulls in.
$(HOST_BIN): $(HOST_SRC) gdn_model.cpp gdn_model.h
	$(CXX) -O2 -std=c++17 -Wall -Wno-unknown-pragmas -Wno-unused-label \
	    $(HOST_INC) -I$(XILINX_HLS_INC) -o $@ $(HOST_SRC) gdn_model.cpp $(HOST_LIB) -lm


# ----------------------------------------------------------------------------
# End-to-end hardware run
# ----------------------------------------------------------------------------
# Builds the .xo (if missing), links the .xclbin (if missing), builds the host
# program, then runs the exact 8-token smoke and 64-token measurement on the
# Alveo card. Make's normal prerequisites reuse a current bitstream; changing
# the kernel, physical configuration, or Tcl hooks rebuilds it automatically.
run_hw: $(XCLBIN) $(HOST_BIN)
	@set -euo pipefail; \
	test -s '$(XCLBIN)'; \
	mkdir -p '$(ONCARD_DIR)'; \
	for spec in '8 smoke8' '64 decode64'; do \
	    read -r decode_len stem <<< "$$spec"; \
	    output='$(ONCARD_DIR)/oncard_'"$${stem}"'.json'; \
	    log="$${output%.json}.log"; \
	    parity="$${output%.json}_parity.log"; \
	    stdbuf -oL -eL ./$(HOST_BIN) \
	        '$(XCLBIN)' \
	        '$(WEIGHTS)' \
	        '$(DECODE_FIXTURE)' \
	        "$$output" '$(HW_DEVICE)' \
	        --decode --decode-from-state '$(DECODE_STATE)' \
	        --decode-len "$$decode_len" 2>&1 | tee "$$log"; \
	    python3 ../scripts/check_gdn_c_parity.py \
	        --decode --golden '$(DECODE_GOLDEN)' --c "$$output" \
	        2>&1 | tee "$$parity"; \
	done; \
	output='$(ONCARD_DIR)/oncard_decode64.json'; \
	jq 'def median: sort as $$s | ($$s | length) as $$n | if $$n % 2 == 1 then $$s[($$n / 2 | floor)] else (($$s[$$n / 2 - 1] + $$s[$$n / 2]) / 2) end; [.examples[0].kernel_ms[] | select(. > 0)] as $$runs | {decode_len, measured_runs: ($$runs | length), min_kernel_ms: ($$runs | min), max_kernel_ms: ($$runs | max), median_kernel_ms: ($$runs | median), mean_kernel_ms: ($$runs | add / length), speedup_vs_iter39c_43_093ms: (43.093 / ($$runs | add / length)), speedup_vs_iter38e_47_079335ms: (47.079335 / ($$runs | add / length)), speedup_vs_8port_121_4ms: (121.4 / ($$runs | add / length))}' \
	    "$$output" | tee '$(ONCARD_DIR)/performance_summary.json'; \
	jq -e '.measured_runs == 63 and .mean_kernel_ms > 0' \
	    '$(ONCARD_DIR)/performance_summary.json' >/dev/null; \
	sha256sum $(HOST_BIN) '$(XCLBIN)' \
	    > '$(ONCARD_DIR)/artifact_hashes.txt'


# ----------------------------------------------------------------------------
# Clean
# ----------------------------------------------------------------------------
clean:
	rm -f gdn_eval

clean_hw:
	rm -rf $(BUILD_DIR) _x v++_*.log v++_*.jou .Xil

distclean: clean clean_hw
	rm -f $(HOST_BIN)
