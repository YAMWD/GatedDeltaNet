# ============================================================================
# GatedDeltaNet c_impl/ — host testbench + v++ build for Alveo U55C.
#
# Builds three classes of artifact:
#   1. Host-side C++ testbenches that drive the HLS source directly through
#      its csim entry points (gdn_eval, gdn_attn_test, gdn_matmul_test).
#      These are what test_parity.sh runs.
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
XILINX_VITIS   ?= /tools/Xilinx/Vitis/2022.1
XILINX_XRT     ?= /opt/xilinx/xrt
VPP            ?= $(XILINX_VITIS)/bin/v++


# ----------------------------------------------------------------------------
# v++ build knobs
# ----------------------------------------------------------------------------
# TARGET: sw_emu | hw_emu | hw
# FREQ:   kernel clock in MHz. Default 100 MHz to match the HLS source's
#         10 ns synthesis target. Without this, gen3x16 platforms (U55C) would
#         use the platform default of 300 MHz, leaving WNS ≈ -8.7 ns in the
#         router and making the design effectively unroutable.
PLATFORM       ?= xilinx_u55c_gen3x16_xdma_3_202210_1
TARGET         ?= hw
FREQ           ?= 130
HLS_FREQ       ?= $(FREQ)
LINK_FREQ      ?= $(FREQ)
JOBS           ?= 8
VPP_DEBUG      ?= 0

KERNEL         := gdn_forward
KERNEL_SRC     := gdn_model.cpp gdn_model.h
HW_CFG         := $(abspath ./hw.cfg)
HLS_PRE_TCL    := $(abspath ./hls_gdn_forward.tcl)

# Include both clocks so changing only the HLS target cannot silently reuse an
# XO synthesized for a different period.
BUILD_DIR      := build.$(TARGET).gdn32.h$(HLS_FREQ).f$(LINK_FREQ).o8
XO             := $(BUILD_DIR)/$(KERNEL).xo
XCLBIN         := $(BUILD_DIR)/$(KERNEL).xclbin

XSA            := $(strip $(patsubst %.xpfm,%,$(shell basename $(PLATFORM))))

VPP_FLAGS      := -t $(TARGET) --platform $(PLATFORM) --save-temps
VPP_FLAGS      += --optimize 2
ifeq ($(VPP_DEBUG),1)
VPP_FLAGS      += -g
endif
VPP_FLAGS      += --hls.jobs $(JOBS)
# Compile and link clocks are deliberately separate. The routed decode designs
# synthesize HLS at 130 MHz, then request the independently chosen physical
# DATA_CLK during link (100 MHz for Iter38). Both default to legacy FREQ so old
# one-line commands retain their behavior.
VPP_COMPILE_FLAGS := $(VPP_FLAGS) --kernel_frequency $(HLS_FREQ)
VPP_LINK_FLAGS    := $(VPP_FLAGS) --kernel_frequency $(LINK_FREQ)

VPP_LDFLAGS    := --config $(HW_CFG)
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
# On-card run knobs (consumed by the `run_hw` target)
# ----------------------------------------------------------------------------
#   WEIGHTS   flat .gdnw weight blob (~5.6 GB for GDN-1.3B)
#   FIXTURE   pretokenised .gdnreq fixture to score
#   HW_OUT    JSON result file (defaults to hw_<fixture-stem>.json)
#   HW_DEVICE XRT device index (0 if multiple U55Cs are on the host)
#   HW_MAX_EX optional cap on examples processed; leave empty for "all"
WEIGHTS        ?= artifacts/gdn-1.3b-f32.gdnw
FIXTURE        ?= fixtures_smoke/wikitext.gdnreq
HW_OUT         ?= hw_$(basename $(notdir $(FIXTURE))).json
HW_DEVICE      ?= 0
HW_MAX_EX      ?=
ITER37_FREQ    ?= 115
ITER38_HLS_FREQ  ?= 130
ITER38_LINK_FREQ ?= 100
ITER38_BUILD_DIR ?= build.hw.iter38e.qkvg.gu.state4packed.f$(ITER38_LINK_FREQ).o8.v2022_2
ITER38_DIAGNOSTICS_DIR ?= diagnostics/iter38e_qkvg_gu_state4packed_f$(ITER38_LINK_FREQ)
ITER38_CFG_TEMPLATE := hw_iter37c_state4_recur32_slr2_f115.cfg
ITER38_RESOLVED_CFG := $(ITER38_DIAGNOSTICS_DIR)/resolved_hw.cfg
ITER38_VITIS_SETTINGS ?= /tools/Xilinx/Vitis/2022.2/settings64.sh
ITER38_IP_CACHE ?= build.hw.iter35.recur16.dmaw15f64.postphys.f100.o8.v2022_2/.ipcache
ITER38_XCLBIN := $(ITER38_BUILD_DIR)/gdn_forward.xclbin
ITER38_ONCARD_DIR := $(ITER38_DIAGNOSTICS_DIR)/on_card
ITER38_DECODE_WEIGHTS ?= artifacts/gdn-1.3b-f32.gdnw
ITER38_DECODE_FIXTURE ?= fixtures_decode/decode.gdnreq
ITER38_DECODE_STATE ?= fixtures_decode/decode_ex0.gdnstate
ITER38_DECODE_GOLDEN ?= results_decode_golden/decode.decode.json


# ----------------------------------------------------------------------------
# Default target
# ----------------------------------------------------------------------------
.PHONY: all host_tb xo xclbin host run_hw iter37 iter38 iter38-verify iter38-oncard reproduce_iter24_success clean clean_hw distclean help

all: host_tb

help:
	@echo "Targets:"
	@echo "  host_tb     C++ host testbenches: gdn_eval gdn_attn_test gdn_matmul_test (default)"
	@echo "  xo          v++ compile  → $(XO)"
	@echo "  xclbin      v++ link     → $(XCLBIN)"
	@echo "  host        XRT host    → $(HOST_BIN)"
	@echo "  run_hw      xo + xclbin + host + ./host.exe ... (full hardware run)"
	@echo "  iter37      reproduce the routed four-port/32-lane recurrent design"
	@echo "  iter38      build packed concurrent-state + unified QKVG/GU at 100 MHz"
	@echo "  iter38-oncard  exact 8-token smoke + 64-token measurement of Iter38"
	@echo "  reproduce_iter24_success  rebuild the correct, routable iter24b XCLBIN"
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
	@echo "run_hw knobs (override on cmdline, e.g. make run_hw FIXTURE=fixtures_smoke/piqa.gdnreq):"
	@echo "  WEIGHTS=$(WEIGHTS)"
	@echo "  FIXTURE=$(FIXTURE)"
	@echo "  HW_OUT=$(HW_OUT)"
	@echo "  HW_DEVICE=$(HW_DEVICE)"
	@echo "  HW_MAX_EX=$(if $(HW_MAX_EX),$(HW_MAX_EX),(all))"
	@echo "  ITER37_FREQ=$(ITER37_FREQ) MHz (100 or 115; default 115)"


# ----------------------------------------------------------------------------
# C++ host testbenches
# ----------------------------------------------------------------------------
# Decode-only build: gdn_eval is the native csim of the decode-only GEMV kernel.
# The old gdn_attn_test / gdn_matmul_test exercised the removed prefill /
# GEMM-matmul tops (gdn_attn_forward, gdn_matmul_top) and are retired.
host_tb: gdn_eval

gdn_eval: gdn_eval.cpp gdn_model.cpp gdn_model.h
	$(CXX) $(CXXFLAGS) -I$(XILINX_HLS_INC) -o $@ gdn_eval.cpp gdn_model.cpp -lm

gdn_attn_test: gdn_attn_test.cpp gdn_model.cpp gdn_model.h
	$(CXX) $(CXXFLAGS) -I$(XILINX_HLS_INC) -o $@ gdn_attn_test.cpp gdn_model.cpp -lm

# gdn_matmul_test exercises the systolic matmul through the gdn_matmul_top
# HLS entry in gdn_model.cpp, comparing against the native-C naive_matmul
# reference in gdn_matmul_test.cpp. Same translation unit as the rest of
# the model — no separate matmul source file.
gdn_matmul_test: gdn_matmul_test.cpp gdn_model.cpp gdn_model.h
	$(CXX) $(CXXFLAGS) -I$(XILINX_HLS_INC) -o $@ \
	    gdn_matmul_test.cpp gdn_model.cpp -lm


# ----------------------------------------------------------------------------
# v++ compile: gdn_model.cpp → $(KERNEL).xo
# ----------------------------------------------------------------------------
xo: $(XO)

$(XO): $(KERNEL_SRC) $(HLS_PRE_TCL)
	@mkdir -p $(BUILD_DIR)
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
# Builds the .xo (if missing), links the .xclbin (if missing), builds the
# host program (if missing), then invokes host.exe to score $(FIXTURE) on the
# Alveo card. Reuses cached build artefacts when the prerequisite files have
# not changed, so a re-run after only editing $(FIXTURE) skips straight to
# the on-card invocation.
run_hw: $(XCLBIN) $(HOST_BIN)
	@echo "=== Running on hardware ==="
	@echo "  xclbin : $(XCLBIN)"
	@echo "  weights: $(WEIGHTS)"
	@echo "  fixture: $(FIXTURE)"
	@echo "  output : $(HW_OUT)"
	@echo "  device : $(HW_DEVICE)"
	@echo "  max_ex : $(if $(HW_MAX_EX),$(HW_MAX_EX),(all))"
	./$(HOST_BIN) $(XCLBIN) $(WEIGHTS) $(FIXTURE) $(HW_OUT) $(HW_DEVICE) $(HW_MAX_EX)


# ----------------------------------------------------------------------------
# Iter37 production build
# ----------------------------------------------------------------------------
# The checksum-guarded launcher reuses the verified XO when available and
# regenerates only relocatable path resolution in diagnostics.
iter37:
	bash $(CURDIR)/build_iter37_state4_recur32.sh $(ITER37_FREQ)


# ----------------------------------------------------------------------------
# Iter38 current optimization build
# ----------------------------------------------------------------------------
# Keep the long build behind one Make target instead of cloning another
# iteration-specific launcher. The committed config remains relocatable; this
# generated copy resolves only the checkout path used by Vivado's Tcl hooks.
$(ITER38_RESOLVED_CFG): $(ITER38_CFG_TEMPLATE) \
                        apply_iter37c_recurrent_slr2.tcl \
                        apply_iter35_dma_w15_fifoaddr_fanout.tcl \
                        apply_iter22_cluster8_slr1_east.tcl \
                        apply_iter23_dma_fanout.tcl
	@mkdir -p $(@D)
	@sed 's|@C_IMPL_DIR@|$(CURDIR)|g' $< > $@
	@if grep -q '@C_IMPL_DIR@' $@; then \
	    echo "iter38: unresolved path placeholder in $@" >&2; \
	    rm -f $@; \
	    exit 1; \
	fi

iter38-verify:
	@echo '0791d1d158dc476e7f2cc1e44b6bf7790038ea8bf0458138ea2e015ca7184708  gdn_model.cpp' | sha256sum -c -
	@echo '7bf6bf9d567241028d916a5def4924c6efb7031e416e573921c11d4b10fa2e74  gdn_model.h' | sha256sum -c -
	@echo 'a76930f3332baac50bf7540b58f9242a2efcd235154215b26e95e8da5a057633  hls_gdn_forward.tcl' | sha256sum -c -
	@echo '998b71e3a8cb3b7f818f12cbe6581f0ffd2e04010dba5db3f20ca2ae844aa08f  $(ITER38_CFG_TEMPLATE)' | sha256sum -c -
	@echo '15587403b5e904345abdee72cd84cfc0fa24f8be559f94f1e5554cdcedbd059e  apply_iter37c_recurrent_slr2.tcl' | sha256sum -c -
	@echo 'd6cd2074b8bd987cf00db922c390c3c92ecd48947108d4d1b54a46e1445263bb  apply_iter35_dma_w15_fifoaddr_fanout.tcl' | sha256sum -c -
	@echo 'b0f07d2128589789641c649d8c127949dd06ca72f6f3140154eda5e24c331b1f  apply_iter22_cluster8_slr1_east.tcl' | sha256sum -c -
	@echo '268f9e9fb8e0e6178317edd88be46811372351251758dc781ba950e84f4ba4d8  apply_iter23_dma_fanout.tcl' | sha256sum -c -

iter38: iter38-verify $(ITER38_RESOLVED_CFG)
	+@set -euo pipefail; \
	test -r '$(ITER38_VITIS_SETTINGS)'; \
	source '$(ITER38_VITIS_SETTINGS)'; \
	mkdir -p '$(ITER38_BUILD_DIR)'; \
	if [[ -d '$(ITER38_IP_CACHE)' && ! -d '$(ITER38_BUILD_DIR)/.ipcache' ]]; then \
	    mkdir -p '$(ITER38_BUILD_DIR)/.ipcache'; \
	    cp -a --reflink=auto '$(ITER38_IP_CACHE)/.' '$(ITER38_BUILD_DIR)/.ipcache/'; \
	fi; \
	manifest='$(ITER38_DIAGNOSTICS_DIR)/build.manifest'; \
	{ \
	    printf 'iteration=iter38e_qkvg_gu_state4packed_f%s\n' '$(ITER38_LINK_FREQ)'; \
	    printf 'hls_frequency_mhz=%s\n' '$(ITER38_HLS_FREQ)'; \
	    printf 'link_frequency_mhz=%s\n' '$(ITER38_LINK_FREQ)'; \
	    printf 'started_at=%s\n' "$$(date --iso-8601=seconds)"; \
	    sha256sum gdn_model.cpp gdn_model.h hls_gdn_forward.tcl '$(ITER38_RESOLVED_CFG)'; \
	} > "$$manifest"; \
	$(MAKE) xclbin \
	    TARGET=hw \
	    FREQ='$(ITER38_LINK_FREQ)' \
	    HLS_FREQ='$(ITER38_HLS_FREQ)' \
	    LINK_FREQ='$(ITER38_LINK_FREQ)' \
	    XILINX_VITIS=/tools/Xilinx/Vitis/2022.2 \
	    VPP=/tools/Xilinx/Vitis/2022.2/bin/v++ \
	    BUILD_DIR='$(ITER38_BUILD_DIR)' \
	    HW_CFG='$(abspath $(ITER38_RESOLVED_CFG))' \
	    JOBS='$(JOBS)'; \
	sha256sum '$(ITER38_BUILD_DIR)/gdn_forward.xo' \
	          '$(ITER38_BUILD_DIR)/gdn_forward.xclbin' >> "$$manifest"; \
	printf 'completed_at=%s\n' "$$(date --iso-8601=seconds)" >> "$$manifest"

iter38-oncard: host
	@set -euo pipefail; \
	test -s '$(ITER38_XCLBIN)'; \
	mkdir -p '$(ITER38_ONCARD_DIR)'; \
	for spec in '8 smoke8' '64 decode64'; do \
	    read -r decode_len stem <<< "$$spec"; \
	    output='$(ITER38_ONCARD_DIR)/oncard_iter38_'; \
	    output="$${output}$${stem}.json"; \
	    log="$${output%.json}.log"; \
	    parity="$${output%.json}_parity.log"; \
	    stdbuf -oL -eL ./$(HOST_BIN) \
	        '$(ITER38_XCLBIN)' \
	        '$(ITER38_DECODE_WEIGHTS)' \
	        '$(ITER38_DECODE_FIXTURE)' \
	        "$$output" 0 1 \
	        --decode --decode-from-state '$(ITER38_DECODE_STATE)' \
	        --decode-len "$$decode_len" 2>&1 | tee "$$log"; \
	    python3 ../scripts/check_gdn_c_parity.py \
	        --decode --golden '$(ITER38_DECODE_GOLDEN)' --c "$$output" \
	        2>&1 | tee "$$parity"; \
	done; \
	output='$(ITER38_ONCARD_DIR)/oncard_iter38_decode64.json'; \
	jq 'def median: sort as $$s | ($$s | length) as $$n | if $$n % 2 == 1 then $$s[($$n / 2 | floor)] else (($$s[$$n / 2 - 1] + $$s[$$n / 2]) / 2) end; [.examples[0].kernel_ms[] | select(. > 0)] as $$runs | {decode_len, measured_runs: ($$runs | length), min_kernel_ms: ($$runs | min), max_kernel_ms: ($$runs | max), median_kernel_ms: ($$runs | median), mean_kernel_ms: ($$runs | add / length), speedup_vs_iter37d_51_450918ms: (51.450918 / ($$runs | add / length)), speedup_vs_8port_121_4ms: (121.4 / ($$runs | add / length))}' \
	    "$$output" | tee '$(ITER38_ONCARD_DIR)/performance_summary.json'; \
	jq -e '.measured_runs == 63 and .mean_kernel_ms > 0' \
	    '$(ITER38_ONCARD_DIR)/performance_summary.json' >/dev/null; \
	sha256sum $(HOST_BIN) '$(ITER38_XCLBIN)' \
	    > '$(ITER38_ONCARD_DIR)/artifact_hashes.txt'


# ----------------------------------------------------------------------------
# Preserved successful implementation
# ----------------------------------------------------------------------------
# This target deliberately uses a dedicated, checksum-guarded recipe rather
# than the mutable default hardware knobs above. It rebuilds both the XO and
# XCLBIN for the correct, routable iter24b physical result.
reproduce_iter24_success:
	bash $(CURDIR)/reproduce_iter24_success.sh


# ----------------------------------------------------------------------------
# Clean
# ----------------------------------------------------------------------------
clean:
	rm -f gdn_eval gdn_attn_test gdn_matmul_test

clean_hw:
	rm -rf $(BUILD_DIR) _x v++_*.log v++_*.jou .Xil

distclean: clean clean_hw
	rm -f $(HOST_BIN)
