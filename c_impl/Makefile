CXX ?= c++
# C++14; -Wno-unknown-pragmas because the Vitis HLS pragmas in the synthesis
# headers are not understood by host gcc.
CXXFLAGS ?= -O3 -std=c++14 -Wall -Wextra -pedantic -Wno-unknown-pragmas

all: gdn_eval gdn_attn_test gdn_matmul_test

gdn_eval: gdn_eval.cpp gdn_model.cpp gdn_model.h
	$(CXX) $(CXXFLAGS) -o $@ gdn_eval.cpp gdn_model.cpp -lm

gdn_attn_test: gdn_attn_test.cpp gdn_model.cpp gdn_model.h
	$(CXX) $(CXXFLAGS) -o $@ gdn_attn_test.cpp gdn_model.cpp -lm

gdn_matmul_test: gdn_matmul_test.cpp gdn_model.cpp gdn_model.h
	$(CXX) $(CXXFLAGS) -o $@ gdn_matmul_test.cpp gdn_model.cpp -lm

clean:
	rm -f gdn_eval gdn_attn_test gdn_matmul_test
