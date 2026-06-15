// probe_alloc.cpp — isolate the on-card weight-bo std::bad_alloc.
// Loads the xclbin and allocates the three big weight bos incrementally
// (weight_data g1, shard0 g19, shard1 g20) with per-step prints, so we see
// exactly which allocation throws — without the multi-minute weight upload.
//
//   build: see Makefile note below / inline g++ command
//   run:   ./probe_alloc <xclbin> <weights.gdnw> <device_index>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <sys/stat.h>
#include "gdn_model.h"

static GDNWeightHeader read_header(const char *path, size_t *file_bytes) {
    FILE *f = std::fopen(path, "rb");
    if (!f) throw std::runtime_error("cannot open weights");
    struct stat st; stat(path, &st); *file_bytes = (size_t)st.st_size;
    GDNWeightHeader h{};
    if (std::fread(&h, 1, sizeof(h), f) != sizeof(h)) throw std::runtime_error("short header");
    std::fclose(f);
    return h;
}

static xrt::bo g_keep[8];
static int g_nkeep = 0;
static void try_alloc(const char *tag, xrt::device &d, size_t bytes, xrt::memory_group grp) {
    std::fprintf(stderr, "  alloc %-10s grp=%-2u  %.3f GiB ... ", tag, (unsigned)grp, bytes / 1073741824.0);
    std::fflush(stderr);
    try {
        xrt::bo bo(d, bytes, grp);              // same ctor as host.cpp
        std::fprintf(stderr, "OK\n");
        g_keep[g_nkeep++] = std::move(bo);      // hold the allocation for later steps
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        throw;
    }
}

int main(int argc, char **argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: %s <xclbin> <weights.gdnw> <dev>\n", argv[0]); return 2; }
    size_t fbytes = 0;
    GDNWeightHeader h = read_header(argv[1+1], &fbytes);
    size_t weight_floats = (fbytes - sizeof(GDNWeightHeader)) / 4;
    size_t shard_floats = gdn_weight_shard_floats(&h);
    size_t W = weight_floats * sizeof(float);
    size_t S = shard_floats * sizeof(float);
    std::fprintf(stderr, "weights=%.3f GiB  shard=%.3f GiB (x2)  banks(512MiB): W=%zu S=%zu\n",
                 W / 1073741824.0, S / 1073741824.0,
                 (W + 536870911) / 536870912, (S + 536870911) / 536870912);

    xrt::device d(std::stoi(argv[3]));
    auto uuid = d.load_xclbin(argv[1]);
    xrt::kernel k(d, uuid, "gdn_forward");
    std::fprintf(stderr, "group_ids: weight_data(1)=%d mm0(19)=%d mm1(20)=%d\n",
                 k.group_id(1), k.group_id(19), k.group_id(20));

    std::fprintf(stderr, "[1] weight_data only:\n");
    try_alloc("weight", d, W, k.group_id(1));
    std::fprintf(stderr, "[2] + shard0 (2 overlapping groups, == singleport):\n");
    try_alloc("shard0", d, S, k.group_id(19));
    std::fprintf(stderr, "[3] + shard1 (3 overlapping groups):\n");
    try_alloc("shard1", d, S, k.group_id(20));
    std::fprintf(stderr, "ALL THREE ALLOCATED OK\n");
    return 0;
}
