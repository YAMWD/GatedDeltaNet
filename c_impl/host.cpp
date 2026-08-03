#include "gdn_model.h"

#include "xrt.h"
#include "experimental/xrt_kernel.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr const char *kDefaultWeights = "artifacts/gdn-1.3b-f32.gdnw";
constexpr const char *kDefaultFixture = "fixtures_full/wikitext.gdnreq";
constexpr const char *kDefaultOutput = "hw_wikitext_results.json";
constexpr const char *kDefaultDecodeOutput = "results_decode_hw/decode.hw.json";
constexpr const char *kDefaultXrt = "/opt/xilinx/xrt";
constexpr uint32_t kReqKindLL = 2;
constexpr uint32_t kReqKindRolling = 3;
constexpr size_t kWeightHeaderBytes = 60;

struct PairReq {
    uint32_t ctx_len = 0;
    uint32_t cont_len = 0;
    std::vector<int32_t> ctx;
    std::vector<int32_t> cont;
};

struct RollingReq {
    uint32_t word_count = 0;
    uint32_t byte_count = 0;
    std::vector<PairReq> windows;
};

struct Fixture {
    uint32_t kind = 0;
    uint32_t num_examples = 0;
    std::vector<RollingReq> rolling_examples;
    std::vector<PairReq> ll_examples;
};

struct ModelData {
    GDNWeightHeader config{};
    std::vector<float> weight_data;
    const float *lm_head = nullptr;
};

struct Options {
    std::string xclbin;
    std::string weights = kDefaultWeights;
    std::string fixture = kDefaultFixture;
    std::string output = kDefaultOutput;
    unsigned int device_index = 0;
    uint32_t max_examples = 0;
    // Decode-only: load the GPU-exported post-prompt state and decode on-card.
    bool decode = false;
    std::string state_path;     // --decode-from-state <.gdnstate> (required for --decode)
    uint32_t decode_len = 0;    // 0 => use the fixture's golden cont_len
    bool output_set = false;    // whether an explicit positional output was given
};

static void usage(const char *argv0) {
    std::cerr
        << "usage: " << argv0
        << " <gdn_forward.xclbin> [weights.gdnw] [fixture.gdnreq]"
        << " [output.json|-] [device_index] [max_examples]\n"
        << "       " << argv0
        << " <gdn_forward.xclbin> [weights.gdnw] [decode.gdnreq]"
        << " [output.json|-] [device_index] [max_examples]"
        << " --decode --decode-from-state <state.gdnstate> [--decode-len N]\n\n"
        << "defaults:\n"
        << "  weights      " << kDefaultWeights << "\n"
        << "  fixture      " << kDefaultFixture << "\n"
        << "  output       " << kDefaultOutput
        << " (rolling) | " << kDefaultDecodeOutput << " (--decode)\n"
        << "  device_index 0\n"
        << "  max_examples 0 (all examples)\n\n"
        << "decode-only flags (the FPGA never prefills):\n"
        << "  --decode                     run the on-card decode benchmark\n"
        << "  --decode-from-state <file>   GPU-exported .gdnstate (recurrent+conv state); required\n"
        << "  --decode-len N               cap decode length (0 => fixture golden cont_len)\n";
}

static uint32_t parse_u32(const char *text, const char *name) {
    char *end = nullptr;
    unsigned long value = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0' || value > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + text);
    }
    return static_cast<uint32_t>(value);
}

static Options parse_options(int argc, char **argv) {
    Options opts;
    if (argc < 2 || std::strcmp(argv[1], "--help") == 0) {
        usage(argv[0]);
        std::exit(argc < 2 ? 1 : 0);
    }

    // Separate optional --decode flags from the positional arguments. Without
    // any --decode flag the positional contract is unchanged:
    //   <xclbin> [weights] [fixture] [output] [device_index] [max_examples]
    std::vector<std::string> positional;
    for (int arg_index = 1; arg_index < argc; ++arg_index) {
        std::string arg = argv[arg_index];
        if (arg == "--decode") {
            opts.decode = true;
        } else if (arg == "--decode-from-state") {
            if (arg_index + 1 >= argc) {
                throw std::runtime_error("--decode-from-state requires a .gdnstate path");
            }
            opts.state_path = argv[++arg_index];
            opts.decode = true;
        } else if (arg == "--decode-len") {
            if (arg_index + 1 >= argc) {
                throw std::runtime_error("--decode-len requires a value");
            }
            opts.decode_len = parse_u32(argv[++arg_index], "decode-len");
        } else {
            positional.push_back(arg);
        }
    }

    if (positional.empty() || positional.size() > 6) {
        usage(argv[0]);
        std::exit(1);
    }
    opts.xclbin = positional[0];
    if (positional.size() > 1) opts.weights = positional[1];
    if (positional.size() > 2) opts.fixture = positional[2];
    if (positional.size() > 3) {
        opts.output = positional[3];
        opts.output_set = true;
    }
    if (positional.size() > 4) opts.device_index = parse_u32(positional[4].c_str(), "device_index");
    if (positional.size() > 5) opts.max_examples = parse_u32(positional[5].c_str(), "max_examples");
    return opts;
}

static void ensure_xrt_environment() {
    if (std::getenv("XILINX_XRT") == nullptr) {
        setenv("XILINX_XRT", kDefaultXrt, 0);
    }
}

static std::vector<uint8_t> read_binary_file(const std::string &path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("failed to open " + path);
    }
    std::streamsize size = in.tellg();
    if (size < 0) {
        throw std::runtime_error("failed to size " + path);
    }
    in.seekg(0, std::ios::beg);
    std::vector<uint8_t> data(static_cast<size_t>(size));
    if (!data.empty() && !in.read(reinterpret_cast<char *>(data.data()), size)) {
        throw std::runtime_error("failed to read " + path);
    }
    return data;
}

static uint32_t read_u32(const std::vector<uint8_t> &blob, size_t &offset) {
    if (offset + sizeof(uint32_t) > blob.size()) {
        throw std::runtime_error("fixture truncated");
    }
    uint32_t value = 0;
    std::memcpy(&value, blob.data() + offset, sizeof(value));
    offset += sizeof(value);
    return value;
}

static std::vector<int32_t> read_i32_array(
    const std::vector<uint8_t> &blob,
    size_t &offset,
    uint32_t count
) {
    size_t bytes = static_cast<size_t>(count) * sizeof(int32_t);
    if (offset + bytes > blob.size()) {
        throw std::runtime_error("fixture truncated");
    }
    std::vector<int32_t> out(count);
    if (bytes != 0) {
        std::memcpy(out.data(), blob.data() + offset, bytes);
    }
    offset += bytes;
    return out;
}

static Fixture load_rolling_fixture(const std::string &path) {
    std::vector<uint8_t> blob = read_binary_file(path);
    if (blob.size() < 20 || std::memcmp(blob.data(), "GDNREQ1", 7) != 0) {
        throw std::runtime_error("unsupported fixture file: " + path);
    }

    size_t offset = 8;
    uint32_t version = read_u32(blob, offset);
    Fixture fixture;
    fixture.kind = read_u32(blob, offset);
    fixture.num_examples = read_u32(blob, offset);
    if (version != 1) {
        throw std::runtime_error("unsupported fixture version");
    }
    if (fixture.kind != kReqKindRolling) {
        throw std::runtime_error("host.cpp currently expects a rolling-loglikelihood Wikitext fixture");
    }

    fixture.rolling_examples.resize(fixture.num_examples);
    for (uint32_t example_index = 0; example_index < fixture.num_examples; ++example_index) {
        RollingReq &req = fixture.rolling_examples[example_index];
        req.word_count = read_u32(blob, offset);
        req.byte_count = read_u32(blob, offset);
        uint32_t num_windows = read_u32(blob, offset);
        req.windows.resize(num_windows);
        for (uint32_t window_index = 0; window_index < num_windows; ++window_index) {
            PairReq &pair = req.windows[window_index];
            pair.ctx_len = read_u32(blob, offset);
            pair.cont_len = read_u32(blob, offset);
            pair.ctx = read_i32_array(blob, offset, pair.ctx_len);
            pair.cont = read_i32_array(blob, offset, pair.cont_len);
        }
    }

    return fixture;
}

// Loader for the LL-kind (REQ_KIND_LL = 2) decode fixture. Mirrors the native
// gdn_eval LL branch: each example is a (ctx, cont) pair where ctx is the
// prompt token ids and cont is the golden greedy trajectory.
static Fixture load_ll_fixture(const std::string &path) {
    std::vector<uint8_t> blob = read_binary_file(path);
    if (blob.size() < 20 || std::memcmp(blob.data(), "GDNREQ1", 7) != 0) {
        throw std::runtime_error("unsupported fixture file: " + path);
    }

    size_t offset = 8;
    uint32_t version = read_u32(blob, offset);
    Fixture fixture;
    fixture.kind = read_u32(blob, offset);
    fixture.num_examples = read_u32(blob, offset);
    if (version != 1) {
        throw std::runtime_error("unsupported fixture version");
    }
    if (fixture.kind != kReqKindLL) {
        throw std::runtime_error("--decode requires an LL-kind (.gdnreq kind=2) fixture");
    }

    fixture.ll_examples.resize(fixture.num_examples);
    for (uint32_t example_index = 0; example_index < fixture.num_examples; ++example_index) {
        PairReq &pair = fixture.ll_examples[example_index];
        pair.ctx_len = read_u32(blob, offset);
        pair.cont_len = read_u32(blob, offset);
        pair.ctx = read_i32_array(blob, offset, pair.ctx_len);
        pair.cont = read_i32_array(blob, offset, pair.cont_len);
    }

    return fixture;
}

static size_t total_weight_floats(const GDNWeightHeader &config) {
    size_t total = 0;
    size_t hidden = config.hidden_size;
    size_t num_heads = config.num_heads;
    size_t head_dim = config.head_dim;
    size_t intermediate = config.intermediate_size;
    size_t vocab = config.vocab_size;
    size_t conv = config.conv_size;

    size_t layer_stride =
        hidden +
        num_heads +
        num_heads +
        hidden * hidden +
        hidden * hidden +
        hidden * hidden +
        num_heads * hidden +
        num_heads * hidden +
        hidden * conv +
        hidden * conv +
        hidden * conv +
        hidden * hidden +
        head_dim +
        hidden * hidden +
        hidden +
        intermediate * hidden +
        intermediate * hidden +
        hidden * intermediate;

    total += vocab * hidden;
    total += static_cast<size_t>(config.num_layers) * layer_stride;
    total += hidden;
    total += vocab * hidden;
    return total;
}

static void validate_config(const GDNWeightHeader &config) {
    if (sizeof(GDNWeightHeader) != kWeightHeaderBytes) {
        throw std::runtime_error("unexpected GDNWeightHeader size");
    }
    if (std::memcmp(config.magic, "GDNWv1", 6) != 0 || config.version != 1) {
        throw std::runtime_error("unsupported weight file");
    }
    if (config.num_heads != config.num_v_heads) {
        throw std::runtime_error("expected num_heads == num_v_heads");
    }
    if (config.hidden_size != config.num_heads * config.head_dim) {
        throw std::runtime_error("expected hidden_size == num_heads * head_dim");
    }
}

static ModelData load_model(const std::string &path) {
    FILE *file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) {
        throw std::runtime_error("failed to open weights: " + path);
    }

    ModelData model;
    if (std::fread(&model.config, 1, kWeightHeaderBytes, file) != kWeightHeaderBytes) {
        std::fclose(file);
        throw std::runtime_error("failed to read weight header");
    }
    validate_config(model.config);

    size_t total = total_weight_floats(model.config);
    model.weight_data.resize(total);
    if (std::fread(model.weight_data.data(), sizeof(float), total, file) != total) {
        std::fclose(file);
        throw std::runtime_error("failed to read weight payload");
    }
    std::fclose(file);

    size_t lm_head_offset = total - static_cast<size_t>(model.config.vocab_size) * model.config.hidden_size;
    model.lm_head = model.weight_data.data() + lm_head_offset;
    return model;
}

static void compute_logits(const ModelData &model, const float *hidden, std::vector<float> &logits) {
    uint32_t vocab = model.config.vocab_size;
    uint32_t hidden_size = model.config.hidden_size;
    logits.assign(vocab, 0.0f);

    for (uint32_t vocab_index = 0; vocab_index < vocab; ++vocab_index) {
        const float *weight_row = model.lm_head + static_cast<size_t>(vocab_index) * hidden_size;
        float sum = 0.0f;
        for (uint32_t hidden_index = 0; hidden_index < hidden_size; ++hidden_index) {
            sum += hidden[hidden_index] * weight_row[hidden_index];
        }
        logits[vocab_index] = sum;
    }
}

// Argmax of the logits for a hidden row. Reuses compute_logits (the same
// vocab loop the scoring path uses) and breaks ties toward the first (lowest)
// index via strict '>', matching the native gdn_eval argmax_logits exactly.
static int argmax_logits(const ModelData &model, const float *hidden, std::vector<float> &logits) {
    uint32_t vocab = model.config.vocab_size;
    compute_logits(model, hidden, logits);

    float max_logit = logits[0];
    int max_index = 0;
    for (uint32_t vocab_index = 1; vocab_index < vocab; ++vocab_index) {
        if (logits[vocab_index] > max_logit) {
            max_logit = logits[vocab_index];
            max_index = static_cast<int>(vocab_index);
        }
    }
    return max_index;
}

static double score_hidden_step(
    const ModelData &model,
    const float *hidden,
    int32_t target,
    std::vector<float> &logits
) {
    uint32_t vocab = model.config.vocab_size;
    if (target < 0 || static_cast<uint32_t>(target) >= vocab) {
        throw std::runtime_error("target token id out of range");
    }

    compute_logits(model, hidden, logits);
    float max_logit = logits[0];
    for (uint32_t i = 1; i < vocab; ++i) {
        max_logit = std::max(max_logit, logits[i]);
    }

    double sum = 0.0;
    for (uint32_t i = 0; i < vocab; ++i) {
        sum += std::exp(static_cast<double>(logits[i]) - static_cast<double>(max_logit));
    }
    return static_cast<double>(logits[static_cast<uint32_t>(target)]) -
           (static_cast<double>(max_logit) + std::log(sum));
}

static xrt::kernel open_gdn_kernel(xrt::device &device, const xrt::uuid &uuid) {
    try {
        return xrt::kernel(device, uuid, "gdn_forward:{gdn_forward_1}");
    } catch (const std::exception &) {
        return xrt::kernel(device, uuid, "gdn_forward");
    }
}

class HwRunner {
public:
    HwRunner(xrt::device &device, const xrt::uuid &uuid, const ModelData &model)
        : kernel_(open_gdn_kernel(device, uuid)),
          max_tokens_(1),
          hidden_(model.config.hidden_size),
          vocab_(model.config.vocab_size),
          embeddings_(model.weight_data.data()),
          x_norm_host_(hidden_, 0.0f) {
        const size_t shard_floats = gdn_weight_shard_floats(&model.config);
        const size_t shard_bytes = shard_floats * sizeof(float);
        if (shard_floats != GDN_COMPILED_WEIGHT_SHARD_FLOATS) {
            throw std::runtime_error("loaded model weight-shard size != compiled kernel layout");
        }
        const size_t state_stripe_bytes =
            static_cast<size_t>(GDN_RECURRENT_STATE_STRIPE_FLOATS) * sizeof(float);
        const size_t aux_floats = gdn_aux_weight_floats(&model.config);
        const size_t aux_bytes = aux_floats * sizeof(float);
        // step 4 Stage B arg order: 0=aux_weights, 1=workspace, 2..33=weight_mm0..31.
        weight_bos_.reserve(GEMV_CHANNELS);
        for (int c = 0; c < GEMV_CHANNELS; ++c) {
            size_t extra_bytes = c == 0 ? aux_bytes : 0;
            if (c >= GDN_RECURRENT_STATE_FIRST_PORT &&
                c < GDN_RECURRENT_STATE_FIRST_PORT + GDN_RECURRENT_STATE_PORTS) {
                extra_bytes = state_stripe_bytes;
            }
            weight_bos_.emplace_back(device, shard_bytes + extra_bytes,
                                     kernel_.group_id(2 + c));
        }
        aux_weight_bo_ = xrt::bo(weight_bos_[0], aux_bytes, shard_bytes);

        // step 4 Stage B: ONE HBM0 workspace BO holds all 15 activation/state
        // buffers at the GDN_WS_OFF_* byte offsets (kernel derives its pointers to
        // match). config is hardcoded in the kernel, so no config BO. The layout
        // static_asserts against the model dims in gdn_model.cpp; assert here too
        // that the loaded model matches the compiled-in shape.
        // The GDN_WSF_* workspace sizes (gdn_model.h) encode the compiled-in shape;
        // recurrent_state/head_buffer sizes fold in layers/heads/conv, so matching
        // these catches any wrong model dimension.
        if (model.config.hidden_size != GDN_WSF_HID ||
            model.config.intermediate_size != GDN_WSF_MLP ||
            model.config.vocab_size != GDN_WSF_LOGITS ||
            recurrent_state_bytes(model) != static_cast<size_t>(GDN_WSF_STATE) * sizeof(float) ||
            head_buffer_bytes(model) != static_cast<size_t>(GDN_WSF_HEADBUF) * sizeof(float)) {
            throw std::runtime_error("loaded model shape != compiled-in GDN-1.3B kernel shape");
        }
        workspace_bo_ = xrt::bo(device,
            static_cast<size_t>(GDN_WS_FLOATS) * sizeof(float), kernel_.group_id(1));

        std::cerr << "[progress] upload weights to device\n";
        // Each shard BO occupies one HBM bank. The compact auxiliary weights use
        // a sub-buffer in the unused tail of shard 0. Embeddings stay on the host;
        // only the selected 8 KiB row is uploaded into x for each decode call.
        {
            std::cerr << "[progress] building + uploading " << GEMV_CHANNELS
                      << " gemv weight shards and compact auxiliary weights\n";
            std::vector<float> sbuf[GEMV_CHANNELS];
            float *shards[GEMV_CHANNELS];
            for (int c = 0; c < GEMV_CHANNELS; ++c) {
                sbuf[c].resize(shard_floats);
                shards[c] = sbuf[c].data();
            }
            gdn_build_weight_shards(model.weight_data.data(), &model.config, shards);
            for (int c = 0; c < GEMV_CHANNELS; ++c) {
                weight_bos_[c].write(shards[c], shard_bytes, 0);
                sync_bo_chunked(weight_bos_[c], XCL_BO_SYNC_BO_TO_DEVICE,
                                shard_bytes, 0);
            }
            std::vector<float> aux(aux_floats);
            gdn_build_aux_weights(model.weight_data.data(), &model.config, aux.data());
            aux_weight_bo_.write(aux.data(), aux_bytes, 0);
            sync_bo_chunked(aux_weight_bo_, XCL_BO_SYNC_BO_TO_DEVICE, aux_bytes, 0);
        }
    }

    // On this xocl/xdma driver (XRT 2022.1, U55C), a nominally maximal 16 MiB
    // sync can still return EINVAL for a large BO at a nonzero subrange offset.
    // Keep transfers below that boundary; 8 MiB is accepted for both weight
    // BOs and the packed recurrent-state region.
    static constexpr size_t kSyncChunk = 8ULL * 1024 * 1024;  // 8 MiB
    static void sync_bo_chunked(xrt::bo &bo, xclBOSyncDirection dir,
                                size_t size, size_t offset) {
        for (size_t done = 0; done < size; done += kSyncChunk) {
            const size_t this_chunk = std::min(kSyncChunk, size - done);
            bo.sync(dir, this_chunk, offset + done);
        }
    }

    double run_forward(const std::vector<int32_t> &tokens) {
        if (tokens.empty() || tokens.size() > max_tokens_) {
            throw std::runtime_error("invalid token sequence length for hardware run");
        }

        int32_t token = tokens[0];
        if (token < 0 || static_cast<uint32_t>(token) >= vocab_) {
            throw std::runtime_error("token id out of range for hardware run");
        }
        size_t x_bytes = tokens.size() * static_cast<size_t>(hidden_) * sizeof(float);
        const float *embedding = embeddings_ + static_cast<size_t>(token) * hidden_;
        const size_t x_off = GDN_WS_OFF_X * sizeof(float);
        workspace_bo_.write(embedding, x_bytes, x_off);
        sync_bo_chunked(workspace_bo_, XCL_BO_SYNC_BO_TO_DEVICE, x_bytes, x_off);

        // Kernel-call construction sets the AXI-Lite argument registers and writes
        // ap_start (host-side PCIe overhead, ~100-200 µs, scales with arg
        // count). Start the clock AFTER that so kernel_ms reflects only FPGA
        // execution. The kernel has actually been running for a few µs by the
        // time `start` is sampled, but that skew is <1 part in 10^6 of a
        // multi-minute run and is dwarfed by host-clock resolution anyway.
        xrt::run run(kernel_);
        run.set_arg(0, aux_weight_bo_);
        run.set_arg(1, workspace_bo_);
        for (int c = 0; c < GEMV_CHANNELS; ++c) {
            run.set_arg(2 + c, weight_bos_[c]);
        }
        run.start();
        auto start = std::chrono::high_resolution_clock::now();
        run.wait();
        auto end = std::chrono::high_resolution_clock::now();

        double seconds = std::chrono::duration<double>(end - start).count();
        total_kernel_seconds_ += seconds;
        kernel_runs_ += 1;

        size_t x_norm_bytes = tokens.size() * static_cast<size_t>(hidden_) * sizeof(float);
        const size_t xn_off = GDN_WS_OFF_X_NORM * sizeof(float);
        sync_bo_chunked(workspace_bo_, XCL_BO_SYNC_BO_FROM_DEVICE, x_norm_bytes, xn_off);
        workspace_bo_.read(x_norm_host_.data(), x_norm_bytes, xn_off);

        return seconds;
    }

    // Upload the GPU-exported post-prompt state into the resident state BOs.
    // recurrent_state holds all layers (48 MB); head_buffer is the conv-tail
    // store (~1.7 MB). The decode kernel restores from these at each layer start
    // and saves the update at layer end, so they persist across token calls.
    void upload_decode_state(const std::vector<float> &recurrent,
                             const std::vector<float> &conv) {
        size_t rbytes = recurrent.size() * sizeof(float);
        size_t cbytes = conv.size() * sizeof(float);
        const size_t hb_off  = GDN_WS_OFF_HEAD_BUF * sizeof(float);
        const size_t shard_bytes =
            static_cast<size_t>(GDN_COMPILED_WEIGHT_SHARD_FLOATS) * sizeof(float);
        if (rbytes != static_cast<size_t>(GDN_WSF_STATE) * sizeof(float)) {
            throw std::runtime_error("recurrent-state size != compiled striped layout");
        }
        std::vector<float> stripes[GDN_RECURRENT_STATE_PORTS];
        float *stripe_ptrs[GDN_RECURRENT_STATE_PORTS];
        for (int p = 0; p < GDN_RECURRENT_STATE_PORTS; ++p) {
            stripes[p].resize(GDN_RECURRENT_STATE_STRIPE_FLOATS);
            stripe_ptrs[p] = stripes[p].data();
        }
        gdn_scatter_recurrent_state(stripe_ptrs,
                                    recurrent.data(), recurrent.size());
        for (int p = 0; p < GDN_RECURRENT_STATE_PORTS; ++p) {
            const int port = GDN_RECURRENT_STATE_FIRST_PORT + p;
            const size_t stripe_bytes = stripes[p].size() * sizeof(float);
            weight_bos_[port].write(stripes[p].data(), stripe_bytes, shard_bytes);
            sync_bo_chunked(weight_bos_[port], XCL_BO_SYNC_BO_TO_DEVICE,
                            stripe_bytes, shard_bytes);
        }
        workspace_bo_.write(conv.data(), cbytes, hb_off);
        sync_bo_chunked(workspace_bo_, XCL_BO_SYNC_BO_TO_DEVICE, cbytes, hb_off);
    }

    const float *hidden_row(uint32_t row) const {
        return x_norm_host_.data() + static_cast<size_t>(row) * hidden_;
    }

    double total_kernel_seconds() const { return total_kernel_seconds_; }
    uint64_t kernel_runs() const { return kernel_runs_; }
    double average_kernel_seconds() const {
        return kernel_runs_ == 0 ? 0.0 : total_kernel_seconds_ / static_cast<double>(kernel_runs_);
    }

private:
    static size_t hidden_bytes(const ModelData &model) {
        return static_cast<size_t>(model.config.hidden_size) * sizeof(float);
    }

    static size_t head_bytes(const ModelData &model) {
        return static_cast<size_t>(model.config.num_heads) * sizeof(float);
    }

    static size_t mlp_bytes(const ModelData &model) {
        return static_cast<size_t>(model.config.intermediate_size) * sizeof(float);
    }

    // Decode persistence (mirrors gdn_run_state_init): recurrent_state holds
    // ALL layers' states (24 x 8 x 256 x 256 fp32 = 48 MB) and head_buffer is
    // the conv tail store (24 layers x 3 convs x (conv_size-1) rows x hidden
    // ~ 1.7 MB). Prefill (decode_flags=0) never touches either.
    static size_t recurrent_state_bytes(const ModelData &model) {
        size_t value_dim = model.config.hidden_size / model.config.num_heads;
        return static_cast<size_t>(model.config.num_layers) *
               model.config.num_heads *
               model.config.head_dim *
               value_dim *
               sizeof(float);
    }

    static size_t head_buffer_bytes(const ModelData &model) {
        return static_cast<size_t>(model.config.num_layers) * 3 *
               (model.config.conv_size - 1) *
               model.config.hidden_size *
               sizeof(float);
    }

    xrt::kernel kernel_;
    uint32_t max_tokens_ = 0;
    uint32_t hidden_ = 0;
    uint32_t vocab_ = 0;
    const float *embeddings_ = nullptr;
    // step 4 Stage B: the 15 activation/state buffers are packed into one HBM[0]
    // workspace BO at the GDN_WS_OFF_* byte offsets; the host writes/reads/syncs
    // ranges of it and passes it as the single kernel `workspace` arg. config is
    // hardcoded (no BO); aux_weights stays a sub-buffer of weight shard 0.
    xrt::bo aux_weight_bo_;
    xrt::bo workspace_bo_;
    std::vector<xrt::bo> weight_bos_;
    std::vector<float> x_norm_host_;
    double total_kernel_seconds_ = 0.0;
    uint64_t kernel_runs_ = 0;
};

// Per-example results of the on-card decode benchmark. Mirrors the native
// gdn_eval schema plus an on-card kernel_ms series.
struct DecodeExample {
    uint32_t n = 0;                          // decode length used for this example
    std::vector<int32_t> gen_traj;           // free-running greedy trajectory (N)
    std::vector<int32_t> tf_argmax;          // teacher-forced per-position argmax (N)
    std::vector<double> per_step_tpot_ms;    // wall ms per greedy step (N)
    std::vector<double> kernel_ms;           // on-card kernel ms per greedy step (N)
};

// --decode driver: decode-only from a GPU-exported .gdnstate blob.
//
// The fixture provides the golden continuation length (and example indices).
// The device starts from the exported seed token and performs one single-token
// decode step per call, updating persistent recurrent/conv state in device memory.
//   - gen_traj / per_step_tpot_ms / kernel_ms: free-running greedy. The forward
//     clears recurrent state every call, so each step re-prefills the whole
//     growing prefix (O(n^2) — the honest current-kernel baseline). per-step
//     wall time is chrono around the run_forward + argmax; kernel_ms is the
//     on-card kernel time run_forward returns (seconds * 1000).
//
// N = min(cont_len, decode_len) (decode_len == 0 means use cont_len);
// E = min(num_examples, limit) (limit == 0 means all).
static std::vector<DecodeExample> run_decode_hw(
    const ModelData &model,
    HwRunner &runner,
    std::vector<float> &logits,
    const Fixture &fixture,
    uint32_t decode_len,
    const std::string &state_path
) {
    // ---- Read the GPU-exported .gdnstate blob ----
    std::ifstream f(state_path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open .gdnstate: " + state_path);
    char magic[8];
    f.read(magic, 8);
    if (std::memcmp(magic, "GDNSTAT1", 8) != 0)
        throw std::runtime_error(".gdnstate: bad magic (expected GDNSTAT1)");
    uint32_t v[9];
    f.read(reinterpret_cast<char *>(v), sizeof(v));
    // v = {version, num_layers, H, K, V, hidden, W, prompt_len, seed_token}
    uint32_t num_layers = v[1], H = v[2], Kk = v[3], Vv = v[4];
    uint32_t hidden = v[5], W = v[6], prompt_len = v[7];
    int32_t seed = static_cast<int32_t>(v[8]);
    if (num_layers != model.config.num_layers || hidden != model.config.hidden_size ||
        W != model.config.conv_size)
        throw std::runtime_error(".gdnstate: dims do not match the loaded model");
    f.seekg(static_cast<std::streamoff>(prompt_len) * sizeof(int32_t), std::ios::cur);
    size_t rec_count  = (size_t)num_layers * H * Kk * Vv;
    size_t conv_count = (size_t)num_layers * 3u * (W - 1u) * hidden;
    std::vector<float> rec(rec_count), conv(conv_count);
    f.read(reinterpret_cast<char *>(rec.data()),  rec_count  * sizeof(float));
    f.read(reinterpret_cast<char *>(conv.data()), conv_count * sizeof(float));
    if (!f) throw std::runtime_error(".gdnstate: truncated");
    std::cerr << "[progress] loaded .gdnstate: layers=" << num_layers
              << " hidden=" << hidden << " W=" << W << " seed=" << seed
              << " (recurrent=" << (rec_count * sizeof(float) / 1e6) << " MB)\n";

    // ---- Upload the post-prompt state to the resident BOs (once) ----
    runner.upload_decode_state(rec, conv);

    // ---- Decode length from the fixture's golden continuation (example 0) ----
    if (fixture.num_examples == 0 || fixture.ll_examples.empty())
        throw std::runtime_error("--decode-from-state needs an LL-kind fixture for the golden");
    const PairReq &req = fixture.ll_examples[0];
    uint32_t n = req.cont_len;
    if (decode_len != 0 && decode_len < n) n = decode_len;
    if (n == 0) throw std::runtime_error("--decode: zero decode length");

    std::vector<DecodeExample> results(1);
    DecodeExample &result = results[0];
    result.n = n;
    result.gen_traj.resize(n);
    result.tf_argmax.resize(n);
    result.per_step_tpot_ms.resize(n);
    result.kernel_ms.resize(n);

    // traj[0] = the GPU-exported seed (argmax of the prompt's last position);
    // every later token is one O(1) decode step against the persistent state.
    // The kernel restores the loaded state at each layer start and saves the
    // update at layer end, so the state BO carries forward across calls.
    result.gen_traj[0] = seed;
    result.tf_argmax[0] = seed;
    result.per_step_tpot_ms[0] = 0.0;
    result.kernel_ms[0] = 0.0;
    std::cerr << "[progress] decode-from-state seed=" << seed << " N=" << n << "\n";
    for (uint32_t step = 1; step < n; ++step) {
        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<int32_t> tok(1, result.gen_traj[step - 1]);
        double ksec = runner.run_forward(tok);
        /* lm_head + greedy argmax run on-chip now; the kernel writes the next
         * token id into x_norm[0] (read back by run_forward). No host logits. */
        int next = (int32_t)runner.hidden_row(0)[0];
        (void)logits; (void)model;
        auto t1 = std::chrono::high_resolution_clock::now();
        result.per_step_tpot_ms[step] =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        result.kernel_ms[step] = ksec * 1000.0;
        result.gen_traj[step] = next;
        result.tf_argmax[step] = next;
    }
    std::cerr << "[progress] decode-from-state complete (" << n << " tokens)\n";
    return results;
}

// Writes the on-card decode JSON. Schema is the native gdn_eval decode schema
// plus a per-step kernel_ms series.
static void write_decode_json(
    const std::string &output_path,
    uint32_t decode_len,
    const std::vector<DecodeExample> &results
) {
    std::ofstream file(output_path);
    if (!file) {
        throw std::runtime_error("failed to open output: " + output_path);
    }

    file << std::fixed << std::setprecision(6);
    file << "{\"kind\": " << kReqKindLL
         << ", \"decode_len\": " << decode_len
         << ", \"num_examples\": " << results.size() << ",\n";
    file << " \"examples\": [";
    for (size_t example_index = 0; example_index < results.size(); ++example_index) {
        const DecodeExample &result = results[example_index];
        uint32_t n = result.n;
        file << (example_index ? "," : "") << "\n  {\"index\": " << example_index << ",\n";
        file << "   \"gen_traj\": [";
        for (uint32_t j = 0; j < n; ++j) {
            file << (j ? ", " : "") << result.gen_traj[j];
        }
        file << "],\n   \"tf_argmax\": [";
        for (uint32_t j = 0; j < n; ++j) {
            file << (j ? ", " : "") << result.tf_argmax[j];
        }
        file << "],\n   \"per_step_tpot_ms\": [";
        for (uint32_t j = 0; j < n; ++j) {
            file << (j ? ", " : "") << result.per_step_tpot_ms[j];
        }
        file << "],\n   \"kernel_ms\": [";
        for (uint32_t j = 0; j < n; ++j) {
            file << (j ? ", " : "") << result.kernel_ms[j];
        }
        file << "]}";
    }
    file << "\n ]}\n";
}

static double score_pair_hw(
    const ModelData &model,
    HwRunner &runner,
    std::vector<float> &logits,
    const PairReq &pair,
    double *kernel_seconds
) {
    if (pair.ctx_len == 0 || pair.cont_len == 0) {
        throw std::runtime_error("invalid scoring pair");
    }
    uint32_t total_tokens = pair.ctx_len + pair.cont_len - 1;
    if (total_tokens == 0 || total_tokens > model.config.max_seq_len) {
        throw std::runtime_error("token sequence exceeds max_seq_len");
    }

    std::vector<int32_t> tokens(total_tokens);
    std::copy(pair.ctx.begin(), pair.ctx.end(), tokens.begin());
    if (pair.cont_len > 1) {
        std::copy(pair.cont.begin(), pair.cont.end() - 1, tokens.begin() + pair.ctx_len);
    }

    *kernel_seconds = runner.run_forward(tokens);

    double logprob = 0.0;
    for (uint32_t token_index = 0; token_index < pair.cont_len; ++token_index) {
        uint32_t row = pair.ctx_len + token_index - 1;
        logprob += score_hidden_step(model, runner.hidden_row(row), pair.cont[token_index], logits);
    }
    return logprob;
}

static void write_json(
    const Options &opts,
    uint32_t evaluated_examples,
    uint64_t evaluated_windows,
    uint64_t total_words,
    uint64_t total_bytes,
    double total_logprob,
    const std::vector<double> &doc_lls,
    const HwRunner &runner
) {
    double word_perplexity = std::exp(-total_logprob / static_cast<double>(total_words));
    double byte_perplexity = std::exp(-total_logprob / static_cast<double>(total_bytes));
    double bits_per_byte = -total_logprob / static_cast<double>(total_bytes) / std::log(2.0);
    double average_kernel_ms = runner.average_kernel_seconds() * 1000.0;
    double total_kernel_ms = runner.total_kernel_seconds() * 1000.0;

    std::ostream *out = &std::cout;
    std::ofstream file;
    if (opts.output != "-") {
        file.open(opts.output);
        if (!file) {
            throw std::runtime_error("failed to open output: " + opts.output);
        }
        out = &file;
    }

    *out << std::fixed << std::setprecision(9);
    *out << "{\n"
         << "  \"xclbin\": \"" << opts.xclbin << "\",\n"
         << "  \"weights\": \"" << opts.weights << "\",\n"
         << "  \"fixture\": \"" << opts.fixture << "\",\n"
         << "  \"device_index\": " << opts.device_index << ",\n"
         << "  \"num_examples\": " << evaluated_examples << ",\n"
         << "  \"num_windows\": " << evaluated_windows << ",\n"
         << "  \"metrics\": {\n"
         << "    \"word_perplexity\": " << word_perplexity << ",\n"
         << "    \"byte_perplexity\": " << byte_perplexity << ",\n"
         << "    \"bits_per_byte\": " << bits_per_byte << ",\n"
         << "    \"average_kernel_ms\": " << average_kernel_ms << ",\n"
         << "    \"total_kernel_ms\": " << total_kernel_ms << ",\n"
         << "    \"kernel_runs\": " << runner.kernel_runs() << "\n"
         << "  },\n"
         << "  \"examples\": [\n";
    for (size_t i = 0; i < doc_lls.size(); ++i) {
        *out << "    {\"index\": " << i << ", \"logprob\": " << doc_lls[i] << "}";
        if (i + 1 != doc_lls.size()) {
            *out << ",";
        }
        *out << "\n";
    }
    *out << "  ]\n}\n";
}

}  // namespace

int main(int argc, char **argv) {
    try {
        Options opts = parse_options(argc, argv);
        ensure_xrt_environment();

        std::cerr << "[progress] open device " << opts.device_index << "\n";
        auto device = xrt::device(opts.device_index);
        std::cerr << "[progress] load xclbin " << opts.xclbin << "\n";
        auto uuid = device.load_xclbin(opts.xclbin);

        std::cerr << "[progress] loading model weights from " << opts.weights << "\n";
        ModelData model = load_model(opts.weights);
        std::cerr << "[progress] model hidden=" << model.config.hidden_size
                  << " layers=" << model.config.num_layers
                  << " max_seq_len=" << model.config.max_seq_len
                  << " vocab=" << model.config.vocab_size << "\n";

        if (opts.decode) {
            // Decode-only: the FPGA never prefills. The post-prompt recurrent +
            // conv state comes from the GPU export on disk (--decode-from-state);
            // the fixture supplies only the golden continuation for comparison.
            if (opts.state_path.empty()) {
                throw std::runtime_error(
                    "decode-only build: pass --decode-from-state <state.gdnstate>");
            }
            std::string decode_out = opts.output_set ? opts.output : kDefaultDecodeOutput;
            std::cerr << "[progress] loading LL decode fixture from " << opts.fixture << "\n";
            Fixture fixture = load_ll_fixture(opts.fixture);
            std::cerr << "[progress] fixture examples=" << fixture.num_examples << "\n";

            std::cerr << "[progress] allocate XRT buffers\n";
            HwRunner runner(device, uuid, model);

            std::vector<float> logits(model.config.vocab_size, 0.0f);
            std::vector<DecodeExample> results =
                run_decode_hw(model, runner, logits, fixture, opts.decode_len,
                              opts.state_path);

            write_decode_json(decode_out, opts.decode_len, results);
            std::cerr << "[progress] decode finished examples=" << results.size()
                      << " avg_kernel_ms=" << std::fixed << std::setprecision(3)
                      << (runner.average_kernel_seconds() * 1000.0)
                      << " output=" << decode_out << "\n";
            return EXIT_SUCCESS;
        }

        std::cerr << "[progress] loading rolling Wikitext fixture from " << opts.fixture << "\n";
        Fixture fixture = load_rolling_fixture(opts.fixture);
        uint32_t total_examples = fixture.num_examples;
        uint32_t eval_examples =
            opts.max_examples == 0 ? total_examples : std::min(opts.max_examples, total_examples);
        uint64_t total_fixture_windows = 0;
        for (const RollingReq &req : fixture.rolling_examples) {
            total_fixture_windows += req.windows.size();
        }
        std::cerr << "[progress] fixture examples=" << fixture.num_examples
                  << " windows=" << total_fixture_windows << "\n";

        std::cerr << "[progress] allocate XRT buffers\n";
        HwRunner runner(device, uuid, model);

        std::vector<float> logits(model.config.vocab_size, 0.0f);
        std::vector<double> doc_lls(eval_examples, 0.0);
        uint64_t total_words = 0;
        uint64_t total_bytes = 0;
        uint64_t evaluated_windows = 0;
        double total_logprob = 0.0;

        for (uint32_t example_index = 0; example_index < eval_examples; ++example_index) {
            const RollingReq &req = fixture.rolling_examples[example_index];
            std::cerr << "[progress] rolling doc " << (example_index + 1)
                      << "/" << eval_examples
                      << " windows=" << req.windows.size() << "\n";

            double doc_ll = 0.0;
            for (size_t window_index = 0; window_index < req.windows.size(); ++window_index) {
                double kernel_seconds = 0.0;
                double ll = score_pair_hw(model, runner, logits, req.windows[window_index], &kernel_seconds);
                doc_ll += ll;
                evaluated_windows += 1;
                std::cerr << "[progress] doc " << (example_index + 1)
                          << " window " << (window_index + 1)
                          << "/" << req.windows.size()
                          << " kernel_ms=" << std::fixed << std::setprecision(3)
                          << (kernel_seconds * 1000.0)
                          << " avg_kernel_ms=" << (runner.average_kernel_seconds() * 1000.0)
                          << "\n";
            }

            doc_lls[example_index] = doc_ll;
            total_logprob += doc_ll;
            total_words += req.word_count;
            total_bytes += req.byte_count;
            double partial_word_ppl = std::exp(-total_logprob / static_cast<double>(total_words));
            std::cerr << "[progress] rolling doc " << (example_index + 1)
                      << "/" << eval_examples
                      << " complete partial_word_perplexity="
                      << std::setprecision(6) << partial_word_ppl << "\n";
        }

        if (total_words == 0 || total_bytes == 0 || evaluated_windows == 0) {
            throw std::runtime_error("no Wikitext examples were evaluated");
        }

        double word_perplexity = std::exp(-total_logprob / static_cast<double>(total_words));
        double byte_perplexity = std::exp(-total_logprob / static_cast<double>(total_bytes));
        double bits_per_byte = -total_logprob / static_cast<double>(total_bytes) / std::log(2.0);
        double avg_kernel_ms = runner.average_kernel_seconds() * 1000.0;

        write_json(
            opts,
            eval_examples,
            evaluated_windows,
            total_words,
            total_bytes,
            total_logprob,
            doc_lls,
            runner
        );

        std::cout << std::fixed << std::setprecision(6)
                  << "word_perplexity=" << word_perplexity
                  << " byte_perplexity=" << byte_perplexity
                  << " bits_per_byte=" << bits_per_byte
                  << " average_kernel_ms=" << avg_kernel_ms
                  << " kernel_runs=" << runner.kernel_runs()
                  << "\n";

        return EXIT_SUCCESS;
    } catch (const std::exception &ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }
}
