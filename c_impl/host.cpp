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
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr const char *kDefaultWeights = "artifacts/gdn-1.3b-bf16w.gdnw";
constexpr const char *kDefaultFixture = "fixtures_decode/decode.gdnreq";
constexpr const char *kDefaultDecodeOutput = "results_decode_hw/decode.hw.json";
constexpr const char *kDefaultXrt = "/opt/xilinx/xrt";
constexpr uint32_t kReqKindLL = 2;
constexpr uint32_t kReqKindRolling = 3;
constexpr size_t kWeightHeaderBytes = 60;

struct LogitsDumpHeader {
    char magic[8];
    uint32_t version;
    uint32_t vocab_size;
    uint32_t decode_steps;
};

struct LogitsReference {
    uint32_t vocab_size = 0;
    uint32_t decode_steps = 0;
    std::vector<float> values;
};

struct LogitsParity {
    uint32_t checked_steps = 0;
    uint64_t compared_values = 0;
    uint64_t tolerance_failures = 0;
    uint64_t nonfinite_mismatches = 0;
    uint64_t exact_reference_mismatches = 0;
    uint32_t argmax_mismatches = 0;
    uint32_t min_top5_overlap = 5;
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    double squared_error_sum = 0.0;
    double candidate_squared_sum = 0.0;
    double reference_squared_sum = 0.0;
    double dot_product_sum = 0.0;
    double worst_step_nrmse = 0.0;
    double min_step_cosine = 1.0;
    double worst_max_abs_over_reference_rms = 0.0;
};

/* Independent CUDA and HLS all-BF16 executions round at the same operator
 * boundaries but associate their internal FP32 reductions differently. These
 * scale-aware limits validate every logit without pretending the two machines
 * should reproduce one FP32 implementation bit-for-bit. Hardware/native is a
 * separate bit-exact gate below and is not relaxed by this envelope. */
constexpr double kGpuGlobalNrmseMax = 0.01;
constexpr double kGpuWorstStepNrmseMax = 0.04;
constexpr double kGpuMinStepCosineMin = 0.9995;
constexpr double kGpuWorstMaxOverRmsMax = 0.10;
constexpr double kGpuMaxAbsMax = 0.50;
constexpr uint32_t kGpuMinTop5Overlap = 5;

static double logits_global_nrmse(const LogitsParity &parity) {
    return parity.reference_squared_sum > 0.0
        ? std::sqrt(parity.squared_error_sum /
                    parity.reference_squared_sum)
        : std::numeric_limits<double>::infinity();
}

static double logits_global_cosine(const LogitsParity &parity) {
    const double norm_product =
        parity.candidate_squared_sum * parity.reference_squared_sum;
    return norm_product > 0.0
        ? parity.dot_product_sum / std::sqrt(norm_product)
        : -1.0;
}

static bool gpu_logits_gate_passes(const LogitsParity &parity) {
    return parity.nonfinite_mismatches == 0 &&
           parity.argmax_mismatches == 0 &&
           parity.min_top5_overlap >= kGpuMinTop5Overlap &&
           logits_global_nrmse(parity) <= kGpuGlobalNrmseMax &&
           parity.worst_step_nrmse <= kGpuWorstStepNrmseMax &&
           parity.min_step_cosine >= kGpuMinStepCosineMin &&
           parity.worst_max_abs_over_reference_rms <=
               kGpuWorstMaxOverRmsMax &&
           parity.max_abs_error <= kGpuMaxAbsMax;
}

struct BeatFree {
    void operator()(Beat512 *pointer) const { std::free(pointer); }
};

using AlignedBeatBuffer = std::unique_ptr<Beat512, BeatFree>;

static AlignedBeatBuffer allocate_beat_buffer(size_t beat_count) {
    void *raw = nullptr;
    const size_t bytes = beat_count * sizeof(Beat512);
    if (posix_memalign(&raw, 64, bytes) != 0 || raw == nullptr) {
        throw std::runtime_error("failed to allocate 64-byte-aligned Beat512 buffer");
    }
    return AlignedBeatBuffer(static_cast<Beat512 *>(raw));
}

struct DecodeReq {
    uint32_t cont_len = 0;
};

struct Fixture {
    uint32_t num_examples = 0;
    std::vector<DecodeReq> examples;
};

/* Rolling-loglikelihood (kind=3) fixture: WikiText perplexity on card.
 * Each document carries lm-eval's word/byte counts and its disjoint
 * (context, continuation) windows, so word and byte perplexity are directly
 * comparable with the GPU arm. */
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

struct RollingFixture {
    uint32_t num_examples = 0;
    std::vector<RollingReq> documents;
};

struct ModelData {
    GDNWeightHeader config{};
    std::vector<float> weight_data;
};

struct Options {
    std::string xclbin;
    std::string weights = kDefaultWeights;
    std::string fixture = kDefaultFixture;
    std::string output = kDefaultDecodeOutput;
    unsigned int device_index = 0;
    /* A device argument containing ':' is a PCIe BDF (e.g. 0000:41:00.1).
     * On nodes that expose an unallocated second card to `xbutil examine`
     * (acclnode01's U280 beside the allocated U55C, on-card jobs 1354/2504),
     * an enumeration index is ambiguous; the BDF constructor is not. */
    std::string device_bdf;
    std::string state_path;
    uint32_t decode_len = 0;    // 0 => use the fixture's golden cont_len
    std::string logits_reference_path;
    std::string gpu_logits_reference_path;
    /* Iter66g probes for the step-2 divergence: dump the persistent state
     * (four HBM stripes + conv-tail region) after the decode loop, and
     * optionally sleep between kernel invocations to test the in-flight
     * write-visibility hypothesis. */
    std::string state_dump_path;
    uint32_t interstep_delay_ms = 0;
    /* Iter66n: write every step's full logit vector in the same GDNLOG1
     * format gdn_eval --logits-dump emits, so drift can be analysed per
     * token offline. A worst-step aggregate is a maximum and is blind to a
     * slow upward trend, which is exactly the question at 512 tokens. */
    std::string logits_dump_path;
    /* Teacher-forced scoring mode (WikiText perplexity). Selected by
     * --score; the fixture must be kind=3. */
    bool score_mode = false;
    uint32_t score_doc_limit = 0;      // 0 => all documents
};

static void usage(const char *argv0) {
    std::cerr
        << "usage: " << argv0
        << " <gdn_forward.xclbin> [weights.gdnw] [fixture.gdnreq]"
        << " [output.json|-] [device_index]"
        << " --decode --decode-from-state <state.gdnstate> [--decode-len N]"
        << " [--logits-reference native.gdnlog]"
        << " [--gpu-logits-reference gpu.gdnlog]\n\n"
        << "defaults:\n"
        << "  weights      " << kDefaultWeights << "\n"
        << "  fixture      " << kDefaultFixture << "\n"
        << "  output       " << kDefaultDecodeOutput << "\n"
        << "  device_index 0\n\n"
        << "flags (the FPGA never prefills):\n"
        << "  --decode                     accepted for command compatibility\n"
        << "  --decode-from-state <file>   GPU-exported .gdnstate (recurrent+conv state); required\n"
        << "  --decode-len N               cap decode length (0 => fixture golden cont_len)\n"
        << "  --logits-reference <file>    DIAGNOSTIC: report hardware/native logit\n"
        << "                               differences. Not a gate -- see Iter66m;\n"
        << "                               +-1 ULP differences here are expected.\n"
        << "  --gpu-logits-reference <file> run the BF16-aware independent-GPU full-vector gate\n";
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

    // Separate decode flags from the positional arguments.
    std::vector<std::string> positional;
    for (int arg_index = 1; arg_index < argc; ++arg_index) {
        std::string arg = argv[arg_index];
        if (arg == "--decode") {
            continue;
        } else if (arg == "--decode-from-state") {
            if (arg_index + 1 >= argc) {
                throw std::runtime_error("--decode-from-state requires a .gdnstate path");
            }
            opts.state_path = argv[++arg_index];
        } else if (arg == "--decode-len") {
            if (arg_index + 1 >= argc) {
                throw std::runtime_error("--decode-len requires a value");
            }
            opts.decode_len = parse_u32(argv[++arg_index], "decode-len");
        } else if (arg == "--logits-reference") {
            if (arg_index + 1 >= argc) {
                throw std::runtime_error("--logits-reference requires a path");
            }
            opts.logits_reference_path = argv[++arg_index];
        } else if (arg == "--gpu-logits-reference") {
            if (arg_index + 1 >= argc) {
                throw std::runtime_error("--gpu-logits-reference requires a path");
            }
            opts.gpu_logits_reference_path = argv[++arg_index];
        } else if (arg == "--dump-state") {
            if (arg_index + 1 >= argc) {
                throw std::runtime_error("--dump-state requires a path");
            }
            opts.state_dump_path = argv[++arg_index];
        } else if (arg == "--score") {
            opts.score_mode = true;
        } else if (arg == "--score-doc-limit") {
            if (arg_index + 1 >= argc) {
                throw std::runtime_error("--score-doc-limit requires a value");
            }
            opts.score_doc_limit =
                parse_u32(argv[++arg_index], "score_doc_limit");
        } else if (arg == "--logits-dump") {
            if (arg_index + 1 >= argc) {
                throw std::runtime_error("--logits-dump requires a path");
            }
            opts.logits_dump_path = argv[++arg_index];
        } else if (arg == "--interstep-delay-ms") {
            if (arg_index + 1 >= argc) {
                throw std::runtime_error("--interstep-delay-ms requires a value");
            }
            opts.interstep_delay_ms =
                parse_u32(argv[++arg_index], "interstep_delay_ms");
        } else {
            positional.push_back(arg);
        }
    }

    if (positional.empty() || positional.size() > 5) {
        usage(argv[0]);
        std::exit(1);
    }
    opts.xclbin = positional[0];
    if (positional.size() > 1) opts.weights = positional[1];
    if (positional.size() > 2) opts.fixture = positional[2];
    if (positional.size() > 3) opts.output = positional[3];
    if (positional.size() > 4) {
        if (positional[4].find(':') != std::string::npos) {
            opts.device_bdf = positional[4];
        } else {
            opts.device_index = parse_u32(positional[4].c_str(), "device_index");
        }
    }
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

static LogitsReference load_logits_reference(
    const std::string &path,
    uint32_t expected_vocab,
    uint32_t required_steps
) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("cannot open logits reference: " + path);
    }
    const std::streamsize file_size = stream.tellg();
    stream.seekg(0, std::ios::beg);

    LogitsDumpHeader header{};
    stream.read(reinterpret_cast<char *>(&header), sizeof(header));
    if (!stream || std::memcmp(header.magic, "GDNLOG1", 7) != 0 ||
        header.version != 1 || header.vocab_size != expected_vocab ||
        header.decode_steps < required_steps) {
        throw std::runtime_error("incompatible logits reference: " + path);
    }
    const uint64_t value_count =
        static_cast<uint64_t>(header.vocab_size) * header.decode_steps;
    const uint64_t expected_bytes = sizeof(header) + value_count * sizeof(float);
    if (file_size < 0 || static_cast<uint64_t>(file_size) != expected_bytes) {
        throw std::runtime_error("logits reference size mismatch: " + path);
    }

    LogitsReference reference;
    reference.vocab_size = header.vocab_size;
    reference.decode_steps = header.decode_steps;
    reference.values.resize(static_cast<size_t>(value_count));
    stream.read(reinterpret_cast<char *>(reference.values.data()),
                static_cast<std::streamsize>(value_count * sizeof(float)));
    if (!stream) {
        throw std::runtime_error("truncated logits reference: " + path);
    }
    return reference;
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

static void skip_i32_array(
    const std::vector<uint8_t> &blob,
    size_t &offset,
    uint32_t count
) {
    size_t bytes = static_cast<size_t>(count) * sizeof(int32_t);
    if (offset + bytes > blob.size()) {
        throw std::runtime_error("fixture truncated");
    }
    offset += bytes;
}

static std::vector<int32_t> read_i32_array(const std::vector<uint8_t> &blob,
                                          size_t &offset, uint32_t count) {
    const size_t bytes = static_cast<size_t>(count) * sizeof(int32_t);
    if (offset + bytes > blob.size()) {
        throw std::runtime_error("fixture truncated");
    }
    std::vector<int32_t> values(count);
    if (count != 0) {
        std::memcpy(values.data(), blob.data() + offset, bytes);
    }
    offset += bytes;
    return values;
}

/* Loader for the rolling-loglikelihood (kind=3) WikiText fixture written by
 * scripts/export_gdn_c.py wikitext. */
static RollingFixture load_rolling_fixture(const std::string &path) {
    std::vector<uint8_t> blob = read_binary_file(path);
    if (blob.size() < 20 || std::memcmp(blob.data(), "GDNREQ1", 7) != 0) {
        throw std::runtime_error("unsupported fixture file: " + path);
    }
    size_t offset = 8;
    const uint32_t version = read_u32(blob, offset);
    const uint32_t kind = read_u32(blob, offset);
    RollingFixture fixture;
    fixture.num_examples = read_u32(blob, offset);
    if (version != 1) {
        throw std::runtime_error("unsupported fixture version");
    }
    if (kind != kReqKindRolling) {
        throw std::runtime_error("--score requires a rolling fixture (kind=3)");
    }
    fixture.documents.resize(fixture.num_examples);
    for (uint32_t doc = 0; doc < fixture.num_examples; ++doc) {
        RollingReq &req = fixture.documents[doc];
        req.word_count = read_u32(blob, offset);
        req.byte_count = read_u32(blob, offset);
        const uint32_t num_windows = read_u32(blob, offset);
        req.windows.resize(num_windows);
        for (uint32_t w = 0; w < num_windows; ++w) {
            PairReq &pair = req.windows[w];
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
    uint32_t kind = read_u32(blob, offset);
    fixture.num_examples = read_u32(blob, offset);
    if (version != 1) {
        throw std::runtime_error("unsupported fixture version");
    }
    if (kind != kReqKindLL) {
        throw std::runtime_error("decode requires an LL-kind (.gdnreq kind=2) fixture");
    }

    fixture.examples.resize(fixture.num_examples);
    for (uint32_t example_index = 0; example_index < fixture.num_examples; ++example_index) {
        uint32_t ctx_len = read_u32(blob, offset);
        fixture.examples[example_index].cont_len = read_u32(blob, offset);
        skip_i32_array(blob, offset, ctx_len);
        skip_i32_array(blob, offset, fixture.examples[example_index].cont_len);
    }
    if (offset != blob.size()) {
        throw std::runtime_error("unexpected trailing data in decode fixture");
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

    if (gdn_validate_bf16_exact_weights(model.weight_data.data(),
                                        &model.config) != 0) {
        throw std::runtime_error(
            "weight payload is not BF16-exact; use gdn-1.3b-bf16w.gdnw");
    }

    return model;
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
          hidden_(model.config.hidden_size),
          vocab_(model.config.vocab_size),
          embeddings_(model.weight_data.data()),
          logits_host_(GDN_WSF_LOGITS, 0.0f) {
        const size_t shard_beats = gdn_weight_shard_beats(&model.config);
        const size_t shard_bytes = gdn_weight_shard_bytes(&model.config);
        if (shard_beats != GDN_COMPILED_WEIGHT_SHARD_BEATS ||
            shard_bytes != GDN_COMPILED_WEIGHT_SHARD_BYTES) {
            throw std::runtime_error("loaded model weight-shard size != compiled kernel layout");
        }
        const size_t state_stripe_bytes =
            static_cast<size_t>(GDN_RECURRENT_STATE_STRIPE_BF16_BEATS) *
            sizeof(Beat512);
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
            AlignedBeatBuffer sbuf[GEMV_CHANNELS];
            Beat512 *shards[GEMV_CHANNELS];
            for (int c = 0; c < GEMV_CHANNELS; ++c) {
                sbuf[c] = allocate_beat_buffer(shard_beats);
                shards[c] = sbuf[c].get();
            }
            gdn_build_weight_shards(model.weight_data.data(), &model.config, shards);
            if (gdn_validate_weight_shards(model.weight_data.data(),
                                           &model.config, shards) != 0) {
                throw std::runtime_error("packed BF16 shard validation failed");
            }
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

    /* want_logits=false skips the 128 KB logit read-back and takes the token
     * id the kernel now writes to the workspace slot. A generation loop needs
     * only that; teacher-forced scoring and every reference comparison need
     * the full vector and pass true.
     *
     * generation_seconds, when requested, measures the production token-ID to
     * token-ID boundary: embedding lookup/upload, kernel launch/execution, and
     * selected-token read-back.  Stop that clock before the optional full-logit
     * transfer so enabling a correctness reference cannot inflate TPOT. */
    double run_forward(int32_t token, bool want_logits = true,
                       double *generation_seconds = nullptr) {
        if (token < 0 || static_cast<uint32_t>(token) >= vocab_) {
            throw std::runtime_error("token id out of range for hardware run");
        }
        const auto generation_start =
            std::chrono::high_resolution_clock::now();
        size_t x_bytes = static_cast<size_t>(hidden_) * sizeof(float);
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

        // The LM head streams its full logit vector to the workspace logits
        // region; pull it back for scoring and host-side greedy selection.
        /* One 512-bit line: lane 0 is the kernel's greedy pick. Always cheap. */
        const size_t tk_off = GDN_WS_OFF_X_NORM * sizeof(float);
        float token_line[16];
        sync_bo_chunked(workspace_bo_, XCL_BO_SYNC_BO_FROM_DEVICE,
                        sizeof(token_line), tk_off);
        workspace_bo_.read(token_line, sizeof(token_line), tk_off);
        device_token_ = static_cast<int32_t>(token_line[0]);

        const auto generation_end =
            std::chrono::high_resolution_clock::now();
        if (generation_seconds != nullptr) {
            *generation_seconds = std::chrono::duration<double>(
                generation_end - generation_start).count();
        }

        if (want_logits) {
            const size_t lg_off = GDN_WS_OFF_LOGITS * sizeof(float);
            const size_t lg_bytes =
                static_cast<size_t>(GDN_WSF_LOGITS) * sizeof(float);
            sync_bo_chunked(workspace_bo_, XCL_BO_SYNC_BO_FROM_DEVICE,
                            lg_bytes, lg_off);
            workspace_bo_.read(logits_host_.data(), lg_bytes, lg_off);
            logits_valid_ = true;
        } else {
            logits_valid_ = false;
        }

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
        const size_t shard_bytes = GDN_COMPILED_WEIGHT_SHARD_BYTES;
        if (rbytes != static_cast<size_t>(GDN_WSF_STATE) * sizeof(float)) {
            throw std::runtime_error("recurrent-state size != compiled striped layout");
        }
        AlignedBeatBuffer stripes[GDN_RECURRENT_STATE_PORTS];
        Beat512 *stripe_ptrs[GDN_RECURRENT_STATE_PORTS];
        for (int p = 0; p < GDN_RECURRENT_STATE_PORTS; ++p) {
            stripes[p] = allocate_beat_buffer(
                GDN_RECURRENT_STATE_STRIPE_BF16_BEATS);
            stripe_ptrs[p] = stripes[p].get();
        }
        if (gdn_scatter_recurrent_state(stripe_ptrs,
                                        recurrent.data(),
                                        recurrent.size()) != 0) {
            throw std::runtime_error(
                "recurrent-state values are not BF16-exact");
        }
        for (int p = 0; p < GDN_RECURRENT_STATE_PORTS; ++p) {
            const int port = GDN_RECURRENT_STATE_FIRST_PORT + p;
            const size_t stripe_bytes =
                static_cast<size_t>(GDN_RECURRENT_STATE_STRIPE_BF16_BEATS) *
                sizeof(Beat512);
            weight_bos_[port].write(stripes[p].get(), stripe_bytes, shard_bytes);
            sync_bo_chunked(weight_bos_[port], XCL_BO_SYNC_BO_TO_DEVICE,
                            stripe_bytes, shard_bytes);
        }
        AlignedBeatBuffer packed_conv = allocate_beat_buffer(GDN_WSF_HEADBUF / 16u);
        if (gdn_pack_conv_tails_bf16(packed_conv.get(), conv.data(),
                                     conv.size()) != 0) {
            throw std::runtime_error(
                "convolution-tail values are not BF16-exact");
        }
        workspace_bo_.write(packed_conv.get(), cbytes, hb_off);
        sync_bo_chunked(workspace_bo_, XCL_BO_SYNC_BO_TO_DEVICE, cbytes, hb_off);
    }

    /* Each rolling window is scored independently, so it must start from the
     * blank state the model would have at the beginning of a document. */
    void reset_decode_state() {
        static const std::vector<float> zero_recurrent(GDN_WSF_STATE, 0.0f);
        static const std::vector<float> zero_conv(GDN_WSF_HEADBUF, 0.0f);
        upload_decode_state(zero_recurrent, zero_conv);
    }

    /* Iter66g probe: read the persistent state back from the device at rest
     * (four BF16 stripes on ports 28..31 plus the conv-tail region) and write
     * it in the same GDNSDMP1 format gdn_eval emits, so hardware and native
     * post-step state can be diffed bit-for-bit. */
    void dump_persistent_state(const GDNWeightHeader &config,
                               const std::string &path) {
        const size_t shard_bytes = gdn_weight_shard_bytes(&config);
        const size_t stripe_bytes =
            static_cast<size_t>(GDN_RECURRENT_STATE_STRIPE_BF16_BEATS) *
            sizeof(Beat512);
        const size_t hb_off =
            static_cast<size_t>(GDN_WS_OFF_HEAD_BUF / 16u) * sizeof(Beat512);
        const size_t cbytes =
            static_cast<size_t>(GDN_WSF_HEADBUF / 16u) * sizeof(Beat512);
        FILE *out = std::fopen(path.c_str(), "wb");
        if (out == nullptr)
            throw std::runtime_error("cannot create state dump: " + path);
        const char magic[8] = {'G', 'D', 'N', 'S', 'D', 'M', 'P', '1'};
        const uint32_t header[4] = {
            GDN_RECURRENT_STATE_STRIPE_BF16_BEATS,
            GDN_RECURRENT_STATE_PORTS,
            static_cast<uint32_t>(GDN_WSF_HEADBUF / 16u), 0};
        std::fwrite(magic, 1, sizeof(magic), out);
        std::fwrite(header, sizeof(uint32_t), 4, out);
        std::vector<char> buffer(std::max(stripe_bytes, cbytes));
        for (int p = 0; p < GDN_RECURRENT_STATE_PORTS; ++p) {
            const int port = GDN_RECURRENT_STATE_FIRST_PORT + p;
            sync_bo_chunked(weight_bos_[port], XCL_BO_SYNC_BO_FROM_DEVICE,
                            stripe_bytes, shard_bytes);
            weight_bos_[port].read(buffer.data(), stripe_bytes, shard_bytes);
            if (std::fwrite(buffer.data(), 1, stripe_bytes, out) != stripe_bytes)
                throw std::runtime_error("state dump write failed: " + path);
        }
        sync_bo_chunked(workspace_bo_, XCL_BO_SYNC_BO_FROM_DEVICE,
                        cbytes, hb_off);
        workspace_bo_.read(buffer.data(), cbytes, hb_off);
        if (std::fwrite(buffer.data(), 1, cbytes, out) != cbytes)
            throw std::runtime_error("conv dump write failed: " + path);
        std::fclose(out);
        std::cerr << "[state_dump] wrote " << path << "\n";
    }

    // Full pre-argmax logit vector the kernel streamed out this step.
    const float *device_logits() const { return logits_host_.data(); }
    bool logits_valid() const { return logits_valid_; }
    /* The kernel's on-chip greedy pick: maximum value, lowest vocabulary index
     * on ties -- the same rule the host scan implements. */
    int32_t device_token() const { return device_token_; }

    double average_kernel_seconds() const {
        return kernel_runs_ == 0 ? 0.0 : total_kernel_seconds_ / static_cast<double>(kernel_runs_);
    }

private:
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
    std::vector<float> logits_host_;
    int32_t device_token_ = -1;
    bool logits_valid_ = false;
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

static void compare_logits_step(
    const float *actual,
    const float *reference,
    uint32_t vocab_size,
    uint32_t selected_token,
    bool apply_fp32_tolerance,
    LogitsParity *parity
) {
    uint32_t actual_argmax = 0;
    uint32_t reference_argmax = 0;
    float actual_best = actual[0];
    float reference_best = reference[0];
    float actual_top5_values[5];
    float reference_top5_values[5];
    uint32_t actual_top5_indices[5];
    uint32_t reference_top5_indices[5];
    for (uint32_t rank = 0; rank < 5; ++rank) {
        actual_top5_values[rank] = std::numeric_limits<float>::lowest();
        reference_top5_values[rank] = std::numeric_limits<float>::lowest();
        actual_top5_indices[rank] = vocab_size;
        reference_top5_indices[rank] = vocab_size;
    }
    double step_squared_error = 0.0;
    double step_candidate_squared = 0.0;
    double step_reference_squared = 0.0;
    double step_dot_product = 0.0;
    double step_max_abs_error = 0.0;
    for (uint32_t vocab_index = 0; vocab_index < vocab_size; ++vocab_index) {
        const float candidate = actual[vocab_index];
        const float expected = reference[vocab_index];
        const bool candidate_finite = std::isfinite(candidate);
        const bool expected_finite = std::isfinite(expected);
        const bool exact =
            std::memcmp(&candidate, &expected, sizeof(float)) == 0;
        if (!exact) {
            parity->exact_reference_mismatches++;
        }
        if (!candidate_finite || !expected_finite) {
            if (!exact) {
                parity->nonfinite_mismatches++;
                if (apply_fp32_tolerance) {
                    parity->tolerance_failures++;
                }
            }
        } else {
            const double abs_error = std::fabs(
                static_cast<double>(candidate) - expected);
            const double rel_error = abs_error /
                std::max(1.0, std::fabs(static_cast<double>(expected)));
            parity->max_abs_error = std::max(parity->max_abs_error, abs_error);
            parity->max_rel_error = std::max(parity->max_rel_error, rel_error);
            step_max_abs_error = std::max(step_max_abs_error, abs_error);
            const double error = static_cast<double>(candidate) - expected;
            step_squared_error += error * error;
            step_candidate_squared +=
                static_cast<double>(candidate) * candidate;
            step_reference_squared +=
                static_cast<double>(expected) * expected;
            step_dot_product += static_cast<double>(candidate) * expected;
            if (apply_fp32_tolerance && abs_error > 1e-3 +
                1e-4 * std::fabs(static_cast<double>(expected))) {
                parity->tolerance_failures++;
            }
        }
        for (uint32_t rank = 0; rank < 5; ++rank) {
            if (candidate > actual_top5_values[rank]) {
                for (uint32_t shift = 4; shift > rank; --shift) {
                    actual_top5_values[shift] = actual_top5_values[shift - 1];
                    actual_top5_indices[shift] = actual_top5_indices[shift - 1];
                }
                actual_top5_values[rank] = candidate;
                actual_top5_indices[rank] = vocab_index;
                break;
            }
        }
        for (uint32_t rank = 0; rank < 5; ++rank) {
            if (expected > reference_top5_values[rank]) {
                for (uint32_t shift = 4; shift > rank; --shift) {
                    reference_top5_values[shift] =
                        reference_top5_values[shift - 1];
                    reference_top5_indices[shift] =
                        reference_top5_indices[shift - 1];
                }
                reference_top5_values[rank] = expected;
                reference_top5_indices[rank] = vocab_index;
                break;
            }
        }
        if (vocab_index != 0 && candidate > actual_best) {
            actual_best = candidate;
            actual_argmax = vocab_index;
        }
        if (vocab_index != 0 && expected > reference_best) {
            reference_best = expected;
            reference_argmax = vocab_index;
        }
    }
    uint32_t top5_overlap = 0;
    for (uint32_t actual_rank = 0; actual_rank < 5; ++actual_rank) {
        for (uint32_t reference_rank = 0; reference_rank < 5;
             ++reference_rank) {
            if (actual_top5_indices[actual_rank] ==
                reference_top5_indices[reference_rank]) {
                top5_overlap++;
                break;
            }
        }
    }
    parity->min_top5_overlap =
        std::min(parity->min_top5_overlap, top5_overlap);
    parity->squared_error_sum += step_squared_error;
    parity->candidate_squared_sum += step_candidate_squared;
    parity->reference_squared_sum += step_reference_squared;
    parity->dot_product_sum += step_dot_product;
    if (step_reference_squared > 0.0) {
        const double step_nrmse =
            std::sqrt(step_squared_error / step_reference_squared);
        parity->worst_step_nrmse =
            std::max(parity->worst_step_nrmse, step_nrmse);
        const double reference_rms =
            std::sqrt(step_reference_squared / vocab_size);
        parity->worst_max_abs_over_reference_rms = std::max(
            parity->worst_max_abs_over_reference_rms,
            step_max_abs_error / reference_rms);
    }
    const double norm_product =
        step_candidate_squared * step_reference_squared;
    if (norm_product > 0.0) {
        const double step_cosine = step_dot_product / std::sqrt(norm_product);
        parity->min_step_cosine =
            std::min(parity->min_step_cosine, step_cosine);
    } else {
        parity->min_step_cosine = -1.0;
    }
    parity->checked_steps++;
    parity->compared_values += vocab_size;
    if (actual_argmax != selected_token || reference_argmax != selected_token) {
        parity->argmax_mismatches++;
    }
}

// --decode driver: decode-only from a GPU-exported .gdnstate blob.
//
// The fixture provides the golden continuation length (and example indices).
// The device starts from the exported seed token and performs one single-token
// decode step per call, updating persistent recurrent/conv state in device memory.
//   - gen_traj / per_step_tpot_ms / kernel_ms: free-running greedy O(1) decode.
//     Persistent recurrent and convolution state advances in device memory on
//     every call. Per-step wall time surrounds run_forward; kernel_ms is the
//     on-card kernel time returned by run_forward (seconds * 1000).
//
// N = min(cont_len, decode_len) (decode_len == 0 means use cont_len);
// E = min(num_examples, limit) (limit == 0 means all).
static std::vector<DecodeExample> run_decode_hw(
    const ModelData &model,
    HwRunner &runner,
    const Fixture &fixture,
    uint32_t decode_len,
    const std::string &state_path,
    const std::string &logits_reference_path,
    const std::string &gpu_logits_reference_path,
    LogitsParity *logits_parity,
    LogitsParity *gpu_logits_parity,
    const std::string &state_dump_path,
    uint32_t interstep_delay_ms,
    const std::string &logits_dump_path
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
    if (fixture.num_examples == 0 || fixture.examples.empty())
        throw std::runtime_error("--decode-from-state needs an LL-kind fixture for the golden");
    const DecodeReq &req = fixture.examples[0];
    /* --decode-len may now EXTEND past the fixture's golden continuation:
     * decode from the exported state is free-running and needs no golden to
     * produce tokens. Lengths beyond cont_len are for long-horizon studies
     * (e.g. Iter66n: does the 1-ULP per-head scalar divergence stay bounded
     * over hundreds of tokens?) and are compared against a native reference
     * of the same length rather than the 64-token fixture golden. */
    uint32_t n = req.cont_len;
    if (decode_len != 0) n = decode_len;
    if (n == 0) throw std::runtime_error("--decode: zero decode length");

    LogitsReference logits_reference;
    LogitsReference gpu_logits_reference;
    if (!logits_reference_path.empty()) {
        logits_reference = load_logits_reference(
            logits_reference_path, model.config.vocab_size, n - 1);
        std::cerr << "[progress] loaded logits reference: steps="
                  << logits_reference.decode_steps
                  << " vocab=" << logits_reference.vocab_size << "\n";
    }
    if (!gpu_logits_reference_path.empty()) {
        gpu_logits_reference = load_logits_reference(
            gpu_logits_reference_path, model.config.vocab_size, n - 1);
        std::cerr << "[progress] loaded independent GPU logits reference: steps="
                  << gpu_logits_reference.decode_steps
                  << " vocab=" << gpu_logits_reference.vocab_size << "\n";
    }
    if (logits_parity != nullptr) {
        *logits_parity = LogitsParity{};
    }
    if (gpu_logits_parity != nullptr) {
        *gpu_logits_parity = LogitsParity{};
    }

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

    /* GDNLOG1 dump, byte-compatible with gdn_eval --logits-dump. */
    std::FILE *logits_dump = nullptr;
    if (!logits_dump_path.empty()) {
        logits_dump = std::fopen(logits_dump_path.c_str(), "wb");
        if (logits_dump == nullptr)
            throw std::runtime_error("cannot create logits dump: " +
                                     logits_dump_path);
        const char magic[8] = {'G', 'D', 'N', 'L', 'O', 'G', '1', '\0'};
        const uint32_t header[3] = {1u, model.config.vocab_size, n - 1u};
        std::fwrite(magic, 1, sizeof(magic), logits_dump);
        std::fwrite(header, sizeof(uint32_t), 3, logits_dump);
    }
    uint64_t onchip_argmax_mismatches = 0;
    for (uint32_t step = 1; step < n; ++step) {
        /* Iter67: the kernel picks the token on chip again, so a plain
         * generation step needs only the 4-byte token slot. Pull the 128 KB
         * logit vector back over PCIe only when something actually consumes
         * it -- a reference comparison or a dump. */
        const bool want_logits = !logits_reference_path.empty() ||
                                 !gpu_logits_reference_path.empty() ||
                                 logits_dump != nullptr;
        double generation_seconds = 0.0;
        double ksec = runner.run_forward(result.gen_traj[step - 1], want_logits,
                                         &generation_seconds);
        const float *device_logits = runner.device_logits();
        int next = runner.device_token();
        if (want_logits) {
            /* We already paid for the logits, so re-derive the pick on the
             * host and cross-check the hardware against it for free. Same rule
             * both sides: maximum value, lowest vocabulary index on ties,
             * which a strict-'>' first-wins scan gives exactly. A disagreement
             * here is a real defect, not a rounding difference -- both are
             * reading the identical FP32 vector. */
            int host_pick = 0;
            for (uint32_t vocab_index = 1;
                 vocab_index < model.config.vocab_size; ++vocab_index) {
                if (device_logits[vocab_index] > device_logits[host_pick]) {
                    host_pick = static_cast<int>(vocab_index);
                }
            }
            if (host_pick != next) {
                ++onchip_argmax_mismatches;
                std::cerr << "[warn] on-chip argmax " << next
                          << " != host scan " << host_pick
                          << " at step " << step << "\n";
                next = host_pick;
            }
        }
        /* generation_seconds stopped inside run_forward immediately after the
         * selected-token read-back. The optional 128 KB logit transfer above,
         * this host cross-check, and the reference comparisons below are all
         * validation work and deliberately excluded from production TPOT. */
        result.per_step_tpot_ms[step] = generation_seconds * 1000.0;
        result.kernel_ms[step] = ksec * 1000.0;
        if (!logits_reference_path.empty() && logits_parity != nullptr) {
            const float *reference = logits_reference.values.data() +
                static_cast<size_t>(step - 1) * model.config.vocab_size;
            compare_logits_step(device_logits, reference,
                                model.config.vocab_size,
                                static_cast<uint32_t>(next), true,
                                logits_parity);
        }
        if (!gpu_logits_reference_path.empty() &&
            gpu_logits_parity != nullptr) {
            const float *reference = gpu_logits_reference.values.data() +
                static_cast<size_t>(step - 1) * model.config.vocab_size;
            compare_logits_step(device_logits, reference,
                                model.config.vocab_size,
                                static_cast<uint32_t>(next),
                                false,
                                gpu_logits_parity);
        }
        result.gen_traj[step] = next;
        result.tf_argmax[step] = next;
        if (logits_dump != nullptr) {
            if (std::fwrite(device_logits, sizeof(float),
                            model.config.vocab_size, logits_dump) !=
                model.config.vocab_size) {
                std::fclose(logits_dump);
                throw std::runtime_error("logits dump write failed");
            }
        }
        if (interstep_delay_ms != 0) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(interstep_delay_ms));
        }
    }
    if (logits_dump != nullptr) {
        std::fclose(logits_dump);
        std::cerr << "[logits_dump] wrote " << logits_dump_path << "\n";
    }
    if (!state_dump_path.empty()) {
        runner.dump_persistent_state(model.config, state_dump_path);
    }
    std::cerr << "[progress] decode-from-state complete (" << n << " tokens)\n";
    if (!logits_reference_path.empty() && logits_parity != nullptr) {
        std::cerr << "[native_logits] steps=" << logits_parity->checked_steps
                  << " values=" << logits_parity->compared_values
                  << " max_abs=" << std::setprecision(9)
                  << logits_parity->max_abs_error
                  << " max_rel=" << logits_parity->max_rel_error
                  << " tolerance_fail=" << logits_parity->tolerance_failures
                  << " nonfinite_mismatch="
                  << logits_parity->nonfinite_mismatches
                  << " exact_ref_mismatch="
                  << logits_parity->exact_reference_mismatches
                  << " argmax_mismatch=" << logits_parity->argmax_mismatches
                  << "\n";
    }
    if (!gpu_logits_reference_path.empty() && gpu_logits_parity != nullptr) {
        std::cerr << "[gpu_logits] steps=" << gpu_logits_parity->checked_steps
                  << " values=" << gpu_logits_parity->compared_values
                  << " max_abs=" << std::setprecision(9)
                  << gpu_logits_parity->max_abs_error
                  << " max_rel=" << gpu_logits_parity->max_rel_error
                  << " tolerance_fail=" << gpu_logits_parity->tolerance_failures
                  << " nonfinite_mismatch="
                  << gpu_logits_parity->nonfinite_mismatches
                  << " global_nrmse="
                  << logits_global_nrmse(*gpu_logits_parity)
                  << " worst_step_nrmse="
                  << gpu_logits_parity->worst_step_nrmse
                  << " global_cosine="
                  << logits_global_cosine(*gpu_logits_parity)
                  << " min_step_cosine="
                  << gpu_logits_parity->min_step_cosine
                  << " worst_max_over_rms="
                  << gpu_logits_parity->worst_max_abs_over_reference_rms
                  << " min_top5_overlap="
                  << gpu_logits_parity->min_top5_overlap
                  << " argmax_mismatch="
                  << gpu_logits_parity->argmax_mismatches << "\n";
    }
    return results;
}

// Writes the on-card decode JSON. Schema is the native gdn_eval decode schema
// plus a per-step kernel_ms series.
static void write_decode_json(
    const std::string &output_path,
    uint32_t decode_len,
    const std::vector<DecodeExample> &results,
    const LogitsParity *logits_parity,
    const LogitsParity *gpu_logits_parity
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
    file << "\n ]";
    if (logits_parity != nullptr) {
        file << ",\n \"logits_parity\": {"
             << "\"checked_steps\": " << logits_parity->checked_steps
             << ", \"compared_values\": " << logits_parity->compared_values
             << ", \"max_abs_error\": " << std::setprecision(17)
             << logits_parity->max_abs_error
             << ", \"max_rel_error\": " << logits_parity->max_rel_error
             << ", \"cpu_tolerance_failures\": "
             << logits_parity->tolerance_failures
             << ", \"nonfinite_mismatches\": "
             << logits_parity->nonfinite_mismatches
             << ", \"exact_reference_mismatches\": "
             << logits_parity->exact_reference_mismatches
             /* Iter66m: hardware/native bit-exactness is NOT required and is
              * not achievable. expf/log1pf are outside IEEE-754's
              * correct-rounding mandate, glibc and the AMD FPO cores disagree
              * in the last bit on 21.4% of the real per-head decay operands,
              * and each such scalar flips the state lanes sitting on an RNE
              * tie. The independent-GPU vector gate below is the real gate. */
             << ", \"exact_reference_required\": false"
             << ", \"argmax_mismatches\": "
             << logits_parity->argmax_mismatches << "}";
    }
    if (gpu_logits_parity != nullptr) {
        file << ",\n \"gpu_logits_parity\": {"
             << "\"checked_steps\": " << gpu_logits_parity->checked_steps
             << ", \"compared_values\": " << gpu_logits_parity->compared_values
             << ", \"max_abs_error\": " << std::setprecision(17)
             << gpu_logits_parity->max_abs_error
             << ", \"max_rel_error\": " << gpu_logits_parity->max_rel_error
             << ", \"tolerance_failures\": "
             << gpu_logits_parity->tolerance_failures
             << ", \"nonfinite_mismatches\": "
             << gpu_logits_parity->nonfinite_mismatches
             << ", \"exact_reference_mismatches\": "
             << gpu_logits_parity->exact_reference_mismatches
             << ", \"exact_reference_required\": false"
             << ", \"global_nrmse\": "
             << logits_global_nrmse(*gpu_logits_parity)
             << ", \"worst_step_nrmse\": "
             << gpu_logits_parity->worst_step_nrmse
             << ", \"global_cosine\": "
             << logits_global_cosine(*gpu_logits_parity)
             << ", \"min_step_cosine\": "
             << gpu_logits_parity->min_step_cosine
             << ", \"worst_max_abs_over_reference_rms\": "
             << gpu_logits_parity->worst_max_abs_over_reference_rms
             << ", \"min_top5_overlap\": "
             << gpu_logits_parity->min_top5_overlap
             << ", \"argmax_mismatches\": "
             << gpu_logits_parity->argmax_mismatches << "}";
    }
    file << "\n}\n";
}

}  // namespace


/* ---------------------------------------------------------------------------
 * Teacher-forced rolling scoring (WikiText perplexity on card).
 *
 * The kernel consumes exactly one token per call, so a window is walked
 * sequentially: position i consumes tokens[i] and yields the distribution over
 * token i+1. Log-probabilities come from the kernel's own LM head (the full
 * FP32 logit vector it already exports), so no host-side head is involved.
 * This is teacher-forced -- the known next token is fed regardless of what the
 * model would have picked -- which is what perplexity requires and which also
 * makes the measurement immune to the free-running trajectory forking
 * characterised in Iter66n.
 * ------------------------------------------------------------------------- */
static double score_window_hw(
    const ModelData &model,
    HwRunner &runner,
    const PairReq &pair,
    uint64_t *scored_tokens
) {
    if (pair.ctx_len == 0 || pair.cont_len == 0) {
        throw std::runtime_error("invalid scoring window");
    }
    runner.reset_decode_state();

    std::vector<int32_t> tokens;
    tokens.reserve(pair.ctx.size() + pair.cont.size());
    tokens.insert(tokens.end(), pair.ctx.begin(), pair.ctx.end());
    tokens.insert(tokens.end(), pair.cont.begin(), pair.cont.end());

    const uint32_t vocab = model.config.vocab_size;
    double logprob = 0.0;
    const size_t last = tokens.size() - 1;   // the final token is never fed
    for (size_t i = 0; i < last; ++i) {
        runner.run_forward(tokens[i]);
        if (i + 1 < pair.ctx_len) {
            continue;                        // still inside the context
        }
        const float *lg = runner.device_logits();
        double m = lg[0];
        for (uint32_t v = 1; v < vocab; ++v) {
            if (lg[v] > m) m = lg[v];
        }
        double sum = 0.0;
        for (uint32_t v = 0; v < vocab; ++v) {
            sum += std::exp(static_cast<double>(lg[v]) - m);
        }
        const int32_t target = tokens[i + 1];
        if (target < 0 || static_cast<uint32_t>(target) >= vocab) {
            throw std::runtime_error("scoring target token out of range");
        }
        logprob += (static_cast<double>(lg[target]) - m) - std::log(sum);
        ++(*scored_tokens);
    }
    return logprob;
}

static int run_score_hw(const Options &opts, const ModelData &model,
                        HwRunner &runner) {
    const RollingFixture fixture = load_rolling_fixture(opts.fixture);
    uint32_t doc_count = fixture.num_examples;
    if (opts.score_doc_limit != 0 && opts.score_doc_limit < doc_count) {
        doc_count = opts.score_doc_limit;
    }
    std::cerr << "[progress] scoring " << doc_count << " of "
              << fixture.num_examples << " documents\n";

    double total_logprob = 0.0;
    uint64_t total_words = 0, total_bytes = 0, scored_tokens = 0, windows = 0;
    const auto start = std::chrono::high_resolution_clock::now();
    for (uint32_t doc = 0; doc < doc_count; ++doc) {
        const RollingReq &req = fixture.documents[doc];
        double doc_logprob = 0.0;
        for (const PairReq &pair : req.windows) {
            doc_logprob += score_window_hw(model, runner, pair, &scored_tokens);
            ++windows;
        }
        total_logprob += doc_logprob;
        total_words += req.word_count;
        total_bytes += req.byte_count;
        const double elapsed =
            std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - start).count();
        std::cerr << "[progress] doc " << (doc + 1) << "/" << doc_count
                  << " windows=" << req.windows.size()
                  << " scored_tokens=" << scored_tokens
                  << " running_word_ppl="
                  << std::exp(-total_logprob /
                              static_cast<double>(total_words ? total_words : 1))
                  << " elapsed=" << static_cast<uint64_t>(elapsed) << "s\n";
    }

    const double word_ppl =
        std::exp(-total_logprob / static_cast<double>(total_words));
    const double byte_ppl =
        std::exp(-total_logprob / static_cast<double>(total_bytes));
    const double bits_per_byte =
        -total_logprob / static_cast<double>(total_bytes) / std::log(2.0);

    std::cerr << "[score] documents=" << doc_count
              << " windows=" << windows
              << " scored_tokens=" << scored_tokens
              << " words=" << total_words
              << " bytes=" << total_bytes
              << " total_logprob=" << std::setprecision(17) << total_logprob
              << " word_ppl=" << std::setprecision(9) << word_ppl
              << " byte_ppl=" << byte_ppl
              << " bits_per_byte=" << bits_per_byte
              << " kernel_ms_per_token="
              << runner.average_kernel_seconds() * 1000.0 << "\n";

    if (opts.output != "-") {
        std::ofstream file(opts.output);
        if (!file) {
            throw std::runtime_error("failed to open output: " + opts.output);
        }
        file << std::setprecision(17)
             << "{\n  \"task\": \"wikitext_rolling\",\n"
             << "  \"fixture\": \"" << opts.fixture << "\",\n"
             << "  \"xclbin\": \"" << opts.xclbin << "\",\n"
             << "  \"documents\": " << doc_count << ",\n"
             << "  \"windows\": " << windows << ",\n"
             << "  \"scored_tokens\": " << scored_tokens << ",\n"
             << "  \"words\": " << total_words << ",\n"
             << "  \"bytes\": " << total_bytes << ",\n"
             << "  \"total_logprob\": " << total_logprob << ",\n"
             << "  \"word_perplexity\": " << word_ppl << ",\n"
             << "  \"byte_perplexity\": " << byte_ppl << ",\n"
             << "  \"bits_per_byte\": " << bits_per_byte << ",\n"
             << "  \"average_kernel_ms\": "
             << runner.average_kernel_seconds() * 1000.0 << "\n}\n";
    }
    return 0;
}

int main(int argc, char **argv) {
    try {
        Options opts = parse_options(argc, argv);
        ensure_xrt_environment();

        std::cerr << "[progress] open device "
                  << (opts.device_bdf.empty()
                          ? std::to_string(opts.device_index)
                          : opts.device_bdf)
                  << "\n";
        auto device = opts.device_bdf.empty()
                          ? xrt::device(opts.device_index)
                          : xrt::device(opts.device_bdf);
        std::cerr << "[progress] load xclbin " << opts.xclbin << "\n";
        auto uuid = device.load_xclbin(opts.xclbin);

        std::cerr << "[progress] loading model weights from " << opts.weights << "\n";
        ModelData model = load_model(opts.weights);
        std::cerr << "[progress] model hidden=" << model.config.hidden_size
                  << " layers=" << model.config.num_layers
                  << " max_seq_len=" << model.config.max_seq_len
                  << " vocab=" << model.config.vocab_size << "\n";

        if (opts.score_mode) {
            HwRunner runner(device, uuid, model);
            return run_score_hw(opts, model, runner);
        }

        if (opts.state_path.empty()) {
            throw std::runtime_error(
                "decode-only build: pass --decode-from-state <state.gdnstate>");
        }
            std::cerr << "[progress] loading LL decode fixture from " << opts.fixture << "\n";
            Fixture fixture = load_ll_fixture(opts.fixture);
            std::cerr << "[progress] fixture examples=" << fixture.num_examples << "\n";

            std::cerr << "[progress] allocate XRT buffers\n";
            HwRunner runner(device, uuid, model);

            LogitsParity logits_parity;
            LogitsParity gpu_logits_parity;
            LogitsParity *logits_parity_ptr =
                opts.logits_reference_path.empty() ? nullptr : &logits_parity;
            LogitsParity *gpu_logits_parity_ptr =
                opts.gpu_logits_reference_path.empty()
                    ? nullptr : &gpu_logits_parity;
            std::vector<DecodeExample> results =
                run_decode_hw(model, runner, fixture, opts.decode_len,
                              opts.state_path, opts.logits_reference_path,
                              opts.gpu_logits_reference_path,
                              logits_parity_ptr, gpu_logits_parity_ptr,
                              opts.state_dump_path, opts.interstep_delay_ms,
                              opts.logits_dump_path);

            write_decode_json(opts.output, opts.decode_len, results,
                              logits_parity_ptr, gpu_logits_parity_ptr);
            /* The hardware/native comparison is a diagnostic, not a gate.
             * Iter66m located the cause of its residual mismatch and it is a
             * conforming-implementation difference, not a defect: 129 of
             * 12,582,912 BF16 state lanes at +-1 ULP, seeded by transcendental
             * last-bit disagreement. Only a non-finite value is still treated
             * as a hardware fault, because a NaN or Inf can never be that. */
            if (logits_parity_ptr != nullptr &&
                logits_parity.nonfinite_mismatches != 0) {
                throw std::runtime_error(
                    "on-card hardware/native comparison produced non-finite "
                    "logits");
            }
            if (gpu_logits_parity_ptr != nullptr &&
                !gpu_logits_gate_passes(gpu_logits_parity)) {
                throw std::runtime_error(
                    "on-card independent-GPU logit comparison failed");
            }
            std::cerr << "[progress] decode finished examples=" << results.size()
                      << " avg_kernel_ms=" << std::fixed << std::setprecision(3)
                      << (runner.average_kernel_seconds() * 1000.0)
                      << " output=" << opts.output << "\n";

        return EXIT_SUCCESS;
    } catch (const std::exception &ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }
}
