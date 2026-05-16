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
constexpr const char *kDefaultXrt = "/opt/xilinx/xrt";
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
};

static void usage(const char *argv0) {
    std::cerr
        << "usage: " << argv0
        << " <gdn_forward.xclbin> [weights.gdnw] [wikitext.gdnreq]"
        << " [output.json|-] [device_index] [max_examples]\n\n"
        << "defaults:\n"
        << "  weights      " << kDefaultWeights << "\n"
        << "  fixture      " << kDefaultFixture << "\n"
        << "  output       " << kDefaultOutput << "\n"
        << "  device_index 0\n"
        << "  max_examples 0 (all examples)\n";
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
    if (argc < 2 || argc > 7 || std::strcmp(argv[1], "--help") == 0) {
        usage(argv[0]);
        std::exit(argc < 2 ? 1 : 0);
    }
    opts.xclbin = argv[1];
    if (argc > 2) opts.weights = argv[2];
    if (argc > 3) opts.fixture = argv[3];
    if (argc > 4) opts.output = argv[4];
    if (argc > 5) opts.device_index = parse_u32(argv[5], "device_index");
    if (argc > 6) opts.max_examples = parse_u32(argv[6], "max_examples");
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
          max_tokens_(model.config.max_seq_len),
          hidden_(model.config.hidden_size),
          config_bo_(device, sizeof(GDNWeightHeader), kernel_.group_id(0)),
          weight_bo_(device, model.weight_data.size() * sizeof(float), kernel_.group_id(1)),
          x_bo_(device, hidden_bytes(model), kernel_.group_id(3)),
          x_norm_bo_(device, hidden_bytes(model), kernel_.group_id(4)),
          q_bo_(device, hidden_bytes(model), kernel_.group_id(5)),
          k_bo_(device, hidden_bytes(model), kernel_.group_id(6)),
          v_bo_(device, hidden_bytes(model), kernel_.group_id(7)),
          a_bo_(device, head_bytes(model), kernel_.group_id(8)),
          b_bo_(device, head_bytes(model), kernel_.group_id(9)),
          gate_bo_(device, hidden_bytes(model), kernel_.group_id(10)),
          attn_bo_(device, hidden_bytes(model), kernel_.group_id(11)),
          tmp_hidden_bo_(device, hidden_bytes(model), kernel_.group_id(12)),
          mlp_gate_bo_(device, mlp_bytes(model), kernel_.group_id(13)),
          mlp_up_bo_(device, mlp_bytes(model), kernel_.group_id(14)),
          recurrent_state_bo_(device, recurrent_state_bytes(model), kernel_.group_id(15)),
          head_buffer_bo_(device, head_buffer_bytes(model), kernel_.group_id(16)),
          tokens_bo_(device, static_cast<size_t>(max_tokens_) * sizeof(int32_t), kernel_.group_id(17)),
          x_norm_host_(static_cast<size_t>(max_tokens_) * hidden_, 0.0f) {
        std::cerr << "[progress] upload config and weights to device\n";
        config_bo_.write(&model.config, sizeof(GDNWeightHeader), 0);
        sync_bo_chunked(config_bo_, XCL_BO_SYNC_BO_TO_DEVICE, sizeof(GDNWeightHeader), 0);
        const size_t weight_bytes = model.weight_data.size() * sizeof(float);
        weight_bo_.write(model.weight_data.data(), weight_bytes, 0);
        sync_bo_chunked(weight_bo_, XCL_BO_SYNC_BO_TO_DEVICE, weight_bytes, 0);
    }

    // On this xocl/xdma driver (XRT 2022.1, U55C) the per-call sync transfer
    // size caps out at 16 MiB; anything larger returns EINVAL/EIO. Wrap every
    // sync that might exceed that with this helper.
    static constexpr size_t kSyncChunk = 16ULL * 1024 * 1024;  // 16 MiB
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

        size_t token_bytes = tokens.size() * sizeof(int32_t);
        tokens_bo_.write(tokens.data(), token_bytes, 0);
        sync_bo_chunked(tokens_bo_, XCL_BO_SYNC_BO_TO_DEVICE, token_bytes, 0);

        // Kernel-call construction sets ~18 AXI-Lite arg registers and writes
        // ap_start (host-side PCIe overhead, ~100-200 µs, scales with arg
        // count). Start the clock AFTER that so kernel_ms reflects only FPGA
        // execution. The kernel has actually been running for a few µs by the
        // time `start` is sampled, but that skew is <1 part in 10^6 of a
        // multi-minute run and is dwarfed by host-clock resolution anyway.
        auto run = kernel_(
            config_bo_,
            weight_bo_,
            max_tokens_,
            x_bo_,
            x_norm_bo_,
            q_bo_,
            k_bo_,
            v_bo_,
            a_bo_,
            b_bo_,
            gate_bo_,
            attn_bo_,
            tmp_hidden_bo_,
            mlp_gate_bo_,
            mlp_up_bo_,
            recurrent_state_bo_,
            head_buffer_bo_,
            tokens_bo_,
            static_cast<uint32_t>(tokens.size())
        );
        auto start = std::chrono::high_resolution_clock::now();
        run.wait();
        auto end = std::chrono::high_resolution_clock::now();

        double seconds = std::chrono::duration<double>(end - start).count();
        total_kernel_seconds_ += seconds;
        kernel_runs_ += 1;

        size_t x_norm_bytes = tokens.size() * static_cast<size_t>(hidden_) * sizeof(float);
        sync_bo_chunked(x_norm_bo_, XCL_BO_SYNC_BO_FROM_DEVICE, x_norm_bytes, 0);
        x_norm_bo_.read(x_norm_host_.data(), x_norm_bytes, 0);

        return seconds;
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
        return static_cast<size_t>(model.config.max_seq_len) *
               model.config.hidden_size *
               sizeof(float);
    }

    static size_t head_bytes(const ModelData &model) {
        return static_cast<size_t>(model.config.max_seq_len) *
               model.config.num_heads *
               sizeof(float);
    }

    static size_t mlp_bytes(const ModelData &model) {
        return static_cast<size_t>(model.config.max_seq_len) *
               model.config.intermediate_size *
               sizeof(float);
    }

    static size_t recurrent_state_bytes(const ModelData &model) {
        size_t value_dim = model.config.hidden_size / model.config.num_heads;
        return static_cast<size_t>(model.config.num_heads) *
               model.config.head_dim *
               value_dim *
               sizeof(float);
    }

    static size_t head_buffer_bytes(const ModelData &model) {
        size_t value_dim = model.config.hidden_size / model.config.num_heads;
        size_t rounded = ((value_dim + 15) / 16) * 16;
        return rounded * sizeof(float);
    }

    xrt::kernel kernel_;
    uint32_t max_tokens_ = 0;
    uint32_t hidden_ = 0;
    xrt::bo config_bo_;
    xrt::bo weight_bo_;
    xrt::bo x_bo_;
    xrt::bo x_norm_bo_;
    xrt::bo q_bo_;
    xrt::bo k_bo_;
    xrt::bo v_bo_;
    xrt::bo a_bo_;
    xrt::bo b_bo_;
    xrt::bo gate_bo_;
    xrt::bo attn_bo_;
    xrt::bo tmp_hidden_bo_;
    xrt::bo mlp_gate_bo_;
    xrt::bo mlp_up_bo_;
    xrt::bo recurrent_state_bo_;
    xrt::bo head_buffer_bo_;
    xrt::bo tokens_bo_;
    std::vector<float> x_norm_host_;
    double total_kernel_seconds_ = 0.0;
    uint64_t kernel_runs_ = 0;
};

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
