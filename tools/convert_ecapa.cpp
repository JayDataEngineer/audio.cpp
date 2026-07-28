// convert_ecapa.cpp — Convert ECAPA-TDNN speaker-encoder HuggingFace
// safetensors to a standalone GGUF for audio.cpp's Qwen3SpeakerEncoderRuntime.
//
// Source model class: EcapaTdnnSpeakerEncoder
//   - marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B  (standalone, 2048-dim, ~23 MB BF16)
//   - Qwen3-TTS-*-Base/model.safetensors              (bundled under speaker_encoder.* prefix)
//   - Qwen3-TTS-0.6B-Base                             (bundled, 1024-dim, ~6 MB BF16, "mini" variant)
//
// The converter preserves source tensor names with a configurable prefix so
// the runtime can bind them via load_weights() in src/models/qwen3_tts/speaker_encoder.cpp.
// Output GGUF is self-describing: KV metadata captures enc_dim, mel_dim,
// res2net_scale, se_channels, attention_channels, sample_rate, enc_channels[],
// enc_kernel_sizes[], enc_dilations[] so the runtime can sanity-check the
// architecture before binding.
//
// Usage:
//   convert_ecapa <model_dir> <output.gguf>
//   convert_ecapa <model_dir> <output.gguf> --filter-prefix speaker_encoder.
//       # Only emit tensors whose name starts with the prefix; strip the
//       # prefix before adding the output prefix. Used when the source
//       # safetensors bundles the ECAPA weights inside a larger talker
//       # checkpoint (e.g. Qwen3-TTS-1.7B-Base/model.safetensors, 3.6 GB).
//   convert_ecapa <model_dir> <output.gguf> --output-prefix speaker.
//       # Use a non-default output prefix. audio.cpp's runtime expects
//       # `speaker_encoder.*` (the default); audiocore expects `speaker.*`.
//   convert_ecapa <model_dir> <output.gguf> --strip-prefix
//       # Source tensors are ALREADY named correctly — emit them verbatim
//       # WITHOUT adding any prefix.
//
// `model_dir` must contain config.json and one .safetensors file.
//
// Build (in repo root):
//   cmake -S . -B build -DENGINE_BUILD_TESTS=ON
//   cmake --build build --target convert_ecapa

#include "engine/framework/io/json.h"

#include <ggml.h>
#include <gguf.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <memory>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;
namespace json = engine::io::json;

// ── BF16 → F16 conversion ──────────────────────────────────────────────────
//
// The marksverdhei checkpoint ships BF16 weights, but ggml's CPU conv_1d
// kernel (used by the ECAPA encoder's same_conv1d_) requires F16 src0.
// Qwen3-TTS talker GGUFs are also F16, so we normalize to F16 here.
// Conversion goes BF16 → FP32 (lossless: BF16 is the upper 16 bits of FP32)
// → F16 (standard half-precision rounding, round-to-nearest-even).
static uint16_t f32_bits_to_f16_bits(uint32_t f) {
    uint32_t sign = (f >> 16) & 0x8000;
    int32_t  exp  = (int32_t)((f >> 23) & 0xff) - 127 + 15;
    uint32_t mant = f & 0x7fffff;

    if (exp >= 0x1f) {                       // overflow → Inf (or NaN if mant)
        return (uint16_t)(sign | 0x7c00u | (mant ? 0x200u : 0u));
    }
    if (exp <= 0) {                          // zero or subnormal
        if (exp < -10) return (uint16_t)sign;  // underflow → signed zero
        mant |= 0x800000u;                     // implicit leading 1
        const uint32_t shift = (uint32_t)(14 - exp);
        const uint32_t half  = 1u << (shift - 1);
        uint32_t m = mant >> shift;
        uint32_t rem = mant & ((half << 1) - 1);
        if (rem > half || (rem == half && (m & 1u))) m++;  // RTE
        return (uint16_t)(sign | m);
    }
    // Normal: round 23-bit mantissa to 10 bits (RTE).
    const uint32_t shift = 13;
    const uint32_t half  = 1u << (shift - 1);
    uint32_t m = mant >> shift;
    uint32_t rem = mant & ((half << 1) - 1);
    if (rem > half || (rem == half && (m & 1u))) m++;
    if (m > 0x3ffu) { m = 0; exp++; }
    if (exp >= 0x1f) return (uint16_t)(sign | 0x7c00u);
    return (uint16_t)(sign | (uint32_t(exp) << 10) | m);
}

static void bf16_buf_to_f16(const void* src, size_t n_elements,
                            std::vector<uint16_t>* out) {
    out->resize(n_elements);
    const uint16_t* s = static_cast<const uint16_t*>(src);
    for (size_t i = 0; i < n_elements; i++) {
        uint32_t f32 = uint32_t(s[i]) << 16;              // BF16 → FP32 (lossless)
        (*out)[i] = f32_bits_to_f16_bits(f32);            // FP32 → F16 (RTE)
    }
}

[[noreturn]] static void die(const char* msg) {
    std::fprintf(stderr, "error: %s\n", msg);
    std::exit(1);
}

template <typename... Args>
[[noreturn]] static void die_f(const char* fmt, const Args&... args) {
    std::fprintf(stderr, "error: ");
    std::fprintf(stderr, fmt, args...);
    std::fprintf(stderr, "\n");
    std::exit(1);
}

// ── JSON helpers (engine::io::json is nlohmann-shaped but has no .value()) ──

static uint32_t get_u32(const json::Value& obj, const std::string& key, uint32_t fallback) {
    const json::Value* v = obj.find(key);
    if (v != nullptr && v->is_number()) {
        return static_cast<uint32_t>(v->as_i64());
    }
    return fallback;
}

static std::vector<uint32_t> get_u32_array(
    const json::Value& obj, const std::string& key,
    std::vector<uint32_t> fallback) {
    const json::Value* v = obj.find(key);
    if (v == nullptr || !v->is_array()) {
        return fallback;
    }
    std::vector<uint32_t> out;
    out.reserve(v->as_array().size());
    for (const auto& item : v->as_array()) {
        if (item.is_number()) {
            out.push_back(static_cast<uint32_t>(item.as_i64()));
        }
    }
    if (out.empty()) return fallback;
    return out;
}

// Resolve a value with two-level fallback: prefer spk_cfg, then root, then default.
static uint32_t get_u32_chained(
    const json::Value& root, const json::Value* spk_cfg,
    const std::string& key, uint32_t fallback) {
    if (spk_cfg != nullptr) {
        const json::Value* v = spk_cfg->find(key);
        if (v != nullptr && v->is_number()) {
            return static_cast<uint32_t>(v->as_i64());
        }
    }
    return get_u32(root, key, fallback);
}

static std::vector<uint32_t> get_u32_array_chained(
    const json::Value& root, const json::Value* spk_cfg,
    const std::string& key, std::vector<uint32_t> fallback) {
    if (spk_cfg != nullptr) {
        const json::Value* v = spk_cfg->find(key);
        if (v != nullptr && v->is_array()) {
            std::vector<uint32_t> out;
            out.reserve(v->as_array().size());
            for (const auto& item : v->as_array()) {
                if (item.is_number()) out.push_back(static_cast<uint32_t>(item.as_i64()));
            }
            if (!out.empty()) return out;
        }
    }
    return get_u32_array(root, key, fallback);
}

// ── Safetensors reader (single-file, mmap) ──────────────────────────────────

struct SSTensor {
    std::string name;
    std::string dtype;            // "F32", "F16", "BF16"
    std::vector<int64_t> shape;
    size_t offset = 0;            // byte offset within the mmap
    size_t nbytes = 0;
};

struct SafeTensorFile {
    char* data = nullptr;
    size_t size = 0;
    int fd = -1;
    std::vector<SSTensor> tensors;

    ~SafeTensorFile() {
        if (data && data != MAP_FAILED) munmap(data, size);
        if (fd >= 0) close(fd);
    }

    const SSTensor* find(const std::string& name) const {
        for (auto& t : tensors) if (t.name == name) return &t;
        return nullptr;
    }
};

static std::unique_ptr<SafeTensorFile> load_safetensors(const fs::path& path) {
    auto out = std::make_unique<SafeTensorFile>();
    out->fd = ::open(path.c_str(), O_RDONLY);
    if (out->fd < 0) die_f("cannot open %s", path.c_str());
    out->size = static_cast<size_t>(::lseek(out->fd, 0, SEEK_END));
    out->data = static_cast<char*>(
        ::mmap(nullptr, out->size, PROT_READ, MAP_PRIVATE, out->fd, 0));
    if (out->data == MAP_FAILED) die("mmap failed");

    uint64_t header_len = 0;
    std::memcpy(&header_len, out->data, sizeof(header_len));
    size_t data_offset = sizeof(uint64_t) + static_cast<size_t>(header_len);
    std::string hdr_str(out->data + sizeof(uint64_t), static_cast<size_t>(header_len));
    auto root = json::parse(hdr_str);

    if (!root.is_object()) die("safetensors header is not a JSON object");
    for (const auto& [key, value] : root.as_object()) {
        if (key == "__metadata__") continue;
        if (!value.is_object()) continue;
        const auto& obj = value.as_object();
        auto dt_it = obj.find("dtype");
        auto sh_it = obj.find("shape");
        auto off_it = obj.find("data_offsets");
        if (dt_it == obj.end() || sh_it == obj.end() || off_it == obj.end()) continue;
        if (!dt_it->second.is_string() || !sh_it->second.is_array() || !off_it->second.is_array()) continue;

        SSTensor t;
        t.name = key;
        t.dtype = dt_it->second.as_string();
        for (const auto& dim : sh_it->second.as_array()) {
            if (!dim.is_number()) die_f("non-numeric shape value in tensor %s", key.c_str());
            t.shape.push_back(dim.as_i64());
        }
        const auto& offs = off_it->second.as_array();
        if (offs.size() != 2 || !offs[0].is_number() || !offs[1].is_number()) {
            die_f("invalid data_offsets for tensor %s", key.c_str());
        }
        t.offset = data_offset + static_cast<size_t>(offs[0].as_i64());
        t.nbytes = static_cast<size_t>(offs[1].as_i64() - offs[0].as_i64());

        // Sanity check
        int64_t n_el = 1;
        for (auto d : t.shape) n_el *= d;
        int el_size = (t.dtype == "F32") ? 4 : 2;  // F16 or BF16
        if (t.dtype != "F32" && t.dtype != "F16" && t.dtype != "BF16")
            die_f("unsupported dtype %s in tensor %s", t.dtype.c_str(), t.name.c_str());
        if (static_cast<int64_t>(t.nbytes) != n_el * el_size)
            die_f("size mismatch for %s: %zu bytes vs %ld elements * %d",
                  t.name.c_str(), t.nbytes, n_el, el_size);

        out->tensors.push_back(std::move(t));
    }
    return out;
}

static ggml_type dtype_to_ggml(const std::string& dt) {
    if (dt == "F32")  return GGML_TYPE_F32;
    if (dt == "F16")  return GGML_TYPE_F16;
    if (dt == "BF16") return GGML_TYPE_BF16;
    return GGML_TYPE_COUNT;
}

// Add a tensor to the GGUF, preserving shape and dtype from the safetensors.
//
// BF16 source tensors are converted on the fly:
//   - rank-1 (biases) → F32 (CPU backend needs them as F32 for mixed F32+F16 binary ops)
//   - rank-2/3 (weights) → F16 (CPU conv_1d requires F16 src0)
// The scratch buffer is owned by `convert_bufs` so the pointer stays valid
// through gguf_write_to_file.
static void add_tensor(gguf_context* gctx, ggml_context* mctx,
                       const char* gguf_name, const SafeTensorFile& stf,
                       const SSTensor& st,
                       std::vector<std::vector<uint16_t>>* convert_bufs) {
    int n_dim = static_cast<int>(st.shape.size());
    if (n_dim < 1 || n_dim > 3)
        die_f("tensor %s has unsupported rank %d", gguf_name, n_dim);

    // ggml ne[]: ne[0] is the innermost (fastest-varying) dimension.
    // PyTorch conv1d weight [O, I, K] maps to ggml ne = [K, I, O].
    int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
    for (int i = 0; i < n_dim; i++) {
        ne[i] = st.shape[n_dim - 1 - i];
    }
    int64_t n_el = 1;
    for (int i = 0; i < n_dim; i++) n_el *= ne[i];

    const void* data_ptr = const_cast<char*>(stf.data) + st.offset;
    ggml_type gt;
    if (st.dtype == "BF16") {
        if (n_dim == 1) {
            // Rank-1 tensors are biases — added to F32 conv outputs, so the
            // CPU backend needs them as F32. Expand BF16 → F32 losslessly.
            std::vector<uint32_t> f32_buf(static_cast<size_t>(n_el));
            const uint16_t* s16 = static_cast<const uint16_t*>(data_ptr);
            for (int64_t i = 0; i < n_el; i++) {
                f32_buf[size_t(i)] = uint32_t(s16[size_t(i)]) << 16;
            }
            // Stash as uint16 vector of doubled size to keep the buffer alive.
            std::vector<uint16_t> stash(static_cast<size_t>(n_el) * 2);
            std::memcpy(stash.data(), f32_buf.data(), stash.size() * 2);
            convert_bufs->push_back(std::move(stash));
            data_ptr = convert_bufs->back().data();
            gt = GGML_TYPE_F32;
        } else {
            // Rank-2/3 tensors are conv/mul-mat weights — convert BF16 → F16 (RTE).
            convert_bufs->emplace_back();
            bf16_buf_to_f16(data_ptr, static_cast<size_t>(n_el), &convert_bufs->back());
            data_ptr = convert_bufs->back().data();
            gt = GGML_TYPE_F16;
        }
    } else {
        gt = dtype_to_ggml(st.dtype);
    }

    ggml_tensor* t = ggml_new_tensor(mctx, gt, n_dim, ne);
    if (!t) die_f("ggml_new_tensor failed for %s", gguf_name);
    t->data = const_cast<void*>(data_ptr);
    ggml_set_name(t, gguf_name);
    gguf_add_tensor(gctx, t);
}

// ── Main ────────────────────────────────────────────────────────────────────

static int usage(const char* argv0) {
    std::fprintf(stderr,
        "usage: %s <model_dir> <output.gguf> [--filter-prefix PREFIX]\n"
        "                             [--output-prefix PREFIX] [--strip-prefix]\n"
        "  Convert ECAPA-TDNN speaker-encoder HuggingFace safetensors to a\n"
        "  standalone audio.cpp GGUF. The runtime resolves tensors via the\n"
        "  `speaker_encoder.*` prefix in src/models/qwen3_tts/speaker_encoder.cpp.\n"
        "  By default this converter PREPENDS `speaker_encoder.` to every\n"
        "  source tensor name (after applying --filter-prefix if given).\n"
        "  --filter-prefix PREFIX: only emit tensors whose name starts with\n"
        "      PREFIX; strip PREFIX before applying the output prefix.\n"
        "      Example: extract from a bundled talker checkpoint:\n"
        "        convert_ecapa Qwen3-TTS-1.7B-Base out.gguf \\\n"
        "            --filter-prefix speaker_encoder.\n"
        "  --output-prefix PREFIX: prepend PREFIX instead of the default\n"
        "      `speaker_encoder.`. Example: produce a `speaker.*`-prefixed\n"
        "      GGUF for runtimes that expect that namespace.\n"
        "  --strip-prefix: emit source tensor names VERBATIM (no prefix added).\n"
        "      Mutually exclusive with --output-prefix.\n",
        argv0);
    return 2;
}

int main(int argc, char** argv) {
    bool strip_prefix = false;
    bool has_output_prefix = false;
    std::string output_prefix = "speaker_encoder.";
    std::string filter_prefix;
    std::vector<std::string> positional;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--strip-prefix") strip_prefix = true;
        else if (a == "--output-prefix") {
            if (++i >= argc) die("--output-prefix needs an argument");
            output_prefix = argv[i];
            has_output_prefix = true;
        }
        else if (a == "--filter-prefix") {
            if (++i >= argc) die("--filter-prefix needs an argument");
            filter_prefix = argv[i];
        }
        else if (a == "-h" || a == "--help") return usage(argv[0]);
        else positional.push_back(a);
    }
    if (positional.size() < 2) return usage(argv[0]);
    if (strip_prefix && has_output_prefix)
        die("--strip-prefix and --output-prefix are mutually exclusive");

    fs::path model_dir = positional[0];
    fs::path output    = positional[1];

    auto config = json::parse_file(model_dir / "config.json");

    // Architecture constants. The standalone marksverdhei checkpoint has
    // these at the top level; the bundled Qwen3-TTS-Base config nests them
    // under `speaker_encoder_config` and only carries enc_dim + sample_rate
    // (the ECAPA-TDNN architecture is fixed across Qwen3-TTS variants).
    // Fall back to defaults that match the 1.7B reference for any field the
    // source config doesn't surface.
    const json::Value* spk_cfg = config.find("speaker_encoder_config");

    uint32_t enc_dim            = get_u32_chained(config, spk_cfg, "enc_dim", 2048);
    uint32_t mel_dim            = get_u32_chained(config, spk_cfg, "mel_dim", 128);
    uint32_t sample_rate        = get_u32_chained(config, spk_cfg, "sample_rate", 24000);
    uint32_t res2net_scale      = get_u32_chained(config, spk_cfg, "enc_res2net_scale", 8);
    uint32_t se_channels        = get_u32_chained(config, spk_cfg, "enc_se_channels", 128);
    uint32_t attention_channels = get_u32_chained(config, spk_cfg, "enc_attention_channels", 128);
    auto enc_channels  = get_u32_array_chained(config, spk_cfg, "enc_channels",
                                                {512, 512, 512, 512, 1536});
    auto enc_kernels   = get_u32_array_chained(config, spk_cfg, "enc_kernel_sizes",
                                                {5, 3, 3, 3, 1});
    auto enc_dilations = get_u32_array_chained(config, spk_cfg, "enc_dilations",
                                                {1, 2, 3, 4, 1});

    std::printf("ECAPA-TDNN config: enc_dim=%u mel_dim=%u sr=%u res2net=%u se=%u attn=%u\n",
                enc_dim, mel_dim, sample_rate, res2net_scale, se_channels, attention_channels);
    if (enc_channels.size() == 5) {
        std::printf("  enc_channels  : [%u, %u, %u, %u, %u]\n",
                    enc_channels[0], enc_channels[1], enc_channels[2],
                    enc_channels[3], enc_channels[4]);
        std::printf("  enc_kernels   : [%u, %u, %u, %u, %u]\n",
                    enc_kernels[0], enc_kernels[1], enc_kernels[2],
                    enc_kernels[3], enc_kernels[4]);
        std::printf("  enc_dilations : [%u, %u, %u, %u, %u]\n",
                    enc_dilations[0], enc_dilations[1], enc_dilations[2],
                    enc_dilations[3], enc_dilations[4]);
    }

    // Find safetensors file (single-file convention for this model)
    fs::path st_path;
    for (auto& e : fs::directory_iterator(model_dir)) {
        if (e.path().extension() == ".safetensors") {
            st_path = e.path();
            break;
        }
    }
    if (st_path.empty()) die("no .safetensors file in model_dir");
    std::printf("Loading %s ...\n", st_path.c_str());
    auto stf = load_safetensors(st_path);
    std::printf("  %zu tensors loaded\n", stf->tensors.size());

    // Set up GGUF writer
    gguf_context* gctx = gguf_init_empty();
    if (!gctx) die("gguf_init_empty failed");

    gguf_set_val_str(gctx, "general.architecture", "qwen3tts_spk");
    gguf_set_val_u32(gctx, "general.alignment", 32);
    gguf_set_val_u32(gctx, "qwen3tts_spk.enc_dim", enc_dim);
    gguf_set_val_u32(gctx, "qwen3tts_spk.mel_dim", mel_dim);
    gguf_set_val_u32(gctx, "qwen3tts_spk.sample_rate", sample_rate);
    gguf_set_val_u32(gctx, "qwen3tts_spk.res2net_scale", res2net_scale);
    gguf_set_val_u32(gctx, "qwen3tts_spk.se_channels", se_channels);
    gguf_set_val_u32(gctx, "qwen3tts_spk.attention_channels", attention_channels);
    if (enc_channels.size() == 5) {
        gguf_set_arr_data(gctx, "qwen3tts_spk.enc_channels",
                          GGUF_TYPE_UINT32, enc_channels.data(), 5);
        gguf_set_arr_data(gctx, "qwen3tts_spk.enc_kernel_sizes",
                          GGUF_TYPE_UINT32, enc_kernels.data(), 5);
        gguf_set_arr_data(gctx, "qwen3tts_spk.enc_dilations",
                          GGUF_TYPE_UINT32, enc_dilations.data(), 5);
    }

    // ggml context for tensor metadata — model is tiny, overhead is generous.
    size_t n_tensors = stf->tensors.size();
    size_t mem = ggml_tensor_overhead() * (n_tensors + 16);
    struct ggml_init_params gip = {mem, nullptr, true};
    ggml_context* mctx = ggml_init(gip);
    if (!mctx) die("ggml_init failed");

    // Write every tensor with the chosen prefix.
    // BF16 source tensors are converted to F16/F32 in scratch buffers that
    // live in `convert_bufs` until gguf_write_to_file completes.
    std::vector<std::vector<uint16_t>> convert_bufs;
    convert_bufs.reserve(n_tensors);

    std::vector<const SSTensor*> sorted;
    sorted.reserve(n_tensors);
    for (auto& t : stf->tensors) sorted.push_back(&t);
    std::sort(sorted.begin(), sorted.end(),
              [](const SSTensor* a, const SSTensor* b) { return a->name < b->name; });

    size_t added = 0, filtered = 0;
    for (auto* t : sorted) {
        if (t->name == "__metadata__") continue;
        std::string base_name = t->name;
        if (!filter_prefix.empty()) {
            if (base_name.size() < filter_prefix.size() ||
                base_name.compare(0, filter_prefix.size(), filter_prefix) != 0) {
                ++filtered;
                continue;
            }
            base_name = base_name.substr(filter_prefix.size());
        }
        // Default prefix matches the bundled-talker layout (speaker_encoder.*),
        // which is what Qwen3SpeakerEncoderRuntime::load_weights() expects.
        // --strip-prefix emits source names verbatim for standalone checkpoints
        // that already use the speaker_encoder.* namespace.
        // --output-prefix overrides the namespace for runtimes that expect a
        // different one (e.g. audiocore uses `speaker.*`).
        std::string out_name = strip_prefix ? base_name : output_prefix + base_name;
        add_tensor(gctx, mctx, out_name.c_str(), *stf, *t, &convert_bufs);
        ++added;
        if (added <= 5 || added == n_tensors) {
            std::printf("  [%4zu] %-60s  shape=[", added, out_name.c_str());
            for (size_t i = 0; i < t->shape.size(); i++)
                std::printf("%s%ld", (i ? "," : ""), t->shape[i]);
            std::printf("] dtype=%s%s\n", t->dtype.c_str(),
                        (t->dtype == "BF16") ? "→F16/F32" : "");
        } else if (added == 6) {
            std::printf("  ...\n");
        }
    }

    std::printf("Writing %s (%zu tensors, %zu filtered) ...\n",
                output.c_str(), added, filtered);
    if (!gguf_write_to_file(gctx, output.c_str(), /*only_meta=*/false))
        die("gguf_write_to_file failed");
    std::printf("done (%zu MB)\n", fs::file_size(output) / 1048576);

    gguf_free(gctx);
    ggml_free(mctx);
    return 0;
}
