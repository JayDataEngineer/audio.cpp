// moss_sfx_v2_te_parity.cpp — standalone Qwen3-TE parity probe.
//
// Verifies the ported TextEncoderRunner against the real upstream Python
// (WanPrompter.encode_prompt) reference artifacts captured in
// dump_encode_prompt.py: the token ids and the last-layer hidden states
// (cond_emb). This bypasses the full session/registry path (which needs the
// DiT + VAE GGUFs) and exercises ONLY the TE + tokenizer, so it runs as soon
// as the TE GGUF + HF tokenizer files are present.
//
// Gate (exit 0 on PASS):
//   - C++ tokenizer ids[:n_real] == reference ids[:n_real]
//   - overall cosine(cond_emb_cpp, cond_emb_ref) >= --cos-bar
//   - mean abs err over real positions <= --mae-bar
//
// Run:
//   moss_sfx_v2_te_parity
//     --te /path/mse2-te.gguf
//     --tokenizer-dir /path/MOSS-SoundEffect-v2.0/tokenizer
//     --ref-dir /path/encprompt_artifacts

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/tokenizers/llama_bpe.h"
#include "engine/models/moss/moss_sfx_v2/text_encoder.h"
#include "engine/models/moss/moss_sfx_v2/weight_bridge.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string arg_value(int argc, char** argv, const std::string& name,
                      const std::string& fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) return argv[i + 1];
    }
    return fallback;
}

double arg_double(int argc, char** argv, const std::string& name,
                  double fallback) {
    const std::string v = arg_value(argc, argv, name, "");
    return v.empty() ? fallback : std::stod(v);
}

// Reference artifacts are raw little-endian dumps (the .npy extension is
// misleading — no numpy header): cond_ids is 512 int32, cond_emb.f32 is
// 512*hidden float32. Read them verbatim.
std::vector<int32_t> read_i32LE(const std::filesystem::path& p, size_t expect) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + p.string());
    std::vector<int32_t> out(expect);
    f.read(reinterpret_cast<char*>(out.data()),
           static_cast<std::streamsize>(expect * sizeof(int32_t)));
    if (!f) throw std::runtime_error("short read on " + p.string());
    return out;
}

std::vector<float> read_f32LE(const std::filesystem::path& p, size_t expect) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + p.string());
    std::vector<float> out(expect);
    f.read(reinterpret_cast<char*>(out.data()),
           static_cast<std::streamsize>(expect * sizeof(float)));
    if (!f) throw std::runtime_error("short read on " + p.string());
    return out;
}

double dot(const float* a, const float* b, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) s += static_cast<double>(a[i]) * b[i];
    return s;
}
double norm2(const float* a, size_t n) { return std::sqrt(dot(a, a, n)); }

}  // namespace

int main(int argc, char** argv) try {
    const std::filesystem::path te_gguf =
        arg_value(argc, argv, "--te", "models/MOSS-SFX-v2-GGUF/mse2-te.gguf");
    const std::filesystem::path tokenizer_dir =
        arg_value(argc, argv, "--tokenizer-dir", "models/MOSS-SFX-v2-GGUF/tokenizer");
    const std::filesystem::path ref_dir =
        arg_value(argc, argv, "--ref-dir", "encprompt_artifacts");
    const std::string prompt = arg_value(argc, argv, "--prompt",
        "heavy thunderstorm with rain hitting a tin roof duration: 10.0s");
    const double cos_bar = arg_double(argc, argv, "--cos-bar", 0.985);
    const double mae_bar = arg_double(argc, argv, "--mae-bar", 0.10);

    const int hidden = 2048;
    const int max_seq = 512;
    const size_t emb_count = static_cast<size_t>(hidden) * max_seq;

    // ── Reference artifacts ───────────────────────────────────────────────
    const auto ref_ids = read_i32LE(ref_dir / "cond_ids.npy", max_seq);
    const auto ref_emb = read_f32LE(ref_dir / "cond_emb.f32", emb_count);
    int n_real = 0;
    {
        std::ifstream f(ref_dir / "cond_seq_lens.txt");
        if (f && (f >> n_real) && n_real > 0) {
            // use the upstream seq_lens (canonical real-token count)
        } else {
            n_real = 0;
            for (int i = 0; i < max_seq; ++i)
                if (ref_ids[i] != 151643 /*pad*/ && ref_ids[i] != 0) ++n_real;
        }
    }
    std::fprintf(stderr, "[parity] ref n_real=%d\n", n_real);
    std::fprintf(stderr, "[parity] ref ids (first %d):", n_real);
    for (int i = 0; i < n_real; ++i) std::fprintf(stderr, " %d", ref_ids[i]);
    std::fprintf(stderr, "\n");

    // ── Load TE weights into a no_alloc ggml context ──────────────────────
    auto source = engine::assets::open_tensor_source(te_gguf);
    if (!source) throw std::runtime_error("open_tensor_source(TE) failed");
    std::string err;
    auto loaded = engine::models::moss_sfx_v2::load_weights_into_context(
        *source, std::string(""), /*capacity_tensors=*/1024, &err);
    if (!loaded) throw std::runtime_error("load_weights_into_context: " + err);
    std::fprintf(stderr, "[parity] TE tensors loaded: %zu\n",
                 loaded->tensors_loaded);

    // ── Build the Qwen3 BPE tokenizer from HF files ───────────────────────
    engine::tokenizers::LlamaBpeTokenizerSpec spec;
    spec.vocab_path            = tokenizer_dir / "vocab.json";
    spec.merges_path           = tokenizer_dir / "merges.txt";
    spec.tokenizer_config_path = tokenizer_dir / "tokenizer_config.json";
    const auto tok_json = tokenizer_dir / "tokenizer.json";
    if (std::filesystem::exists(tok_json)) spec.tokenizer_json_path = tok_json;
    spec.pre_type = engine::tokenizers::LlamaBpePreTokenizer::Qwen2;
    auto tokenizer = engine::tokenizers::load_llama_bpe_tokenizer(spec);

    // ── Tokenizer parity (must match upstream ids before the TE can) ──────
    auto cpp_ids = tokenizer->encode(prompt);
    std::fprintf(stderr, "[parity] cpp ids (%zu):", cpp_ids.size());
    for (auto id : cpp_ids) std::fprintf(stderr, " %d", id);
    std::fprintf(stderr, "\n");

    bool ids_match = static_cast<int>(cpp_ids.size()) == n_real;
    for (int i = 0; ids_match && i < n_real; ++i) {
        if (cpp_ids[i] != ref_ids[i]) ids_match = false;
    }
    std::fprintf(stderr, "[parity] token ids %s\n",
                 ids_match ? "MATCH ✓" : "MISMATCH ✗");
    if (!ids_match) {
        std::fprintf(stderr, "[parity] ids differ — TE output comparison is "
                             "meaningless until the tokenizer matches.\n");
    }

    // ── Run the TE forward ────────────────────────────────────────────────
    engine::models::moss_sfx_v2::TextEncoderConfig cfg;  // Qwen3-1.7B defaults
    engine::models::moss_sfx_v2::TextEncoderRunner runner(loaded->ctx, cfg,
                                                           tokenizer);
    std::vector<float> cpp_emb(emb_count, 0.0f);
    const int got = runner.encode(prompt, cpp_emb.data(), &err);
    if (got < 0) throw std::runtime_error("TE encode failed: " + err);
    std::fprintf(stderr, "[parity] TE returned n_real=%d\n", got);

    // ── Embedding parity over the real positions ─────────────────────────
    const int N = std::max(1, n_real);
    double per_pos_cos_sum = 0.0, per_pos_cos_min = 1e9, per_pos_cos_max = -1e9;
    double abs_err_sum = 0.0, abs_err_max = 0.0;
    for (int t = 0; t < N; ++t) {
        const float* a = &cpp_emb[static_cast<size_t>(t) * hidden];
        const float* b = &ref_emb[static_cast<size_t>(t) * hidden];
        const double na = norm2(a, hidden);
        const double nb = norm2(b, hidden);
        double c = (na > 0.0 && nb > 0.0) ? dot(a, b, hidden) / (na * nb) : 0.0;
        per_pos_cos_sum += c;
        per_pos_cos_min = std::min(per_pos_cos_min, c);
        per_pos_cos_max = std::max(per_pos_cos_max, c);
        for (int i = 0; i < hidden; ++i) {
            const double e = std::fabs(static_cast<double>(a[i]) - b[i]);
            abs_err_sum += e;
            abs_err_max = std::max(abs_err_max, e);
        }
    }
    const double mae = abs_err_sum / static_cast<double>(N * hidden);
    // Overall cosine over the flattened real block too.
    const double overall_cos = [&] {
        const double na = norm2(cpp_emb.data(), static_cast<size_t>(N) * hidden);
        const double nb = norm2(ref_emb.data(), static_cast<size_t>(N) * hidden);
        return (na > 0.0 && nb > 0.0)
            ? dot(cpp_emb.data(), ref_emb.data(),
                  static_cast<size_t>(N) * hidden) / (na * nb)
            : 0.0;
    }();

    std::fprintf(stderr, "\n=== TE embedding parity (over %d real positions) ===\n", N);
    std::fprintf(stderr, "per-position cosine: min=%.6f mean=%.6f max=%.6f\n",
                 per_pos_cos_min, per_pos_cos_sum / N, per_pos_cos_max);
    std::fprintf(stderr, "overall cosine     : %.6f\n", overall_cos);
    std::fprintf(stderr, "max abs err        : %.6f\n", abs_err_max);
    std::fprintf(stderr, "mean abs err       : %.6f (bar %.4f)\n", mae, mae_bar);

    const bool emb_pass = overall_cos >= cos_bar && mae <= mae_bar;
    const bool pass = ids_match && emb_pass;
    std::fprintf(stderr, "\n[parity] %s  (ids=%s emb=%s; cos-bar=%.4f mae-bar=%.4f)\n",
                 pass ? "VERDICT: PASS" : "VERDICT: FAIL",
                 ids_match ? "ok" : "BAD",
                 emb_pass ? "ok" : "BAD", cos_bar, mae_bar);
    return pass ? 0 : 2;
} catch (const std::exception& ex) {
    std::fprintf(stderr, "moss_sfx_v2_te_parity failed: %s\n", ex.what());
    return 1;
}
