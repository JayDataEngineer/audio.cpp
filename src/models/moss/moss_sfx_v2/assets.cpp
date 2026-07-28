// assets.cpp — MOSS-SoundEffect-v2 asset loading for audiocpp.
//
// The model ships as up to three GGUF sidecars in one directory:
//   • mse2-dit.gguf  — DiT weights + architecture KV metadata
//   • mse2-vae.gguf  — DAC VAE decoder weights + VAE KV metadata
//   • mse2-te.gguf   — optional Qwen3-Embedding text encoder
//
// audiocpp's TensorSource does not expose GGUF KV pairs publicly, so we read
// the architecture + scheduler scalars directly via gguf_init_from_file (the
// same C API audiocore's GgufReader uses), and we build the ResourceBundle by
// scanning the directory — mirroring audiocore's loader.cpp.

#include "engine/models/moss/moss_sfx_v2/assets.h"

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>

#include "ggml.h"
#include "gguf.h"

namespace engine::models::moss_sfx_v2 {

namespace fs = std::filesystem;

namespace {

// ── Tiny GGUF KV reader (gguf_init_from_file + gguf_get_val_*) ──────────────
// audiocore's GgufReader equivalent. Reads scalar KV pairs straight out of the
// file; only the keys we care about, no tensor data.
struct GgufKv {
    gguf_context* ctx = nullptr;
    explicit GgufKv(const std::string& path) {
        gguf_init_params params = {
            /*.no_alloc   =*/ true,
            /*.ctx        =*/ nullptr,
        };
        ctx = gguf_init_from_file(path.c_str(), params);
        if (!ctx) {
            throw std::runtime_error("failed to open GGUF for KV read: " + path);
        }
    }
    ~GgufKv() { if (ctx) gguf_free(ctx); }
    GgufKv(const GgufKv&) = delete;
    GgufKv& operator=(const GgufKv&) = delete;

    // All gguf_get_val_* take a key_id from gguf_find_key, and abort on a
    // type mismatch — so resolve the id once and only call the getter when
    // the key is actually present.
    int32_t i32(const char* key, int32_t def) const {
        const int64_t kid = gguf_find_key(ctx, key);
        if (kid < 0) return def;
        return static_cast<int32_t>(gguf_get_val_u32(ctx, kid));
    }
    float f32(const char* key, float def) const {
        const int64_t kid = gguf_find_key(ctx, key);
        if (kid < 0) return def;
        return gguf_get_val_f32(ctx, kid);
    }
    bool boolean(const char* key, bool def) const {
        const int64_t kid = gguf_find_key(ctx, key);
        if (kid < 0) return def;
        return gguf_get_val_bool(ctx, kid) != 0;
    }
};

// Lowercase substring test for filename classification.
bool name_contains_lower(const fs::path& p, const char* needle) {
    std::string s = p.filename().string();
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s.find(needle) != std::string::npos;
}

struct DiscoveredPaths {
    fs::path dit;
    fs::path vae;
    fs::path te;
    fs::path tokenizer_dir;   // dir holding vocab.json + merges.txt + tokenizer_config.json
    bool found = false;
};

DiscoveredPaths discover_ggufs(const fs::path& model_root) {
    DiscoveredPaths out;
    if (!fs::is_directory(model_root)) {
        // Single file — treat as the DiT GGUF.
        out.dit = model_root;
        out.found = true;
        const fs::path parent = model_root.parent_path();
        if (!parent.empty() && fs::exists(parent / "vocab.json")) {
            out.tokenizer_dir = parent;
        }
        return out;
    }
    for (auto& e : fs::directory_iterator(model_root)) {
        const auto p = e.path();
        if (p.extension() != ".gguf") continue;
        // VAE must be checked before "te" (both contain 'e'); and "dit" is
        // distinguished from the others by the 'd' or 'dit' substring.
        if (name_contains_lower(p, "vae")) {
            out.vae = p;
        } else if (name_contains_lower(p, "te") ||
                   name_contains_lower(p, "tokenizer") ||
                   name_contains_lower(p, "encoder")) {
            out.te = p;
        } else {
            // Default bucket — the DiT is the primary model file.
            if (out.dit.empty()) out.dit = p;
        }
    }
    out.found = !out.dit.empty();

    // Discover the HF tokenizer files for the Qwen3 text encoder. Accept either
    // a `tokenizer/` subdirectory or the files sitting next to the GGUFs.
    auto has_tokenizer = [](const fs::path& dir) {
        return !dir.empty() && fs::exists(dir / "vocab.json") &&
               fs::exists(dir / "merges.txt") &&
               fs::exists(dir / "tokenizer_config.json");
    };
    const fs::path tok_subdir = model_root / "tokenizer";
    if (has_tokenizer(tok_subdir)) {
        out.tokenizer_dir = tok_subdir;
    } else if (has_tokenizer(model_root)) {
        out.tokenizer_dir = model_root;
    }

    return out;
}

}  // namespace

SfxConfig parse_config(const assets::ResourceBundle& resources) {
    SfxConfig cfg;
    // Config sidecar is optional — the GGUF KV metadata is the source of truth.
    // (Kept for future overrides via a config.json "files" entry.)
    (void) resources;
    return cfg;
}

std::shared_ptr<const MossSfxV2Assets> load_moss_sfx_v2_assets(
    const fs::path& model_path) {

    const auto paths = discover_ggufs(model_path);
    if (!paths.found) {
        throw std::runtime_error(
            "MOSS-SFX-v2: no DiT GGUF found in " + model_path.string());
    }

    MossSfxV2Assets assets;
    assets.resources = assets::ResourceBundle(model_path);

    // ── Parse DiT architecture + scheduler from GGUF KV ──────────────────
    SfxConfig cfg;
    {
        GgufKv kv(paths.dit.string());
        cfg.dit.n_layers = kv.i32("moss_sfx_v2.n_layers", 30);
        cfg.dit.dim      = kv.i32("moss_sfx_v2.dim", 1536);
        cfg.dit.ffn_dim  = kv.i32("moss_sfx_v2.ffn_dim", 8960);
        cfg.dit.n_heads  = kv.i32("moss_sfx_v2.n_heads", 12);
        cfg.dit.in_dim   = kv.i32("moss_sfx_v2.in_dim", 128);
        cfg.dit.out_dim  = kv.i32("moss_sfx_v2.out_dim", 128);
        cfg.dit.text_dim = kv.i32("moss_sfx_v2.text_dim", 2048);
        cfg.dit.freq_dim = kv.i32("moss_sfx_v2.freq_dim", 256);
        cfg.dit.eps      = kv.f32("moss_sfx_v2.eps", 1e-6f);
        cfg.dit.patch_size = kv.i32("moss_sfx_v2.patch_size", 1);
        cfg.dit.head_dim = cfg.dit.dim / std::max(cfg.dit.n_heads, 1);

        cfg.scheduler.shift =
            kv.i32("moss_sfx_v2.scheduler.shift", 5);
        cfg.scheduler.sigma_min =
            kv.f32("moss_sfx_v2.scheduler.sigma_min", 0.0f);
        cfg.scheduler.extra_one_step =
            kv.boolean("moss_sfx_v2.scheduler.extra_one_step", true);
        cfg.scheduler.num_train_timesteps =
            kv.i32("moss_sfx_v2.scheduler.num_train_timesteps", 1000);
    }

    // ── Parse VAE config from the VAE GGUF KV (or DiT in_dim fallback) ────
    cfg.sample_rate = 48000;
    cfg.vae.latent_dim   = cfg.dit.in_dim;
    cfg.vae.decoder_dim  = 2048;
    cfg.vae.hop_length   = 960;
    cfg.vae.sample_rate  = 48000;
    cfg.vae.continuous    = true;
    if (!paths.vae.empty()) {
        GgufKv kv(paths.vae.string());
        const int32_t ld = kv.i32("moss_sfx_v2.vae.latent_dim", -1);
        if (ld > 0) cfg.vae.latent_dim = ld;
        cfg.vae.decoder_dim = kv.i32("moss_sfx_v2.vae.decoder_dim",
                                      cfg.vae.decoder_dim);
        cfg.vae.hop_length  = kv.i32("moss_sfx_v2.vae.hop_length",
                                      cfg.vae.hop_length);
        cfg.vae.sample_rate = kv.i32("moss_sfx_v2.vae.sample_rate",
                                      cfg.vae.sample_rate);
        cfg.vae.continuous  = kv.boolean("moss_sfx_v2.vae.continuous", true);
        cfg.sample_rate     = cfg.vae.sample_rate;
    }

    // ── Parse Qwen3 text-encoder config from the TE GGUF KV (qwen3.*) ─────
    // Defaults match Qwen3-1.7B; each field is overridden from the file so the
    // runner stays in lockstep with the actual converted weights.
    if (!paths.te.empty()) {
        GgufKv kv(paths.te.string());
        cfg.te.n_layers   = kv.i32("qwen3.block_count",             cfg.te.n_layers);
        cfg.te.hidden     = kv.i32("qwen3.embedding_length",         cfg.te.hidden);
        cfg.te.n_heads    = kv.i32("qwen3.attention.head_count",     cfg.te.n_heads);
        cfg.te.n_kv_heads = kv.i32("qwen3.attention.head_count_kv",  cfg.te.n_kv_heads);
        cfg.te.ffn_dim    = kv.i32("qwen3.feed_forward_length",      cfg.te.ffn_dim);
        cfg.te.rms_eps    = kv.f32("qwen3.attention.layer_norm_rms_epsilon", cfg.te.rms_eps);
        cfg.te.rope_base  = kv.f32("qwen3.rope.freq_base",           cfg.te.rope_base);
        const int32_t rope_dims = kv.i32("qwen3.rope.dimension_count", cfg.te.head_dim);
        cfg.te.head_dim   = rope_dims > 0 ? rope_dims : (cfg.te.hidden / std::max(cfg.te.n_heads, 1));
    }

    assets.config = cfg;

    // ── Register tensor sources on the bundle and open them ──────────────
    // We build the bundle by scanning the directory (multi-GGUF case), since
    // the model ships DiT + VAE (+ optional TE) as separate files.
    //
    // We deliberately pass NO tensor_prefix here. ResourceBundle's prefixed
    // view treats the prefix as a slash-separated namespace ("moss_sfx_v2/"),
    // but convert_mse2 stores tensors with a dot prefix ("moss_sfx_v2."). We
    // also need the FULL ggml tensor names preserved — the ported runners
    // call ggml_get_tensor(ctx, "moss_sfx_v2.layers.0...") verbatim, and
    // weight_bridge filters by prefix itself when populating the no_alloc
    // context. A prefixed view would both fail to match and strip the
    // namespace the runners rely on.
    assets.resources.add_tensor_source("dit_weights", paths.dit);
    if (!paths.vae.empty()) {
        assets.resources.add_tensor_source("vae_weights", paths.vae);
    }
    if (!paths.te.empty()) {
        assets.resources.add_tensor_source("te_weights", paths.te);
    }

    assets.dit_weights = assets.resources.open_tensor_source("dit_weights");
    if (!assets.dit_weights) {
        throw std::runtime_error("MOSS-SFX-v2: failed to open DiT tensor source");
    }
    if (!paths.vae.empty()) {
        assets.vae_weights = assets.resources.open_tensor_source("vae_weights");
        if (!assets.vae_weights) {
            throw std::runtime_error("MOSS-SFX-v2: failed to open VAE tensor source");
        }
    }
    if (!paths.te.empty()) {
        assets.te_weights = assets.resources.open_tensor_source("te_weights");
    }
    assets.tokenizer_dir = paths.tokenizer_dir;

    return std::make_shared<const MossSfxV2Assets>(std::move(assets));
}

}  // namespace engine::models::moss_sfx_v2
