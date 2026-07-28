#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/models/moss/moss_sfx_v2/dit_runner.h"
#include "engine/models/moss/moss_sfx_v2/text_encoder.h"
#include "engine/models/moss/moss_sfx_v2/vae_runner.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace engine::models::moss_sfx_v2 {

// Scheduler parameters parsed from the DiT GGUF KV metadata. Ported from
// audiocore's SfxConfig scheduler fields; matches FlowMatchScheduler in
// moss_soundeffect_v2/diffsynth/schedulers/flow_match.py.
struct SfxSchedulerConfig {
    int32_t shift                 = 5;
    float   sigma_min             = 0.0f;
    bool    extra_one_step        = true;
    int32_t num_train_timesteps   = 1000;
};

// Top-level config. DitConfig + VAEConfig live in their runner headers.
struct SfxConfig {
    DitConfig             dit;
    VAEConfig             vae;
    TextEncoderConfig     te;        // parsed from the TE GGUF KV (qwen3.*)
    int32_t               sample_rate = 48000;
    SfxSchedulerConfig    scheduler;
};

struct MossSfxV2Assets {
    assets::ResourceBundle resources;
    SfxConfig              config;
    // Tensor sources opened from the resource bundle. The DiT and VAE live in
    // separate GGUFs; the text encoder sidecar is optional (nullptr when absent).
    std::shared_ptr<const assets::TensorSource> dit_weights;
    std::shared_ptr<const assets::TensorSource> vae_weights;
    std::shared_ptr<const assets::TensorSource> te_weights;  // may be nullptr
    // Directory holding the HF tokenizer files (vocab.json, merges.txt,
    // tokenizer_config.json) for the Qwen3 text encoder. Empty when not found;
    // the session then falls back to a zero context.
    std::filesystem::path  tokenizer_dir;
};

// Parses config.json (architecture overrides) + GGUF KV metadata into SfxConfig.
SfxConfig parse_config(const assets::ResourceBundle & resources);

// Opens the resource bundle via model_spec::load_resource_bundle and the
// moss_sfx_v2 spec, parses config, and opens the DiT / VAE / (optional) TE
// tensor sources. Throws std::runtime_error on hard failure.
std::shared_ptr<const MossSfxV2Assets> load_moss_sfx_v2_assets(
    const std::filesystem::path & model_path);

}  // namespace engine::models::moss_sfx_v2
