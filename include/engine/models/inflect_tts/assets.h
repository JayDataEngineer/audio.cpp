#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/text/english_g2p.h"

#include <cstdint>
#include <filesystem>
#include <memory>

namespace engine::models::inflect_tts {

struct InflectConfig {
    int sample_rate = 24000;
    int64_t vocab_size = 178;
    int64_t inter_channels = 192;
    int64_t hidden_channels = 96;
    int64_t filter_channels = 768;
    int64_t n_heads = 2;
    int64_t n_layers = 3;
    int64_t kernel_size = 3;
    int64_t duration_channels = 256;
    int64_t max_len = 4000;
};

struct InflectAssets {
    assets::ResourceBundle resources;
    InflectConfig config;
    std::shared_ptr<const engine::text::english_g2p::EnglishG2P> english_g2p;
    std::shared_ptr<const assets::TensorSource> weights;
};

std::shared_ptr<const InflectAssets> load_inflect_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::inflect_tts
