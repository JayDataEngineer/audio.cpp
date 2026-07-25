#include "engine/models/inflect_tts/assets.h"

#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"

namespace engine::models::inflect_tts {
namespace {

namespace json = engine::io::json;

InflectConfig parse_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("config");
    InflectConfig config;
    const auto & data = root.require("data");
    config.sample_rate = static_cast<int>(json::require_i64(data, "sampling_rate"));
    config.max_len = json::optional_i64(data, "max_len", config.max_len);
    const auto & model = root.require("model");
    config.inter_channels = json::require_i64(model, "inter_channels");
    config.hidden_channels = json::require_i64(model, "hidden_channels");
    config.filter_channels = json::require_i64(model, "filter_channels");
    config.n_heads = json::require_i64(model, "n_heads");
    config.n_layers = json::require_i64(model, "n_layers");
    config.kernel_size = json::require_i64(model, "kernel_size");
    return config;
}

void validate_weight_anchors(const InflectAssets & assets) {
    assets::require_tensor_shape(*assets.weights, "enc_p.emb.weight", {assets.config.vocab_size, assets.config.hidden_channels});
    assets::require_tensor_shape(*assets.weights, "enc_p.proj.weight", {assets.config.inter_channels * 2, assets.config.hidden_channels, 1});
    assets::require_tensor_shape(*assets.weights, "dp.proj.weight", {1, assets.config.duration_channels, 1});
    assets::require_tensor_shape(*assets.weights, "flow.flows.0.pre.weight", {assets.config.hidden_channels, assets.config.hidden_channels, 1});
    assets::require_tensor_shape(*assets.weights, "dec.conv_pre.weight", {320, assets.config.inter_channels, 7});
    assets::require_tensor_shape(*assets.weights, "dec.conv_post.weight", {1, 20, 7});
}

}  // namespace

std::shared_ptr<const InflectAssets> load_inflect_assets(const std::filesystem::path & model_path) {
    InflectAssets assets;
    assets.resources = engine::model_spec::load_resource_bundle(
        model_path,
        engine::model_spec::default_spec_path("inflect_tts"));
    assets.config = parse_config(assets.resources);
    assets.weights = assets.resources.open_tensor_source("weights");
    validate_weight_anchors(assets);
    return std::make_shared<InflectAssets>(std::move(assets));
}

}  // namespace engine::models::inflect_tts
