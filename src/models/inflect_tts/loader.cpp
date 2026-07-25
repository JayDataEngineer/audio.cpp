#include "engine/models/inflect_tts/loader.h"

#include "engine/framework/model_spec/package.h"
#include "engine/models/inflect_tts/session.h"

#include <memory>
#include <stdexcept>
#include <utility>

namespace engine::models::inflect_tts {
namespace {

runtime::CapabilitySet capabilities(const InflectAssets &) {
    runtime::CapabilitySet out;
    out.supported_tasks = {
        {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
    };
    out.languages = {"en"};
    return out;
}

runtime::ModelMetadata metadata(const InflectAssets & assets) {
    runtime::ModelMetadata out;
    out.family = "inflect_tts";
    out.variant = assets.resources.model_root().filename().string();
    out.description = "Inflect Micro v2 TTS loaded from local assets.";
    return out;
}

runtime::ModelCliInterface cli(const InflectAssets &) {
    runtime::ModelCliInterface out;
    out.request_options = {
        {"speaking_rate", "float", "Speech speed multiplier; default 1.0."},
        {"variation", "float", "Latent noise scale; default 0.667."},
        {"text_chunk_size", "n", "Long-form text chunk size; default 280."},
        {"text_chunk_mode", "default|tag_aware|japanese|endline", "Framework text chunking mode."},
        {"seed", "n", "Sampling seed; omitted means random."},
    };
    out.session_options = {
        {"inflect_tts.weight_type", "native|f32|f16|bf16|q8_0", "Inflect weight storage type."},
    };
    return out;
}

class InflectLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "inflect_tts";
    }

    runtime::CapabilitySet advertised_capabilities() const override {
        runtime::CapabilitySet out;
        out.supported_tasks = {
            {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
        };
        out.languages = {"en"};
        return out;
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        if (request.family_hint.has_value() && *request.family_hint != family()) {
            return false;
        }
        try {
            (void)engine::model_spec::load_resource_bundle(
                request.model_path,
                engine::model_spec::default_spec_path(family()));
            return true;
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto assets = load_inflect_assets(request.model_path);
        runtime::ModelInspection inspection;
        inspection.metadata = metadata(*assets);
        inspection.capabilities = capabilities(*assets);
        inspection.cli = cli(*assets);
        inspection.model_root = assets->resources.model_root();
        const auto spec_path = engine::model_spec::default_spec_path(family());
        inspection.discovered_configs = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            spec_path,
            engine::model_spec::ResourceKind::Files);
        inspection.discovered_weights = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            spec_path,
            engine::model_spec::ResourceKind::Tensors);
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return load_inflect_model(request);
    }
};

}  // namespace

InflectLoadedModel::InflectLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const InflectAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {
    if (assets_ == nullptr) {
        throw std::runtime_error("Inflect loaded model requires assets");
    }
}

const runtime::ModelMetadata & InflectLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & InflectLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> InflectLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (task.task != runtime::VoiceTaskKind::Tts) {
        throw std::runtime_error("Inflect TTS supports --task tts");
    }
    if (task.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Inflect TTS only supports offline mode");
    }
    return std::make_unique<InflectTTSSession>(task, options, assets_);
}

std::unique_ptr<InflectLoadedModel> load_inflect_model(const runtime::ModelLoadRequest & request) {
    auto assets = load_inflect_assets(request.model_path);
    return std::make_unique<InflectLoadedModel>(
        metadata(*assets),
        capabilities(*assets),
        std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_inflect_tts_loader() {
    return std::make_shared<InflectLoader>();
}

}  // namespace engine::models::inflect_tts
