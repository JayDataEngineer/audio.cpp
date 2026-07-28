#include "engine/models/moss/moss_sfx_v2/loader.h"

#include "engine/models/moss/moss_sfx_v2/assets.h"
#include "engine/models/moss/moss_sfx_v2/session.h"

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::moss_sfx_v2 {

namespace {

runtime::CapabilitySet capabilities() {
    runtime::CapabilitySet out;
    out.supported_tasks = {
        {runtime::VoiceTaskKind::AudioGeneration, {runtime::RunMode::Offline}},
    };
    out.languages = {"en"};
    return out;
}

runtime::ModelMetadata metadata() {
    runtime::ModelMetadata out;
    out.family = "moss_sfx_v2";
    out.variant = "MOSS-SoundEffect-v2";
    out.description =
        "MOSS Sound Effect v2 — 1.3B DiT (WanAudioModel, 30 layers) with a DAC "
        "VAE decoder and a flow-matching Euler schedule. Text-to-sound-effects.";
    return out;
}

runtime::ModelCliInterface cli() {
    runtime::ModelCliInterface out;
    out.request_options = {
        {"duration_sec", "float", "Target output duration in seconds; the pipeline denoises a 30s latent then crops. Default 10.", false, "10", "0", ""},
        {"num_inference_steps", "int", "Flow-matching denoising steps; default 50.", false, "50", "1", ""},
        {"guidance_scale", "float", "Classifier-free guidance scale; default 5.0.", false, "5.0", "0", ""},
        {"seed", "int", "Noise seed; 0/omitted uses default 42.", false, "42", "0", ""},
        {"negative_prompt", "string", "Unconditional/negative prompt for CFG; empty by default.", false, "", "", ""},
    };
    out.session_options = {
        {"weight_type", "enum", "DiT weight storage type (auto/native/f32/f16/bf16).", false, "auto", "", ""},
    };
    return out;
}

// Detect a moss_sfx_v2 model directory: it must contain at least one GGUF
// whose KV carries the moss_sfx_v2 architecture key. We avoid pulling gguf.h
// here by sniffing the filename first; can_load() must not throw.
bool looks_like_moss_sfx_v2(const std::filesystem::path& model_path) {
    namespace fs = std::filesystem;
    if (model_path.empty()) return false;
    if (!fs::exists(model_path)) return false;
    if (fs::is_directory(model_path)) {
        bool has_dit = false;
        for (auto& e : fs::directory_iterator(model_path)) {
            const auto p = e.path();
            if (p.extension() != ".gguf") continue;
            const std::string n = p.filename().string();
            if (n.find("dit") != std::string::npos ||
                n.find("sfx") != std::string::npos ||
                n.find("mse2") != std::string::npos) {
                has_dit = true;
                break;
            }
        }
        return has_dit;
    }
    // Single-file load — assume it's the DiT GGUF.
    return model_path.extension() == ".gguf";
}

class MossSfxV2LoadedModel final : public runtime::ILoadedVoiceModel {
public:
    MossSfxV2LoadedModel(
        runtime::ModelMetadata metadata,
        runtime::CapabilitySet capabilities,
        std::shared_ptr<const MossSfxV2Assets> assets)
        : metadata_(std::move(metadata)),
          capabilities_(std::move(capabilities)),
          assets_(std::move(assets)) {}

    const runtime::ModelMetadata& metadata() const noexcept override { return metadata_; }
    const runtime::CapabilitySet& capabilities() const noexcept override { return capabilities_; }

    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec& task,
        const runtime::SessionOptions& options) const override {
        if (task.mode != runtime::RunMode::Offline) {
            throw std::runtime_error("MOSS-SFX-v2 only supports offline sessions");
        }
        if (task.task != runtime::VoiceTaskKind::AudioGeneration) {
            throw std::runtime_error("MOSS-SFX-v2 only supports the AudioGeneration task");
        }
        return std::make_unique<MossSfxV2Session>(task, options, assets_);
    }

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const MossSfxV2Assets> assets_;
};

class MossSfxV2Loader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override { return "moss_sfx_v2"; }

    runtime::CapabilitySet advertised_capabilities() const override {
        return capabilities();
    }

    bool can_load(const runtime::ModelLoadRequest& request) const override {
        if (!looks_like_moss_sfx_v2(request.model_path)) return false;
        if (request.family_hint && *request.family_hint != family()) return false;
        return true;
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest& request) const override {
        const auto assets = load_moss_sfx_v2_assets(request.model_path);
        runtime::ModelInspection inspection;
        inspection.model_root = assets->resources.model_root();
        inspection.metadata = metadata();
        inspection.capabilities = capabilities();
        inspection.cli = cli();
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(
        const runtime::ModelLoadRequest& request) const override {
        auto assets = load_moss_sfx_v2_assets(request.model_path);
        return std::make_unique<MossSfxV2LoadedModel>(
            metadata(), capabilities(), std::move(assets));
    }
};

}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_moss_sfx_v2_loader() {
    return std::make_shared<MossSfxV2Loader>();
}

}  // namespace engine::models::moss_sfx_v2
