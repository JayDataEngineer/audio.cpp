#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/inflect_tts/runtime.h"

#include <memory>

namespace engine::models::inflect_tts {

class InflectTTSSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    InflectTTSSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const InflectAssets> assets);

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    runtime::TaskSpec task_;
    std::shared_ptr<const InflectAssets> assets_;
    InflectRuntime runtime_;
};

class InflectLoadedModel final : public runtime::ILoadedVoiceModel {
public:
    InflectLoadedModel(
        runtime::ModelMetadata metadata,
        runtime::CapabilitySet capabilities,
        std::shared_ptr<const InflectAssets> assets);

    const runtime::ModelMetadata & metadata() const noexcept override;
    const runtime::CapabilitySet & capabilities() const noexcept override;
    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override;

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const InflectAssets> assets_;
};

std::unique_ptr<InflectLoadedModel> load_inflect_model(const runtime::ModelLoadRequest & request);
std::shared_ptr<runtime::IVoiceModelLoader> make_inflect_tts_loader();

}  // namespace engine::models::inflect_tts
