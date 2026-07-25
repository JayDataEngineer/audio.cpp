#pragma once

#include "engine/framework/core/backend.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/inflect_tts/assets.h"

#include <memory>
#include <string>
#include <vector>

namespace engine::models::inflect_tts {

struct InflectSynthesisOptions {
    float speed = 1.0f;
    float variation = 0.667f;
    uint32_t seed = 0;
};

class InflectRuntime {
public:
    InflectRuntime(
        std::shared_ptr<const InflectAssets> assets,
        engine::core::ExecutionContext & execution_context,
        assets::TensorStorageType weight_type);

    runtime::AudioBuffer synthesize(
        const std::string & text,
        const InflectSynthesisOptions & options);

private:
    class Impl;
    std::shared_ptr<Impl> impl_;
};

}  // namespace engine::models::inflect_tts
