#pragma once

#include "engine/framework/runtime/model.h"

#include <memory>

namespace engine::models::inflect_tts {

std::shared_ptr<runtime::IVoiceModelLoader> make_inflect_tts_loader();

}  // namespace engine::models::inflect_tts
