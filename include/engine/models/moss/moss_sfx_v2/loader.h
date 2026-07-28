#pragma once

#include "engine/framework/runtime/model.h"

#include <memory>

namespace engine::models::moss_sfx_v2 {

std::shared_ptr<runtime::IVoiceModelLoader> make_moss_sfx_v2_loader();

}  // namespace engine::models::moss_sfx_v2
