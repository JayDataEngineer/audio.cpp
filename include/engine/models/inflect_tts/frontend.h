#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace engine::models::inflect_tts {

std::vector<int32_t> encode_inflect_text(const std::string & text);

}  // namespace engine::models::inflect_tts
