#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace engine::models::inflect_tts {

std::vector<int32_t> encode_inflect_text(const std::string & text);
std::vector<std::string> split_inflect_text_chunks(const std::string & text, int64_t limit);

}  // namespace engine::models::inflect_tts
