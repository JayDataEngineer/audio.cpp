#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace engine::models::inflect_tts {

std::vector<int32_t> encode_inflect_text(const std::string & text);
std::vector<std::string> split_inflect_text_chunks(const std::string & text, size_t max_chars);

}  // namespace engine::models::inflect_tts
