#pragma once

#include "engine/framework/text/english_g2p.h"

#include <cstdint>
#include <string>
#include <vector>

namespace engine::models::inflect_tts {

std::vector<int32_t> encode_inflect_text(
    const std::string & text,
    const engine::text::english_g2p::EnglishG2P & english_g2p);
std::vector<std::string> split_inflect_text_chunks(const std::string & text, int64_t limit);

}  // namespace engine::models::inflect_tts
