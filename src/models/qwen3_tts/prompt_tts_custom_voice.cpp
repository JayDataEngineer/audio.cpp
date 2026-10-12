#include "engine/models/qwen3_tts/prompt_tts_custom_voice.h"

#include <stdexcept>
#include <string>

namespace engine::models::qwen3_tts {
namespace {

void require_token_limit(size_t actual, int64_t limit, const char * what) {
    if (limit <= 0) {
        throw std::runtime_error(std::string("Qwen3 custom voice ") + what + " token limit must be positive");
    }
    if (actual > static_cast<size_t>(limit)) {
        throw std::runtime_error(
            std::string("Qwen3 custom voice ") + what + " token count "
            + std::to_string(actual) + " exceeds limit " + std::to_string(limit));
    }
}

}  // namespace

Qwen3TTSCustomVoicePromptBuilder::Qwen3TTSCustomVoicePromptBuilder(
    const Qwen3TextTokenizer & tokenizer,
    int64_t text_token_limit,
    int64_t instruction_token_limit)
    : tokenizer_(tokenizer),
      text_token_limit_(text_token_limit),
      instruction_token_limit_(instruction_token_limit) {}

Qwen3TalkerPrefill Qwen3TTSCustomVoicePromptBuilder::build_prefill(const Qwen3TTSRequest & request) const {
    if (!request.custom_voice.has_value()) {
        throw std::runtime_error("Qwen3 custom voice prefill requires custom voice input");
    }
    // Speaker name is optional — PERIOD (operator ruling 2026-10-12,
    // upstream parity modeling_qwen3_tts.py:2088: `if speaker == "" or
    // speaker == None: speaker_embed = None  # Instruct create speaker`).
    // A speakerless, referenceless request is PROMPT-ONLY: the prefill
    // builds with no speaker row and the model creates the voice. The
    // old throw here invented a requirement the upstream model does not
    // have ("requires speaker or voice reference" — lane injury
    // run-d5f94f216419).
    Qwen3TalkerPrefill prefill;
    prefill.prompt_mode = Qwen3TalkerPromptMode::CustomVoice;
    prefill.input_ids = tokenizer_.encode(tokenizer_.build_assistant_prompt(request.text));
    require_token_limit(prefill.input_ids.size(), text_token_limit_, "text");
    if (!request.custom_voice->instruct.empty()) {
        prefill.instruct_ids = tokenizer_.encode(tokenizer_.build_instruct_prompt(request.custom_voice->instruct));
        require_token_limit(prefill.instruct_ids.size(), instruction_token_limit_, "instruction");
    }
    prefill.speaker = request.custom_voice->speaker;
    prefill.language = request.language;
    return prefill;
}

}  // namespace engine::models::qwen3_tts
