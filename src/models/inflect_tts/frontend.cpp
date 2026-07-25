#include "engine/models/inflect_tts/frontend.h"

#include "engine/framework/io/dynamic_library.h"

#include <algorithm>
#include <cctype>
#include <iterator>
#include <regex>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

namespace engine::models::inflect_tts {
namespace {

const char * kSymbols[] = {
    "_", ";", ":", ",", ".", "!", "?", "¡", "¿", "—", "…", "\"", "«", "»", "“", "”", " ",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q",
    "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q",
    "r", "s", "t", "u", "v", "w", "x", "y", "z",
    "ɑ", "ɐ", "ɒ", "æ", "ɓ", "ʙ", "β", "ɔ", "ɕ", "ç", "ɗ", "ɖ", "ð", "ʤ", "ə", "ɘ",
    "ɚ", "ɛ", "ɜ", "ɝ", "ɞ", "ɟ", "ʄ", "ɡ", "ɠ", "ɢ", "ʛ", "ɦ", "ɧ", "ħ", "ɥ", "ʜ",
    "ɨ", "ɪ", "ʝ", "ɭ", "ɬ", "ɫ", "ɮ", "ʟ", "ɱ", "ɯ", "ɰ", "ŋ", "ɳ", "ɲ", "ɴ", "ø",
    "ɵ", "ɸ", "θ", "œ", "ɶ", "ʘ", "ɹ", "ɺ", "ɾ", "ɻ", "ʀ", "ʁ", "ɽ", "ʂ", "ʃ", "ʈ",
    "ʧ", "ʉ", "ʊ", "ʋ", "ⱱ", "ʌ", "ɣ", "ɤ", "ʍ", "χ", "ʎ", "ʏ", "ʑ", "ʐ", "ʒ", "ʔ",
    "ʡ", "ʕ", "ʢ", "ǀ", "ǁ", "ǂ", "ǃ", "ˈ", "ˌ", "ː", "ˑ", "ʼ", "ʴ", "ʰ", "ʱ", "ʲ",
    "ʷ", "ˠ", "ˤ", "˞", "↓", "↑", "→", "↗", "↘", "\u0329", "ᵻ",
};

std::unordered_map<std::string, int32_t> make_vocab() {
    std::unordered_map<std::string, int32_t> out;
    for (size_t i = 0; i < std::size(kSymbols); ++i) {
        out.emplace(kSymbols[i], static_cast<int32_t>(i));
    }
    return out;
}

const std::unordered_map<std::string, int32_t> & vocab() {
    static const auto value = make_vocab();
    return value;
}

std::string normalize_spaces(std::string text) {
    std::string out;
    bool pending_space = false;
    for (unsigned char ch : text) {
        if (std::isspace(ch) != 0) {
            pending_space = true;
            continue;
        }
        if (pending_space && !out.empty()) {
            out.push_back(' ');
        }
        pending_space = false;
        out.push_back(static_cast<char>(ch));
    }
    return out;
}

void replace_all(std::string & text, const std::string & from, const std::string & to) {
    size_t pos = 0;
    while ((pos = text.find(from, pos)) != std::string::npos) {
        text.replace(pos, from.size(), to);
        pos += to.size();
    }
}

std::string normalize_inflect_text(std::string text) {
    replace_all(text, "\xE2\x80\x98", "'");
    replace_all(text, "\xE2\x80\x99", "'");
    replace_all(text, "\xE2\x80\x9C", "\"");
    replace_all(text, "\xE2\x80\x9D", "\"");
    replace_all(text, "\xE2\x80\x93", "-");
    replace_all(text, "\xE2\x80\x94", ", ");
    replace_all(text, "\xE2\x80\xA6", "...");
    for (char & ch : text) {
        if (ch == '(' || ch == ')' || ch == '[' || ch == ']' || ch == '{' || ch == '}') {
            ch = ',';
        }
    }
    text = normalize_spaces(text);
    const std::pair<const char *, const char *> word_overrides[] = {
        {"Qwen3", "Qwen three"},
        {"Qwen", "Qwen"},
        {"PyTorch", "pie torch"},
        {"SQLite", "ess cue lite"},
        {"USB-C", "you ess bee see"},
        {"RTX 3060", "ar tee ex thirty sixty"},
        {"RTX 3090", "ar tee ex thirty ninety"},
        {"RTX 4090", "ar tee ex forty ninety"},
        {"RTX 5080", "ar tee ex fifty eighty"},
        {"RTX 5090", "ar tee ex fifty ninety"},
    };
    for (const auto & [source, replacement] : word_overrides) {
        text = std::regex_replace(text, std::regex("\\b" + std::string(source) + "\\b"), replacement);
    }
    const std::pair<const char *, const char *> abbreviations[] = {
        {"Dr\\.", "doctor"},
        {"Mr\\.", "mister"},
        {"Mrs\\.", "missus"},
        {"Ms\\.", "miss"},
        {"Prof\\.", "professor"},
        {"St\\.", "saint"},
        {"vs\\.", "versus"},
        {"etc\\.", "et cetera"},
        {"e\\.g\\.", "for example"},
        {"i\\.e\\.", "that is"},
    };
    for (const auto & [source, replacement] : abbreviations) {
        text = std::regex_replace(text, std::regex("\\b" + std::string(source), std::regex_constants::icase), replacement);
    }
    text = std::regex_replace(text, std::regex(",(?:\\s*,)+"), ",");
    text = std::regex_replace(text, std::regex(",\\s*([.!?])"), "$1");
    text = std::regex_replace(text, std::regex("\\s+([,;:.!?])"), "$1");
    text = std::regex_replace(text, std::regex("([,;:.!?])(?=\\S)"), "$1 ");
    return normalize_spaces(text);
}

std::string trim_ascii(std::string_view text) {
    size_t begin = 0;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
        ++begin;
    }
    size_t end = text.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) {
        --end;
    }
    return std::string(text.substr(begin, end - begin));
}

bool is_inflect_sentence_break(char ch) {
    return ch == '.' || ch == '!' || ch == '?' || ch == ';' || ch == ':';
}

std::vector<std::string> split_inflect_sentences(const std::string & normalized) {
    std::vector<std::string> sentences;
    size_t begin = 0;
    size_t pos = 0;
    while (pos < normalized.size()) {
        if (!is_inflect_sentence_break(normalized[pos]) ||
            pos + 1 >= normalized.size() ||
            std::isspace(static_cast<unsigned char>(normalized[pos + 1])) == 0) {
            ++pos;
            continue;
        }
        auto sentence = trim_ascii(std::string_view(normalized).substr(begin, pos + 1 - begin));
        if (!sentence.empty()) {
            sentences.push_back(std::move(sentence));
        }
        pos += 1;
        while (pos < normalized.size() && std::isspace(static_cast<unsigned char>(normalized[pos])) != 0) {
            ++pos;
        }
        begin = pos;
    }
    auto tail = trim_ascii(std::string_view(normalized).substr(begin));
    if (!tail.empty()) {
        sentences.push_back(std::move(tail));
    }
    return sentences;
}

std::vector<std::string> utf8_symbols(const std::string & text) {
    std::vector<std::string> out;
    for (size_t i = 0; i < text.size();) {
        const unsigned char lead = static_cast<unsigned char>(text[i]);
        size_t width = 1;
        if ((lead & 0x80u) == 0) {
            width = 1;
        } else if ((lead & 0xE0u) == 0xC0u) {
            width = 2;
        } else if ((lead & 0xF0u) == 0xE0u) {
            width = 3;
        } else if ((lead & 0xF8u) == 0xF0u) {
            width = 4;
        } else {
            throw std::runtime_error("invalid UTF-8 in Inflect phoneme text");
        }
        if (i + width > text.size()) {
            throw std::runtime_error("truncated UTF-8 in Inflect phoneme text");
        }
        out.push_back(text.substr(i, width));
        i += width;
    }
    return out;
}

class EspeakFrontend {
public:
    EspeakFrontend() {
        library_ = engine::io::open_dynamic_library(
            {"libespeak-ng.so.1", "libespeak-ng.so", "libespeak-ng.dylib", "libespeak-ng.dll", "espeak-ng.dll"});
        if (library_ == nullptr) {
            throw std::runtime_error("Inflect TTS official frontend requires eSpeak NG shared library");
        }
        initialize_ = reinterpret_cast<InitializeFn>(engine::io::dynamic_library_symbol(library_, "espeak_Initialize"));
        set_voice_by_name_ =
            reinterpret_cast<SetVoiceByNameFn>(engine::io::dynamic_library_symbol(library_, "espeak_SetVoiceByName"));
        text_to_phonemes_ =
            reinterpret_cast<TextToPhonemesFn>(engine::io::dynamic_library_symbol(library_, "espeak_TextToPhonemes"));
        if (initialize_ == nullptr || set_voice_by_name_ == nullptr || text_to_phonemes_ == nullptr) {
            throw std::runtime_error("Inflect TTS eSpeak NG frontend symbols were not resolved");
        }
        constexpr int kAudioOutputSynchronous = 2;
        if (initialize_(kAudioOutputSynchronous, 0, nullptr, 0) < 0 || set_voice_by_name_("en-us") != 0) {
            throw std::runtime_error("Inflect TTS failed to initialize eSpeak NG en-us frontend");
        }
    }

    std::string phonemize_segment(const std::string & text) const {
        constexpr int kCharsUtf8 = 1;
        constexpr int kPhonemesIpa = 2;
        const void * cursor = text.c_str();
        std::string out;
        while (cursor != nullptr && *static_cast<const char *>(cursor) != '\0') {
            const void * before = cursor;
            const char * phonemes = text_to_phonemes_(&cursor, kCharsUtf8, kPhonemesIpa);
            if (phonemes != nullptr && phonemes[0] != '\0') {
                if (!out.empty()) {
                    out.push_back(' ');
                }
                out += phonemes;
            }
            if (cursor == before) {
                break;
            }
        }
        return out;
    }

private:
    using InitializeFn = int (*)(int, int, const char *, int);
    using SetVoiceByNameFn = int (*)(const char *);
    using TextToPhonemesFn = const char * (*)(const void **, int, int);

    engine::io::DynamicLibraryHandle library_ = nullptr;
    InitializeFn initialize_ = nullptr;
    SetVoiceByNameFn set_voice_by_name_ = nullptr;
    TextToPhonemesFn text_to_phonemes_ = nullptr;
};

const EspeakFrontend & espeak_frontend() {
    static const EspeakFrontend frontend;
    return frontend;
}

bool is_preserved_punctuation(char ch) {
    return ch == ',' || ch == ';' || ch == ':' || ch == '.' || ch == '!' || ch == '?';
}

std::string apply_phoneme_overrides(std::string phoneme_text) {
    replace_all(phoneme_text, "sˈæskɐtʃˌuːən", "sɐskˈætʃəwən");
    replace_all(phoneme_text, "flʊɹɹˈɛsənt", "flʊˈɹɛsənt");
    return normalize_spaces(phoneme_text);
}

std::string phonemize_inflect_text(const std::string & text) {
    const std::string normalized = normalize_inflect_text(text);
    std::string out;
    std::string segment;
    const auto flush_segment = [&]() {
        const std::string trimmed = trim_ascii(segment);
        segment.clear();
        if (trimmed.empty()) {
            return;
        }
        const std::string phonemes = espeak_frontend().phonemize_segment(trimmed);
        if (!phonemes.empty()) {
            if (!out.empty() && out.back() != ' ') {
                out.push_back(' ');
            }
            out += phonemes;
        }
    };
    for (char ch : normalized) {
        if (is_preserved_punctuation(ch)) {
            flush_segment();
            if (!out.empty() && out.back() == ' ') {
                out.pop_back();
            }
            out.push_back(ch);
            out.push_back(' ');
            continue;
        }
        segment.push_back(ch);
    }
    flush_segment();
    return apply_phoneme_overrides(out);
}

}  // namespace

std::vector<std::string> split_inflect_text_chunks(const std::string & text, int64_t limit) {
    if (limit <= 0) {
        throw std::runtime_error("Inflect TTS text chunk limit must be positive");
    }
    const std::string normalized = normalize_spaces(text);
    if (normalized.empty()) {
        return {};
    }
    auto sentences = split_inflect_sentences(normalized);
    if (sentences.empty()) {
        sentences.push_back(normalized);
    }
    std::vector<std::string> chunks;
    for (auto sentence : sentences) {
        while (static_cast<int64_t>(sentence.size()) > limit) {
            const auto window = sentence.substr(0, static_cast<size_t>(limit + 1));
            size_t punctuation = std::string::npos;
            for (char mark : {',', ';', ':'}) {
                const size_t pos = window.rfind(mark);
                if (pos != std::string::npos && (punctuation == std::string::npos || pos > punctuation)) {
                    punctuation = pos;
                }
            }
            size_t split_at = std::string::npos;
            if (punctuation != std::string::npos && static_cast<int64_t>(punctuation) >= limit / 2) {
                split_at = punctuation + 1;
            } else {
                split_at = sentence.rfind(' ', static_cast<size_t>(limit));
            }
            if (split_at == std::string::npos || static_cast<int64_t>(split_at) < limit / 2) {
                split_at = static_cast<size_t>(limit);
            }
            auto chunk = trim_ascii(std::string_view(sentence).substr(0, split_at));
            if (!chunk.empty()) {
                chunks.push_back(std::move(chunk));
            }
            sentence = trim_ascii(std::string_view(sentence).substr(split_at));
        }
        if (!sentence.empty()) {
            chunks.push_back(std::move(sentence));
        }
    }
    return chunks;
}

std::vector<int32_t> encode_inflect_text(const std::string & text) {
    const auto phonemes = utf8_symbols(phonemize_inflect_text(text));
    std::vector<int32_t> sequence;
    sequence.reserve(phonemes.size() * 2 + 1);
    sequence.push_back(0);
    for (const auto & symbol : phonemes) {
        const auto it = vocab().find(symbol);
        if (it == vocab().end()) {
            continue;
        }
        sequence.push_back(it->second);
        sequence.push_back(0);
    }
    if (sequence.size() <= 1) {
        throw std::runtime_error("Inflect TTS text produced no phoneme tokens");
    }
    return sequence;
}

}  // namespace engine::models::inflect_tts
