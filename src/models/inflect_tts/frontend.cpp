#include "engine/models/inflect_tts/frontend.h"

#include <algorithm>
#include <cctype>
#include <iterator>
#include <sstream>
#include <stdexcept>
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

std::string lower_ascii(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return text;
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

std::string word_to_ipa(const std::string & word) {
    static const std::unordered_map<std::string, std::string> words = {
        {"a", "ə"}, {"an", "ən"}, {"and", "ænd"}, {"are", "ɑɹ"}, {"audio", "ˈɔdioʊ"},
        {"cpp", "si pi pi"}, {"dot", "dɑt"}, {"for", "fɔɹ"}, {"is", "ɪz"}, {"local", "ˈloʊkəl"},
        {"of", "ʌv"}, {"small", "smɔl"}, {"still", "stɪl"}, {"the", "ðə"}, {"to", "tu"},
        {"useful", "ˈjusfəl"}, {"voice", "vɔɪs"}, {"can", "kæn"}, {"be", "bi"},
    };
    if (const auto it = words.find(word); it != words.end()) {
        return it->second;
    }
    std::string out;
    for (size_t i = 0; i < word.size(); ++i) {
        const char ch = word[i];
        const char next = i + 1 < word.size() ? word[i + 1] : '\0';
        if (ch == 'c' && next == 'h') {
            out += "ʧ";
            ++i;
        } else if (ch == 's' && next == 'h') {
            out += "ʃ";
            ++i;
        } else if (ch == 't' && next == 'h') {
            out += "θ";
            ++i;
        } else if (ch == 'n' && next == 'g') {
            out += "ŋ";
            ++i;
        } else {
            switch (ch) {
                case 'a': out += "æ"; break;
                case 'b': out += "b"; break;
                case 'c': out += "k"; break;
                case 'd': out += "d"; break;
                case 'e': out += "ɛ"; break;
                case 'f': out += "f"; break;
                case 'g': out += "ɡ"; break;
                case 'h': out += "h"; break;
                case 'i': out += "ɪ"; break;
                case 'j': out += "ʤ"; break;
                case 'k': out += "k"; break;
                case 'l': out += "l"; break;
                case 'm': out += "m"; break;
                case 'n': out += "n"; break;
                case 'o': out += "ɑ"; break;
                case 'p': out += "p"; break;
                case 'q': out += "k"; break;
                case 'r': out += "ɹ"; break;
                case 's': out += "s"; break;
                case 't': out += "t"; break;
                case 'u': out += "ʌ"; break;
                case 'v': out += "v"; break;
                case 'w': out += "w"; break;
                case 'x': out += "ks"; break;
                case 'y': out += "j"; break;
                case 'z': out += "z"; break;
                default: break;
            }
        }
    }
    return out;
}

std::string approximate_phonemes(const std::string & text) {
    const std::string normalized = lower_ascii(normalize_spaces(text));
    std::string out;
    std::string word;
    const auto flush_word = [&]() {
        if (word.empty()) {
            return;
        }
        if (!out.empty() && out.back() != ' ') {
            out.push_back(' ');
        }
        out += word_to_ipa(word);
        word.clear();
    };
    for (unsigned char ch : normalized) {
        if (std::isalpha(ch) != 0) {
            word.push_back(static_cast<char>(ch));
            continue;
        }
        flush_word();
        if (std::isdigit(ch) != 0) {
            static const char * names[] = {"zɪɹoʊ", "wʌn", "tu", "θɹi", "fɔɹ", "faɪv", "sɪks", "sɛvən", "eɪt", "naɪn"};
            if (!out.empty() && out.back() != ' ') {
                out.push_back(' ');
            }
            out += names[ch - '0'];
        } else if (ch == ';' || ch == ':' || ch == ',' || ch == '.' || ch == '!' || ch == '?' || ch == ' ') {
            out.push_back(static_cast<char>(ch));
        }
    }
    flush_word();
    return normalize_spaces(out);
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

}  // namespace

std::vector<int32_t> encode_inflect_text(const std::string & text) {
    const auto phonemes = utf8_symbols(approximate_phonemes(text));
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

std::vector<std::string> split_inflect_text_chunks(const std::string & text, size_t max_chars) {
    std::vector<std::string> chunks;
    std::string current;
    std::istringstream words(text);
    std::string word;
    while (words >> word) {
        const bool punctuation_break =
            !word.empty() && (word.back() == '.' || word.back() == '!' || word.back() == '?' || word.back() == ';');
        if (!current.empty() && current.size() + word.size() + 1 > max_chars) {
            chunks.push_back(current);
            current.clear();
        }
        if (!current.empty()) {
            current.push_back(' ');
        }
        current += word;
        if (punctuation_break && !current.empty()) {
            chunks.push_back(current);
            current.clear();
        }
    }
    if (!current.empty()) {
        chunks.push_back(current);
    }
    if (chunks.empty()) {
        throw std::runtime_error("Inflect TTS requires non-empty text");
    }
    return chunks;
}

}  // namespace engine::models::inflect_tts
