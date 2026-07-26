#include "engine/models/inflect_tts/frontend.h"

#include <algorithm>
#include <cctype>
#include <functional>
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

std::string lower_ascii(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return text;
}

std::string regex_replace_fn(
    const std::string & text,
    const std::regex & pattern,
    const std::function<std::string(const std::smatch &)> & fn) {
    std::string out;
    std::sregex_iterator it(text.begin(), text.end(), pattern);
    std::sregex_iterator end;
    size_t pos = 0;
    for (; it != end; ++it) {
        const auto & match = *it;
        out.append(text, pos, static_cast<size_t>(match.position()) - pos);
        out += fn(match);
        pos = static_cast<size_t>(match.position() + match.length());
    }
    out.append(text, pos, std::string::npos);
    return out;
}

std::string int_to_cardinal(int64_t value);

std::string two_digit_cardinal(int value) {
    static const char * below_20[] = {
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
        "eighteen", "nineteen"};
    static const char * tens[] = {
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"};
    if (value < 20) {
        return below_20[value];
    }
    const int ten = value / 10;
    const int one = value % 10;
    return one == 0 ? std::string(tens[ten]) : std::string(tens[ten]) + " " + below_20[one];
}

std::string int_to_cardinal(int64_t value) {
    if (value < 0) {
        return "minus " + int_to_cardinal(-value);
    }
    if (value < 100) {
        return two_digit_cardinal(static_cast<int>(value));
    }
    if (value < 1000) {
        const int64_t hundreds = value / 100;
        const int64_t rest = value % 100;
        return rest == 0
            ? int_to_cardinal(hundreds) + " hundred"
            : int_to_cardinal(hundreds) + " hundred " + int_to_cardinal(rest);
    }
    const std::pair<int64_t, const char *> scales[] = {
        {1000000000000000000LL, "quintillion"},
        {1000000000000000LL, "quadrillion"},
        {1000000000000LL, "trillion"},
        {1000000000LL, "billion"},
        {1000000LL, "million"},
        {1000LL, "thousand"},
    };
    for (const auto & [scale, name] : scales) {
        if (value >= scale) {
            const int64_t head = value / scale;
            const int64_t rest = value % scale;
            return rest == 0
                ? int_to_cardinal(head) + " " + name
                : int_to_cardinal(head) + " " + name + " " + int_to_cardinal(rest);
        }
    }
    return {};
}

std::string int_to_ordinal(int64_t value) {
    static const std::unordered_map<std::string, std::string> replacements = {
        {"one", "first"}, {"two", "second"}, {"three", "third"}, {"five", "fifth"},
        {"eight", "eighth"}, {"nine", "ninth"}, {"twelve", "twelfth"},
    };
    const std::string cardinal = int_to_cardinal(value);
    const size_t last_space = cardinal.find_last_of(' ');
    const std::string prefix = last_space == std::string::npos ? std::string() : cardinal.substr(0, last_space + 1);
    const std::string last = last_space == std::string::npos ? cardinal : cardinal.substr(last_space + 1);
    if (const auto it = replacements.find(last); it != replacements.end()) {
        return prefix + it->second;
    }
    if (!last.empty() && last.back() == 'y') {
        return prefix + last.substr(0, last.size() - 1) + "ieth";
    }
    return cardinal + "th";
}

std::string strip_commas(std::string value) {
    value.erase(std::remove(value.begin(), value.end(), ','), value.end());
    return value;
}

std::string digit_words(const std::string & text) {
    std::string out;
    for (char ch : text) {
        if (std::isdigit(static_cast<unsigned char>(ch)) == 0) {
            continue;
        }
        if (!out.empty()) {
            out.push_back(' ');
        }
        out += int_to_cardinal(ch - '0');
    }
    return out;
}

std::string identifier_digits(const std::string & text) {
    std::string out;
    for (size_t i = 0; i < text.size(); ++i) {
        const char ch = text[i];
        if (std::isdigit(static_cast<unsigned char>(ch)) == 0) {
            continue;
        }
        if (!out.empty()) {
            out.push_back(' ');
        }
        out += ch == '0' && i > 0 ? "oh" : int_to_cardinal(ch - '0');
    }
    return out;
}

const std::unordered_map<char, const char *> & letter_names() {
    static const std::unordered_map<char, const char *> values = {
        {'A', "ay"}, {'B', "bee"}, {'C', "see"}, {'D', "dee"}, {'E', "ee"}, {'F', "eff"},
        {'G', "gee"}, {'H', "aitch"}, {'I', "eye"}, {'J', "jay"}, {'K', "kay"}, {'L', "ell"},
        {'M', "em"}, {'N', "en"}, {'O', "oh"}, {'P', "pee"}, {'Q', "cue"}, {'R', "ar"},
        {'S', "ess"}, {'T', "tee"}, {'U', "you"}, {'V', "vee"}, {'W', "double you"},
        {'X', "ex"}, {'Y', "why"}, {'Z', "zee"},
    };
    return values;
}

std::string letter_name(char ch) {
    const auto upper = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    const auto it = letter_names().find(upper);
    return it == letter_names().end() ? std::string(1, ch) : it->second;
}

std::string expand_identifier_token(const std::string & token) {
    const std::regex pattern("([A-Za-z]?)(\\d+)([A-Za-z]?)");
    std::smatch match;
    if (!std::regex_match(token, match, pattern)) {
        return token;
    }
    std::vector<std::string> pieces;
    const std::string prefix = match[1].str();
    const std::string digits = match[2].str();
    const std::string suffix = match[3].str();
    if (!prefix.empty()) {
        pieces.push_back(letter_name(prefix[0]));
    }
    pieces.push_back(digits.size() == 3 || digits.front() == '0'
        ? identifier_digits(digits)
        : int_to_cardinal(std::stoll(digits)));
    if (!suffix.empty()) {
        pieces.push_back(letter_name(suffix[0]));
    }
    std::string out;
    for (const auto & piece : pieces) {
        if (!out.empty()) {
            out.push_back(' ');
        }
        out += piece;
    }
    return out;
}

bool valid_date(int year, int month, int day) {
    if (month < 1 || month > 12 || day < 1) {
        return false;
    }
    const int days[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    int max_day = days[month - 1];
    const bool leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    if (month == 2 && leap) {
        max_day = 29;
    }
    return day <= max_day;
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
    text = regex_replace_fn(text, std::regex("\\b([A-Z])(?:\\.([A-Z]))+\\."), [](const std::smatch & match) {
        std::string out;
        for (char ch : match.str()) {
            if (std::isupper(static_cast<unsigned char>(ch)) != 0) {
                if (!out.empty()) {
                    out.push_back(' ');
                }
                out.push_back(ch);
            }
        }
        return out;
    });
    text = regex_replace_fn(
        text,
        std::regex("\\b(apartment|apt\\.?|suite|unit|room|flight|extension|order|invoice|locker|aisle|gate)\\s+([A-Za-z]?\\d{1,4}[A-Za-z]?)\\b", std::regex_constants::icase),
        [](const std::smatch & match) {
            return match[1].str() + " " + expand_identifier_token(match[2].str());
        });
    text = regex_replace_fn(
        text,
        std::regex("\\b(\\d{3})(?=\\s+(?:North|South|East|West)\\b)", std::regex_constants::icase),
        [](const std::smatch & match) {
            return identifier_digits(match[1].str());
        });
    text = regex_replace_fn(text, std::regex("\\$(\\d[\\d,]*(?:\\.\\d{1,2})?)"), [](const std::smatch & match) {
        const std::string raw = strip_commas(match[1].str());
        const size_t dot = raw.find('.');
        const int64_t dollars = std::stoll(dot == std::string::npos ? raw : raw.substr(0, dot));
        std::string out = int_to_cardinal(dollars) + (dollars == 1 ? " dollar" : " dollars");
        if (dot != std::string::npos) {
            std::string cents = raw.substr(dot + 1);
            cents.resize(2, '0');
            const int64_t cent_count = std::stoll(cents);
            if (cent_count != 0) {
                out += " and " + int_to_cardinal(cent_count) + (cent_count == 1 ? " cent" : " cents");
            }
        }
        return out;
    });
    static const char * months[] = {
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"};
    text = regex_replace_fn(
        text,
        std::regex("\\b(0?[1-9]|1[0-2])/(0?[1-9]|[12]\\d|3[01])/(20\\d{2}|19\\d{2})\\b"),
        [&](const std::smatch & match) {
            const int month = std::stoi(match[1].str());
            const int day = std::stoi(match[2].str());
            const int year = std::stoi(match[3].str());
            if (!valid_date(year, month, day)) {
                return match.str();
            }
            return std::string(months[month - 1]) + " " + int_to_ordinal(day) + " " + int_to_cardinal(year);
        });
    text = regex_replace_fn(
        text,
        std::regex("\\b(\\d{1,2}):(\\d{2})\\s*([AaPp]\\.?\\s*[Mm]\\.?)?\\b"),
        [](const std::smatch & match) {
            const int hour = std::stoi(match[1].str());
            const int minute = std::stoi(match[2].str());
            std::string out = int_to_cardinal(hour);
            if (minute == 0) {
                out += " o clock";
            } else if (minute < 10) {
                out += " oh " + int_to_cardinal(minute);
            } else {
                out += " " + int_to_cardinal(minute);
            }
            std::string suffix = lower_ascii(match[3].str());
            suffix.erase(std::remove_if(suffix.begin(), suffix.end(), [](unsigned char ch) {
                return !std::isalpha(ch);
            }), suffix.end());
            for (char ch : suffix) {
                out += " ";
                out.push_back(ch);
            }
            return out;
        });
    text = regex_replace_fn(
        text,
        std::regex("\\b(\\d{1,2})\\s*([AaPp]\\.?\\s*[Mm]\\.?)\\b"),
        [](const std::smatch & match) {
            std::string suffix = lower_ascii(match[2].str());
            suffix.erase(std::remove_if(suffix.begin(), suffix.end(), [](unsigned char ch) {
                return !std::isalpha(ch);
            }), suffix.end());
            std::string out = int_to_cardinal(std::stoll(match[1].str()));
            for (char ch : suffix) {
                out += " ";
                out.push_back(ch);
            }
            return out;
        });
    text = regex_replace_fn(text, std::regex("\\b(\\d{3})-(\\d{4})\\b"), [](const std::smatch & match) {
        return digit_words(match[1].str()) + ", " + digit_words(match[2].str());
    });
    text = regex_replace_fn(text, std::regex("\\b\\d+(?:\\.\\d+){2,}\\b"), [](const std::smatch & match) {
        std::string out;
        size_t begin = 0;
        const std::string value = match.str();
        while (true) {
            const size_t dot = value.find('.', begin);
            const auto part = value.substr(begin, dot == std::string::npos ? std::string::npos : dot - begin);
            if (!out.empty()) {
                out += " point ";
            }
            out += int_to_cardinal(std::stoll(part));
            if (dot == std::string::npos) {
                break;
            }
            begin = dot + 1;
        }
        return out;
    });
    text = regex_replace_fn(text, std::regex("\\b(\\d+)\\.(\\d+)\\b"), [](const std::smatch & match) {
        return int_to_cardinal(std::stoll(match[1].str())) + " point " + digit_words(match[2].str());
    });
    text = regex_replace_fn(text, std::regex("\\b(\\d+)(st|nd|rd|th)\\b", std::regex_constants::icase), [](const std::smatch & match) {
        return int_to_ordinal(std::stoll(match[1].str()));
    });
    text = regex_replace_fn(text, std::regex("\\b\\d[\\d,]*\\b"), [](const std::smatch & match) {
        const std::string value = strip_commas(match.str());
        if (value.size() >= 5 && value.rfind("20", 0) != 0) {
            return digit_words(value);
        }
        return int_to_cardinal(std::stoll(value));
    });
    text = regex_replace_fn(text, std::regex("\\b[A-Z]{2,}\\b"), [](const std::smatch & match) {
        std::string out;
        for (char ch : match.str()) {
            if (!out.empty()) {
                out.push_back(' ');
            }
            out += letter_name(ch);
        }
        return out;
    });
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

bool is_preserved_punctuation(char ch) {
    return ch == ',' || ch == ';' || ch == ':' || ch == '.' || ch == '!' || ch == '?';
}

std::string translate_misaki_phonemes_to_inflect_symbols(const std::string & phonemes) {
    std::string out;
    out.reserve(phonemes.size());
    const auto symbols = utf8_symbols(phonemes);
    for (const auto & symbol : symbols) {
        if (symbol.size() == 1) {
            switch (symbol[0]) {
                case 'A': out += "eɪ"; continue;
                case 'I': out += "aɪ"; continue;
                case 'O': out += "oʊ"; continue;
                case 'Q': out += "ɔɪ"; continue;
                case 'T': out += "ɾ"; continue;
                case 'W': out += "aʊ"; continue;
                case 'Y': out += "ju"; continue;
                default: break;
            }
        }
        if (symbol == "ʧ") {
            out += "tʃ";
            continue;
        }
        if (symbol == "ʤ") {
            out += "dʒ";
            continue;
        }
        if (symbol == "ᵊ") {
            out += "ə";
            continue;
        }
        out += symbol;
    }
    return out;
}

std::string apply_phoneme_overrides(std::string phoneme_text) {
    replace_all(phoneme_text, "əɹi", "ɚɹi");
    replace_all(phoneme_text, "əɹ", "ɚ");
    replace_all(phoneme_text, "sˈæskɐtʃˌuːən", "sɐskˈætʃəwən");
    replace_all(phoneme_text, "flʊɹɹˈɛsənt", "flʊˈɹɛsənt");
    return normalize_spaces(phoneme_text);
}

std::string phonemize_inflect_text(
    const std::string & text,
    const engine::text::english_g2p::EnglishG2P & english_g2p) {
    const std::string normalized = normalize_inflect_text(text);
    std::string out;
    std::string segment;
    const auto flush_segment = [&]() {
        const std::string trimmed = trim_ascii(segment);
        segment.clear();
        if (trimmed.empty()) {
            return;
        }
        const auto phonemes = english_g2p(trimmed, false).first;
        if (!out.empty() && out.back() != ' ') {
            out.push_back(' ');
        }
        out += translate_misaki_phonemes_to_inflect_symbols(phonemes);
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

std::vector<int32_t> encode_inflect_text(
    const std::string & text,
    const engine::text::english_g2p::EnglishG2P & english_g2p) {
    const auto phonemes = utf8_symbols(phonemize_inflect_text(text, english_g2p));
    std::vector<int32_t> sequence;
    sequence.reserve(phonemes.size() * 2 + 1);
    sequence.push_back(0);
    for (const auto & symbol : phonemes) {
        const auto it = vocab().find(symbol);
        if (it == vocab().end()) {
            throw std::runtime_error("Inflect TTS frontend produced unsupported symbol: " + symbol);
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
