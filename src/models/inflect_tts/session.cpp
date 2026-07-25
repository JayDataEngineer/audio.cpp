#include "engine/models/inflect_tts/session.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/text/chunking.h"
#include "engine/models/inflect_tts/frontend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace engine::models::inflect_tts {
namespace {

constexpr int64_t kDefaultTextChunkSize = 280;
constexpr double kEdgeFadeMilliseconds = 5.0;

int64_t boundary_pause_samples(const std::string & chunk, int64_t sample_rate) {
    char ending = '\0';
    for (auto it = chunk.rbegin(); it != chunk.rend(); ++it) {
        if (std::isspace(static_cast<unsigned char>(*it)) == 0) {
            ending = *it;
            break;
        }
    }
    double seconds = 0.08;
    switch (ending) {
        case '?': seconds = 0.28; break;
        case '!': seconds = 0.24; break;
        case '.': seconds = 0.22; break;
        case ';': seconds = 0.16; break;
        case ':': seconds = 0.13; break;
        case ',': seconds = 0.09; break;
        default: break;
    }
    return static_cast<int64_t>(std::llround(seconds * static_cast<double>(sample_rate)));
}

void apply_edge_fade(std::vector<float> & samples, int64_t sample_rate) {
    const int64_t frames = std::min<int64_t>(
        static_cast<int64_t>(std::llround(static_cast<double>(sample_rate) * kEdgeFadeMilliseconds / 1000.0)),
        static_cast<int64_t>(samples.size() / 2));
    if (frames <= 0) {
        return;
    }
    for (int64_t frame = 0; frame < frames; ++frame) {
        const float gain = frames == 1 ? 1.0f : static_cast<float>(frame) / static_cast<float>(frames - 1);
        samples[static_cast<size_t>(frame)] *= gain;
        samples[samples.size() - 1 - static_cast<size_t>(frame)] *= gain;
    }
}

assets::TensorStorageType resolve_weight_type(const runtime::SessionOptions & options) {
    const auto it = options.options.find("inflect_tts.weight_type");
    if (it == options.options.end() || it->second.empty()) {
        return assets::TensorStorageType::Native;
    }
    const auto storage = assets::parse_tensor_storage_type(it->second);
    if (storage == assets::TensorStorageType::Native ||
        storage == assets::TensorStorageType::F32 ||
        storage == assets::TensorStorageType::F16 ||
        storage == assets::TensorStorageType::BF16 ||
        storage == assets::TensorStorageType::Q8_0) {
        return storage;
    }
    throw std::runtime_error("inflect_tts.weight_type supports native, f32, f16, bf16, and q8_0");
}

InflectSynthesisOptions parse_options(const std::unordered_map<std::string, std::string> & options) {
    InflectSynthesisOptions out;
    out.speed = runtime::parse_float_option(options, {"speaking_rate"}).value_or(out.speed);
    out.variation = runtime::parse_float_option(options, {"variation"}).value_or(out.variation);
    out.seed = runtime::parse_u32_option(options, {"seed"}).value_or(runtime::random_u32_seed());
    if (out.speed <= 0.0f) {
        throw std::runtime_error("speaking_rate must be positive");
    }
    if (out.variation < 0.0f) {
        throw std::runtime_error("variation must be non-negative");
    }
    return out;
}

}  // namespace

InflectTTSSession::InflectTTSSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const InflectAssets> assets)
    : RuntimeSessionBase(std::move(options)),
      task_(std::move(task)),
      assets_(std::move(assets)),
      runtime_(assets_, execution_context(), resolve_weight_type(this->options())) {
    if (assets_ == nullptr) {
        throw std::runtime_error("Inflect TTS session requires assets");
    }
    if (task_.task != runtime::VoiceTaskKind::Tts) {
        throw std::runtime_error("Inflect TTS session supports --task tts");
    }
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Inflect TTS session only supports offline mode");
    }
}

std::string InflectTTSSession::family() const {
    return "inflect_tts";
}

runtime::VoiceTaskKind InflectTTSSession::task_kind() const {
    return task_.task;
}

runtime::RunMode InflectTTSSession::run_mode() const {
    return task_.mode;
}

void InflectTTSSession::prepare(const runtime::SessionPreparationRequest &) {
    mark_prepared();
}

runtime::TaskResult InflectTTSSession::run(const runtime::TaskRequest & request) {
    require_prepared("Inflect TTS run()");
    if (!request.text_input.has_value() || request.text_input->text.empty()) {
        throw std::runtime_error("Inflect TTS run requires text_input");
    }
    const auto wall_start = std::chrono::steady_clock::now();
    const int64_t text_chunk_size =
        engine::text::parse_text_chunk_size_override(request.options).value_or(kDefaultTextChunkSize);
    const auto text_chunk_mode = engine::text::parse_text_chunk_mode_override(request.options);
    std::vector<std::string> text_chunks;
    if (text_chunk_mode.has_value()) {
        text_chunks = engine::text::split_text_chunks(request.text_input->text, text_chunk_size, *text_chunk_mode);
    } else {
        text_chunks = split_inflect_text_chunks(request.text_input->text, text_chunk_size);
    }
    engine::debug::trace_log_scalar("inflect_tts.text_chunk_size", text_chunk_size);
    engine::debug::trace_log_scalar(
        "inflect_tts.text_chunk_mode",
        text_chunk_mode.has_value() ? engine::text::text_chunk_mode_name(*text_chunk_mode) : "python");
    engine::debug::trace_log_scalar("inflect_tts.text_chunk_count", static_cast<int64_t>(text_chunks.size()));

    const auto options = parse_options(request.options);
    runtime::TaskResult result;
    for (size_t chunk_index = 0; chunk_index < text_chunks.size(); ++chunk_index) {
        const auto & text_chunk = text_chunks[chunk_index];
        if (text_chunk.empty()) {
            throw std::runtime_error("Inflect TTS text chunking produced an empty request");
        }
        auto chunk_audio = runtime_.synthesize(
            text_chunk,
            InflectSynthesisOptions{
                options.speed,
                options.variation,
                options.seed == 0 ? 0 : options.seed + static_cast<uint32_t>(chunk_index),
            });
        apply_edge_fade(chunk_audio.samples, chunk_audio.sample_rate);
        if (!result.audio_output.has_value()) {
            result.audio_output = std::move(chunk_audio);
            continue;
        }
        if (!result.audio_output->samples.empty()) {
            result.audio_output->samples.insert(
                result.audio_output->samples.end(),
                static_cast<size_t>(boundary_pause_samples(text_chunks[chunk_index - 1], result.audio_output->sample_rate)),
                0.0f);
        }
        runtime::append_audio_buffer(*result.audio_output, chunk_audio);
    }
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start));
    return result;
}

}  // namespace engine::models::inflect_tts
