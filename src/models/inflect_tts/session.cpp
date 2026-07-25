#include "engine/models/inflect_tts/session.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"

#include <chrono>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace engine::models::inflect_tts {
namespace {

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
    runtime::TaskResult result;
    result.audio_output = runtime_.synthesize(request.text_input->text, parse_options(request.options));
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start));
    return result;
}

}  // namespace engine::models::inflect_tts
