#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

std::string arg_value(int argc, char** argv, const std::string& name, const std::string& fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) return argv[i + 1];
    }
    return fallback;
}

int int_arg(int argc, char** argv, const std::string& name, int fallback) {
    std::string v = arg_value(argc, argv, name, "");
    return v.empty() ? fallback : std::stoi(v);
}

float float_arg(int argc, char** argv, const std::string& name, float fallback) {
    std::string v = arg_value(argc, argv, name, "");
    return v.empty() ? fallback : std::stof(v);
}

engine::core::BackendType parse_backend(const std::string& value) {
    if (value == "cpu")    return engine::core::BackendType::Cpu;
    if (value == "cuda")   return engine::core::BackendType::Cuda;
    if (value == "vulkan") return engine::core::BackendType::Vulkan;
    if (value == "best")   return engine::core::BackendType::BestAvailable;
    throw std::runtime_error("unsupported backend: " + value);
}

}  // namespace

int main(int argc, char** argv) try {
    const std::filesystem::path model_path =
        arg_value(argc, argv, "--model", "models/MOSS-SFX-v2-GGUF");
    const std::string backend_name = arg_value(argc, argv, "--backend", "cuda");
    const int device    = int_arg(argc, argv, "--device", 0);
    const int threads   = int_arg(argc, argv, "--threads", 1);
    const int warmup    = int_arg(argc, argv, "--warmup", 0);
    const int iterations = int_arg(argc, argv, "--iterations", 1);
    const std::string prompt = arg_value(argc, argv, "--prompt",
        "rain falling on a tin roof, steady medium intensity");
    const float duration_sec = float_arg(argc, argv, "--duration", 10.0f);
    const int steps          = int_arg(argc, argv, "--steps", 50);
    const float guidance     = float_arg(argc, argv, "--guidance-scale", 5.0f);
    const uint32_t seed      = static_cast<uint32_t>(int_arg(argc, argv, "--seed", 42));
    const std::filesystem::path output_dir =
        arg_value(argc, argv, "--output-dir", "");
    const std::filesystem::path log_path =
        arg_value(argc, argv, "--log-file", "");

    if (!log_path.empty()) {
        engine::debug::configure_logging(
            engine::debug::LoggingConfig{true, log_path.string()});
    }

    auto registry = engine::runtime::make_default_registry();
    engine::runtime::ModelLoadRequest load_request;
    load_request.model_path = model_path;
    load_request.family_hint = "moss_sfx_v2";
    auto model = registry.load(load_request);

    engine::runtime::TaskSpec task{
        engine::runtime::VoiceTaskKind::AudioGeneration,
        engine::runtime::RunMode::Offline};
    engine::runtime::SessionOptions options;
    options.backend.type    = parse_backend(backend_name);
    options.backend.device  = device;
    options.backend.threads = threads;
    auto session_base = model->create_task_session(task, options);
    auto* session =
        dynamic_cast<engine::runtime::IOfflineVoiceTaskSession*>(
            session_base.get());
    if (!session) {
        throw std::runtime_error(
            "moss_sfx_v2 warmbench: session is not IOfflineVoiceTaskSession");
    }
    session->prepare({});

    engine::runtime::TaskRequest req;
    req.text_input = engine::runtime::Transcript{prompt, "en"};
    req.options["duration_sec"]         = std::to_string(duration_sec);
    req.options["num_inference_steps"]  = std::to_string(steps);
    req.options["guidance_scale"]       = std::to_string(guidance);
    req.options["seed"]                 = std::to_string(seed);

    for (int i = 0; i < warmup; ++i) {
        session->run(req);
    }

    double total_ms = 0.0;
    engine::runtime::TaskResult last_result;
    for (int i = 0; i < iterations; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        last_result = session->run(req);
        const auto t1 = std::chrono::steady_clock::now();
        total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    const double wall_ms = total_ms / static_cast<double>(iterations);
    std::cout << "moss_sfx_v2.wall_ms=" << wall_ms << "\n";

    if (last_result.audio_output.has_value()) {
        const auto& audio = *last_result.audio_output;
        const double seconds = audio.sample_rate > 0
            ? static_cast<double>(audio.samples.size()) /
              static_cast<double>(audio.sample_rate)
            : 0.0;
        std::cout << "moss_sfx_v2.audio_seconds=" << seconds << "\n";
        std::cout << "moss_sfx_v2.rtf="
                  << (seconds > 0.0 ? (wall_ms / 1000.0) / seconds : 0.0)
                  << "\n";

        if (!output_dir.empty()) {
            std::filesystem::create_directories(output_dir);
            const auto out_path = output_dir / "moss_sfx_v2_output.wav";
            engine::audio::write_pcm16_wav(
                out_path, audio.sample_rate, audio.channels, audio.samples);
            std::cout << "moss_sfx_v2.audio_out=" << out_path << "\n";
        }
    } else {
        std::cout << "moss_sfx_v2.audio_output=absent\n";
    }

    return 0;
} catch (const std::exception& ex) {
    std::cerr << "moss_sfx_v2_warm_bench failed: " << ex.what() << "\n";
    return 1;
}
