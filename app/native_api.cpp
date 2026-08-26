// native_api.cpp — C ABI wrapper around engine_runtime for Python ctypes.
//
// ELIMINATES THE HTTP SERVER. The ComfyUI extension loads this .so directly
// and calls these functions to load models, run inference, and get PCM audio
// back — all in-process, no subprocess, no HTTP, no port management.
//
// The inference path mirrors app/cli/main.cpp's offline flow exactly:
//   registry.load(load_request)
//   → model->create_task_session(task_spec, session_options)
//   → session->prepare(build_preparation_request(request))
//   → offline->run(request) → TaskResult.audio_output

#include "native_api.h"

#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/module.h"
#include "engine/framework/io/json.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session.h"

#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

// ─── Thread-local error storage ───────────────────────────────────────────

namespace {

thread_local std::string g_last_error;

void set_error(const std::string& msg) {
    g_last_error = msg;
}

void clear_error() {
    g_last_error.clear();
}

// Convert a JSON value to its option-string form (matching the CLI's
// json_option_string behavior — numbers and bools become their string
// representation, strings pass through).
std::string json_to_option_string(const engine::io::json::Value& v) {
    if (v.is_string()) {
        return v.as_string();
    }
    if (v.is_number()) {
        const auto d = v.as_number();
        // Integer-valued doubles print without a trailing .0 to match CLI
        if (d == static_cast<int64_t>(d)) {
            return std::to_string(static_cast<int64_t>(d));
        }
        return std::to_string(d);
    }
    if (v.is_bool()) {
        return v.as_bool() ? "true" : "false";
    }
    // Null, object, array — fall back to empty
    return "";
}

void add_option_from_json(
    std::unordered_map<std::string, std::string>& options,
    const engine::io::json::Value& object,
    const std::string& field,
    const std::string& option_key
) {
    const auto* value = object.find(field);
    if (value != nullptr && !value->is_null()) {
        options[option_key] = json_to_option_string(*value);
    }
}

engine::core::BackendType parse_backend_str(const char* backend) {
    std::string s(backend && *backend ? backend : "cpu");
    if (s == "cuda" || s == "ggml_cuda" || s == "CUDA") {
        return engine::core::BackendType::Cuda;
    }
    if (s == "vulkan" || s == "Vulkan") {
        return engine::core::BackendType::Vulkan;
    }
    if (s == "metal" || s == "Metal") {
        return engine::core::BackendType::Metal;
    }
    if (s == "best" || s == "best_available" || s == "auto") {
        return engine::core::BackendType::BestAvailable;
    }
    return engine::core::BackendType::Cpu;
}

engine::runtime::VoiceTaskKind parse_task_str(const char* task) {
    std::string s(task && *task ? task : "tts");
    if (s == "vad") return engine::runtime::VoiceTaskKind::Vad;
    if (s == "asr") return engine::runtime::VoiceTaskKind::Asr;
    if (s == "diar") return engine::runtime::VoiceTaskKind::Diarization;
    if (s == "sep") return engine::runtime::VoiceTaskKind::SourceSeparation;
    if (s == "gen" || s == "generation" || s == "audio_generation") {
        return engine::runtime::VoiceTaskKind::AudioGeneration;
    }
    if (s == "tts") return engine::runtime::VoiceTaskKind::Tts;
    if (s == "clon" || s == "clone" || s == "voice_cloning") {
        return engine::runtime::VoiceTaskKind::VoiceCloning;
    }
    if (s == "vc" || s == "voice_conversion") {
        return engine::runtime::VoiceTaskKind::VoiceConversion;
    }
    if (s == "s2s" || s == "speech_to_speech") {
        return engine::runtime::VoiceTaskKind::SpeechToSpeech;
    }
    if (s == "align" || s == "alignment") {
        return engine::runtime::VoiceTaskKind::Alignment;
    }
    if (s == "vdes" || s == "voice_design") {
        return engine::runtime::VoiceTaskKind::VoiceDesign;
    }
    if (s == "spk" || s == "speaker") {
        return engine::runtime::VoiceTaskKind::SpeakerRecognition;
    }
    if (s == "svc") return engine::runtime::VoiceTaskKind::Svc;
    // Default to TTS — the most common task
    return engine::runtime::VoiceTaskKind::Tts;
}

}  // namespace

// ─── Opaque handle implementations ────────────────────────────────────────

struct audiocore_registry {
    engine::runtime::ModelRegistry impl;
};

struct audiocore_session {
    std::unique_ptr<engine::runtime::IVoiceTaskSession> session;
    engine::runtime::IOfflineVoiceTaskSession* offline = nullptr;
    engine::core::BackendConfig backend_config = {};
    std::string family_name;
    // Keep the progress callback alive
    audiocore_progress_cb progress_cb = nullptr;
};

// ─── C ABI implementations ────────────────────────────────────────────────

extern "C" {

const char* audiocore_last_error(void) {
    return g_last_error.empty() ? nullptr : g_last_error.c_str();
}

audiocore_registry_t* audiocore_registry_create(void) {
    clear_error();
    try {
        auto reg = engine::runtime::make_default_registry();
        return new audiocore_registry_t{std::move(reg)};
    } catch (const std::exception& e) {
        set_error(std::string("registry_create failed: ") + e.what());
        return nullptr;
    }
}

void audiocore_registry_destroy(audiocore_registry_t* reg) {
    delete reg;
}

audiocore_session_t* audiocore_session_create(
    audiocore_registry_t* reg,
    const char* family,
    const char* model_path,
    const char* task,
    const char* backend,
    int device,
    int threads,
    const char* model_spec_override,
    const char* variant,
    const char* load_options_json
) {
    clear_error();
    if (reg == nullptr) {
        set_error("session_create: registry is NULL");
        return nullptr;
    }
    if (model_path == nullptr || *model_path == '\0') {
        set_error("session_create: model_path is required");
        return nullptr;
    }

    try {
        engine::runtime::ModelLoadRequest load_request;
        load_request.model_path = std::filesystem::path(model_path);
        if (family && *family) {
            load_request.family_hint = family;
        }
        if (model_spec_override && *model_spec_override) {
            load_request.model_spec_override = std::filesystem::path(model_spec_override);
        }
        if (variant && *variant) {
            load_request.options["variant"] = variant;
        }
        // Parse load options JSON if provided
        if (load_options_json && *load_options_json) {
            const auto opts = engine::io::json::parse(load_options_json);
            if (opts.is_object()) {
                for (const auto& [key, value] : opts.as_object()) {
                    if (!value.is_null()) {
                        load_request.options[key] = json_to_option_string(value);
                    }
                }
            }
        }

        engine::runtime::SessionOptions session_options;
        session_options.backend.type = parse_backend_str(backend);
        session_options.backend.device = device;
        session_options.backend.threads = (threads > 0) ? threads : 4;

        auto loaded_model = reg->impl.load(load_request);

        engine::runtime::TaskSpec task_spec;
        task_spec.task = parse_task_str(task);
        task_spec.mode = engine::runtime::RunMode::Offline;

        auto session = loaded_model->create_task_session(task_spec, session_options);
        auto* offline = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession*>(session.get());
        if (offline == nullptr) {
            set_error("session_create: model does not provide offline execution for task '" +
                      std::string(task ? task : "") + "'");
            return nullptr;
        }

        auto* handle = new audiocore_session_t;
        handle->family_name = session->family();
        handle->session = std::move(session);
        handle->offline = offline;
        handle->backend_config = session_options.backend;
        return handle;
    } catch (const std::exception& e) {
        set_error(std::string("session_create failed: ") + e.what());
        return nullptr;
    }
}

void audiocore_session_destroy(audiocore_session_t* sess) {
    delete sess;
}

audiocore_audio_t audiocore_session_run(
    audiocore_session_t* sess,
    const char* request_json
) {
    audiocore_audio_t out = {};
    out.success = 0;

    if (sess == nullptr) {
        set_error("session_run: session is NULL");
        return out;
    }
    if (request_json == nullptr || *request_json == '\0') {
        set_error("session_run: request_json is required");
        return out;
    }

    clear_error();
    try {
        const auto body = engine::io::json::parse(request_json);

        engine::runtime::TaskRequest request;

        // "input" (required by the server) → text_input
        const auto* input_val = body.find("input");
        if (input_val != nullptr && input_val->is_string()) {
            const auto language = engine::io::json::optional_string(body, "language", "");
            request.text_input = engine::runtime::Transcript{
                input_val->as_string(),
                language,
            };
        }

        // Merge "options" object into request.options
        const auto* opts_val = body.find("options");
        if (opts_val != nullptr && opts_val->is_object()) {
            for (const auto& [key, value] : opts_val->as_object()) {
                if (!value.is_null()) {
                    request.options[key] = json_to_option_string(value);
                }
            }
        }

        // Map known fields → options (mirrors build_speech_request exactly)
        add_option_from_json(request.options, body, "seed", "seed");
        add_option_from_json(request.options, body, "temperature", "temperature");
        add_option_from_json(request.options, body, "top_k", "top_k");
        add_option_from_json(request.options, body, "top_p", "top_p");
        add_option_from_json(request.options, body, "max_tokens", "max_tokens");
        add_option_from_json(request.options, body, "max_steps", "max_steps");
        add_option_from_json(request.options, body, "repetition_penalty", "repetition_penalty");
        add_option_from_json(request.options, body, "guidance_scale", "guidance_scale");
        add_option_from_json(request.options, body, "num_inference_steps", "num_inference_steps");
        add_option_from_json(request.options, body, "do_sample", "do_sample");
        add_option_from_json(request.options, body, "num_beams", "num_beams");
        add_option_from_json(request.options, body, "duration_seconds", "duration_seconds");
        add_option_from_json(request.options, body, "text_chunk_size", "text_chunk_size");

        // "instructions" → options["instruct"]
        if (const auto* instr = body.find("instructions")) {
            if (instr->is_string()) {
                request.options["instruct"] = instr->as_string();
            }
        }

        // Voice reference (for cloning/TTS) — reads a WAV file path
        const auto* voice_ref = body.find("voice_ref");
        if (voice_ref != nullptr && voice_ref->is_string()) {
            const auto ref_path = std::filesystem::path(voice_ref->as_string());
            if (std::filesystem::exists(ref_path)) {
                const auto wav = engine::audio::read_wav_f32(ref_path);
                engine::runtime::VoiceReference voice_ref_data;
                voice_ref_data.audio = engine::runtime::AudioBuffer{
                    wav.sample_rate,
                    wav.channels,
                    wav.samples,
                };
                if (!request.voice.has_value()) {
                    request.voice = engine::runtime::VoiceCondition{};
                }
                request.voice->speaker = voice_ref_data;
            }
        }

        // Named voice — reads a voice ID string (NOT a file path).
        // Mirrors the old HTTP server's runtime.cpp:782:
        //   voice.speaker->cached_voice_id = value->as_string();
        // This was MISSING from the initial native_api.cpp — the voice param
        // was silently dropped (named voice selection produced the default
        // voice with zero error). Fixed 2026-08-11.
        if (const auto* voice_val = body.find("voice"); voice_val != nullptr && voice_val->is_string()) {
            if (!request.voice.has_value()) {
                request.voice = engine::runtime::VoiceCondition{};
            }
            if (!request.voice->speaker.has_value()) {
                request.voice->speaker = engine::runtime::VoiceReference{};
            }
            request.voice->speaker->cached_voice_id = voice_val->as_string();
        }

        // reference_text → options["reference_text"]
        if (const auto* ref_text = body.find("reference_text")) {
            if (ref_text->is_string()) {
                request.options["reference_text"] = ref_text->as_string();
            }
        }

        // Prepare + run (exactly like the CLI offline path)
        sess->session->prepare(engine::runtime::build_preparation_request(request));
        const auto result = sess->offline->run(request);

        if (result.audio_output.has_value()) {
            const auto& audio = *result.audio_output;
            out.num_samples = static_cast<int64_t>(audio.samples.size());
            out.sample_rate = audio.sample_rate;
            out.channels = audio.channels;
            if (out.num_samples > 0) {
                out.samples = new float[out.num_samples];
                std::memcpy(out.samples, audio.samples.data(),
                            out.num_samples * sizeof(float));
                out.success = 1;
            } else {
                set_error("session_run: model produced 0 samples");
                out.success = 0;
            }
        } else {
            // Check for named audio outputs (some models output named channels)
            if (!result.named_audio_outputs.empty()) {
                const auto& audio = result.named_audio_outputs[0].audio;
                out.num_samples = static_cast<int64_t>(audio.samples.size());
                out.sample_rate = audio.sample_rate;
                out.channels = audio.channels;
                if (out.num_samples > 0) {
                    out.samples = new float[out.num_samples];
                    std::memcpy(out.samples, audio.samples.data(),
                                out.num_samples * sizeof(float));
                    out.success = 1;
                }
            } else {
                set_error("session_run: model produced no audio output");
                out.success = 0;
            }
        }
    } catch (const std::exception& e) {
        set_error(std::string("session_run failed: ") + e.what());
        out.success = 0;
    }

    return out;
}

void audiocore_audio_free(audiocore_audio_t* audio) {
    if (audio && audio->samples) {
        delete[] audio->samples;
        audio->samples = nullptr;
        audio->num_samples = 0;
    }
}

void audiocore_session_set_progress(
    audiocore_session_t* sess,
    audiocore_progress_cb cb
) {
    if (sess == nullptr || sess->session == nullptr) return;
    sess->progress_cb = cb;
    sess->session->set_progress_callback(
        [cb](int64_t step, int64_t total, const std::string& label) {
            if (cb) {
                cb(step, total, label.c_str());
            }
        }
    );
}

const char* audiocore_session_family(const audiocore_session_t* sess) {
    if (sess == nullptr) return nullptr;
    return sess->family_name.c_str();
}

int64_t audiocore_session_vram_used_bytes(const audiocore_session_t* sess) {
    if (sess == nullptr) return 0;
    try {
        const auto snapshot = engine::core::query_backend_memory(sess->backend_config);
        if (snapshot.available) {
            return snapshot.used_bytes;
        }
    } catch (...) {
        // VRAM query is best-effort
    }
    return 0;
}

}  // extern "C"
