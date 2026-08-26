#pragma once

// native_api.h — C ABI for the audiocpp engine_runtime, designed for ctypes
// binding from Python (ComfyUI extension). NO HTTP, NO subprocess.
//
// Architecture: [GGUF files] → [engine_runtime (C++)] → (this .so) → Python
//
// The library exposes the minimal "load model → run inference → get PCM"
// lifecycle. Every function that can fail returns a sentinel (NULL or a
// struct with success=0) and stores a thread-local error message retrieved
// via audiocore_last_error().
//
// Memory ownership:
//   - audiocore_audio_t.samples is heap-allocated by the library and MUST be
//     freed by the caller via audiocore_audio_free().
//   - audiocore_last_error() returns a pointer owned by the library; do NOT
//     free it. The pointer is valid until the next library call on the same
//     thread.

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// ─── Opaque handles ───────────────────────────────────────────────────────

typedef struct audiocore_registry audiocore_registry_t;
typedef struct audiocore_session audiocore_session_t;

// ─── Audio output (POD — safe for ctypes) ─────────────────────────────────

typedef struct {
    float* samples;       // interleaved float32 PCM [-1.0, 1.0], heap-allocated
    int64_t num_samples;  // total float values (frames * channels)
    int sample_rate;      // e.g. 44100, 24000
    int channels;         // 1 = mono, 2 = stereo
    int success;          // 1 = OK, 0 = error (call audiocore_last_error)
} audiocore_audio_t;

// ─── Progress callback ────────────────────────────────────────────────────

typedef void (*audiocore_progress_cb)(int64_t step, int64_t total, const char* label);

// ─── Registry lifecycle ───────────────────────────────────────────────────

// Create the default model registry (loads all compiled-in model family
// loaders). One registry can serve multiple sessions.
audiocore_registry_t* audiocore_registry_create(void);

void audiocore_registry_destroy(audiocore_registry_t* reg);

// ─── Session lifecycle (model loaded + ready for inference) ───────────────

// Load a model and create an inference session. The model stays loaded
// (in-process, on the GPU) until audiocore_session_destroy is called.
//
// family:             model family name ("moss_sfx_v2", "moss_tts",
//                     "ace_step", "qwen3_tts", ...)
// model_path:         directory containing the GGUF files
// task:               "gen" (SFX/music), "tts", "clon", "vdes", "vc", etc.
// backend:            "cuda", "cpu", "vulkan", "metal", "best"
// device:             device ID (0 for single-GPU)
// threads:            CPU thread count (4 is typical)
// model_spec_override: path to model_specs directory, or NULL
// variant:            variant hint ("nano", "local", "CustomVoice", ...), or NULL
// load_options_json:  JSON object of load-time options, or NULL
//
// Returns NULL on error; call audiocore_last_error().
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
);

// Destroy a session, unloading the model and freeing all GPU resources.
// Safe to call with NULL (no-op).
void audiocore_session_destroy(audiocore_session_t* sess);

// ─── Inference ────────────────────────────────────────────────────────────

// Run offline inference. The request_json format matches the server's
// /v1/audio/speech body:
//
//   {
//     "input": "prompt text",           // required — the text prompt
//     "language": "en",                 // optional
//     "seed": 42,                       // optional (int)
//     "guidance_scale": 5.0,            // optional (float) — SFX diffusion
//     "num_inference_steps": 50,        // optional (int) — SFX diffusion
//     "temperature": 0.8,               // optional (float) — TTS autoregressive
//     "top_p": 0.9,                     // optional (float)
//     "top_k": 40,                      // optional (int)
//     "max_tokens": 1000,               // optional (int)
//     "max_steps": 50,                  // optional (int)
//     "repetition_penalty": 1.2,        // optional (float)
//     "instructions": "design text",    // optional — voice design
//     "voice_ref": "/path/to/ref.wav",  // optional — voice cloning reference
//     "reference_text": "ref transcript",// optional — clone reference text
//     "options": {                       // optional — extra engine options
//       "duration_seconds": "10.0",
//       "lyrics": "[verse]\n...",
//       "speaker": "alice"
//     }
//   }
//
// Returns audiocore_audio_t. If success==0, call audiocore_last_error().
// The caller MUST free the samples via audiocore_audio_free().
audiocore_audio_t audiocore_session_run(
    audiocore_session_t* sess,
    const char* request_json
);

// Free the audio buffer returned by audiocore_session_run.
void audiocore_audio_free(audiocore_audio_t* audio);

// ─── Progress ─────────────────────────────────────────────────────────────

// Install a progress callback. The session MAY invoke it from run() loops.
// Pass NULL to clear. The callback is invoked synchronously from the
// inference thread.
void audiocore_session_set_progress(
    audiocore_session_t* sess,
    audiocore_progress_cb cb
);

// ─── Error handling ───────────────────────────────────────────────────────

// Get the last error message (thread-local). Returns NULL if no error.
// The pointer is valid until the next library call on the same thread.
const char* audiocore_last_error(void);

// ─── Introspection ────────────────────────────────────────────────────────

// Get the family name of the loaded model session. Returns NULL if the
// session is invalid.
const char* audiocore_session_family(const audiocore_session_t* sess);

// Query VRAM usage of the session's backend. Returns 0 if unavailable.
int64_t audiocore_session_vram_used_bytes(const audiocore_session_t* sess);

#ifdef __cplusplus
}  // extern "C"
#endif
