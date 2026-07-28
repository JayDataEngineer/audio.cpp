#pragma once

#include "engine/framework/runtime/session.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/moss/moss_sfx_v2/assets.h"
#include "engine/models/moss/moss_sfx_v2/dit_runner.h"
#include "engine/models/moss/moss_sfx_v2/text_encoder.h"
#include "engine/models/moss/moss_sfx_v2/vae_runner.h"
#include "engine/models/moss/moss_sfx_v2/weight_bridge.h"

#include <memory>
#include <string>

namespace engine::models::moss_sfx_v2 {

// Offline sound-effect generation session: flow-matching DiT denoising loop →
// DAC VAE decode → mono PCM.
//
//   prepare()  — bridges DiT + VAE GGUF weights into a no_alloc ggml_context
//                via weight_bridge, then constructs DiTRunner + VAERunner.
//   run()      — tokenizes/pads the prompt, runs the Euler flow-matching loop
//                with CFG, decodes the latent via the DAC VAE, returns PCM in
//                a TaskResult.audio_output.
//
// The Qwen3 text encoder sidecar is optional. When present it feeds the DiT
// cross-attention; when absent the session falls back to a zero context (the
// same fallback path used here), which still exercises the full DiT + VAE
// pipeline. A real TE is wired in a later phase.
class MossSfxV2Session final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    MossSfxV2Session(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const MossSfxV2Assets> assets);

    ~MossSfxV2Session() override;

    std::string family() const override { return "moss_sfx_v2"; }
    runtime::VoiceTaskKind task_kind() const override { return runtime::VoiceTaskKind::AudioGeneration; }
    runtime::RunMode run_mode() const override { return runtime::RunMode::Offline; }

    void prepare(const runtime::SessionPreparationRequest& request) override;
    runtime::TaskResult run(const runtime::TaskRequest& request) override;

private:
    // run_sfx implementation. Returns mono PCM at the
    // configured sample rate.
    bool run_sfx(const std::string& prompt,
                 const std::string& negative_prompt,
                 float duration_sec,
                 int num_steps,
                 float guidance_scale,
                 uint32_t seed,
                 std::vector<float>* pcm_out,
                 std::string* error);

    runtime::TaskSpec task_;
    std::shared_ptr<const MossSfxV2Assets> assets_;

    // Weight bridge storage — must outlive the runners (declared first so it
    // is destroyed last). Holds the no_alloc ggml_context + owned byte buffers
    // that DiTRunner/VAERunner read via ggml_get_tensor(ctx, name). Each runner
    // receives its own context: DiTRunner reads "moss_sfx_v2.*" (non-vae) from
    // dit_weights_->ctx; VAERunner reads "moss_sfx_v2.vae.*" from
    // vae_weights_->ctx.
    std::unique_ptr<LoadedWeights> dit_weights_;
    std::unique_ptr<LoadedWeights> vae_weights_;
    std::unique_ptr<LoadedWeights> te_weights_;

    std::unique_ptr<DiTRunner>         dit_runner_;
    std::unique_ptr<VAERunner>         vae_runner_;
    std::unique_ptr<TextEncoderRunner> text_encoder_;  // nullptr when no TE GGUF / tokenizer
};

}  // namespace engine::models::moss_sfx_v2
