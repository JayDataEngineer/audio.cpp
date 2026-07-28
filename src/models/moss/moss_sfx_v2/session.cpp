// session.cpp — MOSS-SoundEffect-v2 offline generation session.
//
// Ports audiocore's SfxSession (run_sfx flow-matching loop) onto audiocpp's
// RuntimeSessionBase. The DiT + VAE graph logic is unchanged (verbatim from
// audiocore); only the request/response plumbing + weight front-end differ.

#include "engine/models/moss/moss_sfx_v2/session.h"

#include "engine/framework/runtime/options.h"

#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::moss_sfx_v2 {

namespace runtime = engine::runtime;

// ═══════════════════════════════════════════════════════════════════════════
//  Construction / teardown
// ═══════════════════════════════════════════════════════════════════════════

MossSfxV2Session::MossSfxV2Session(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const MossSfxV2Assets> assets)
    : runtime::RuntimeSessionBase(options),
      task_(std::move(task)),
      assets_(std::move(assets)) {
    if (!assets_) {
        throw std::runtime_error("MOSS-SFX-v2: assets are null");
    }
}

MossSfxV2Session::~MossSfxV2Session() {
    dit_runner_.reset();
    vae_runner_.reset();
    // LoadedWeights freed after the runners that reference their contexts.
}

// ═══════════════════════════════════════════════════════════════════════════
//  prepare — bridge weights, build runners
// ═══════════════════════════════════════════════════════════════════════════

void MossSfxV2Session::prepare(const runtime::SessionPreparationRequest& /*request*/) {
    if (dit_runner_ && vae_runner_) {
        mark_prepared();
        return;
    }
    const auto& cfg = assets_->config;
    std::string err;

    // DiT weights → no_alloc ggml_context
    dit_weights_ = load_weights_into_context(
        *assets_->dit_weights, /*prefix=*/"moss_sfx_v2",
        /*capacity_tensors=*/1024, &err);
    if (!dit_weights_) {
        throw std::runtime_error("MOSS-SFX-v2: DiT weight bridge failed: " + err);
    }
    ggml_context* dit_ctx = dit_weights_->ctx;

    // VAE weights → separate context. If the VAE GGUF is absent, the runner
    // will report a decode failure at run time; we still let prepare succeed.
    ggml_context* vae_ctx = nullptr;
    if (assets_->vae_weights) {
        vae_weights_ = load_weights_into_context(
            *assets_->vae_weights, /*prefix=*/"moss_sfx_v2.vae",
            /*capacity_tensors=*/512, &err);
        if (!vae_weights_) {
            throw std::runtime_error("MOSS-SFX-v2: VAE weight bridge failed: " + err);
        }
        vae_ctx = vae_weights_->ctx;
    }

    std::fprintf(stderr, "[moss_sfx_v2] constructing runners (dim=%d n_layers=%d, "
                 "vae decoder_dim=%d hop=%d)\n",
                 cfg.dit.dim, cfg.dit.n_layers, cfg.vae.decoder_dim, cfg.vae.hop_length);

    dit_runner_ = std::make_unique<DiTRunner>(dit_ctx, cfg.dit);
    if (vae_ctx) {
        vae_runner_ = std::make_unique<VAERunner>(vae_ctx, cfg.vae);
    }

    mark_prepared();
}

// ═══════════════════════════════════════════════════════════════════════════
//  Flow-matching Euler schedule (ported from audiocore build_schedule)
// ═══════════════════════════════════════════════════════════════════════════

struct FlowSchedule {
    std::vector<float> sigmas;
    int n_steps;
};

static FlowSchedule build_schedule(int num_steps, float shift,
                                   float sigma_min, bool extra_one_step) {
    FlowSchedule s;
    s.n_steps = num_steps;
    s.sigmas.resize(static_cast<size_t>(num_steps));
    const float sigma_start = 1.0f;
    if (extra_one_step) {
        // torch.linspace(sigma_start, sigma_min, n+1)[:-1]
        for (int i = 0; i < num_steps; ++i) {
            s.sigmas[static_cast<size_t>(i)] = sigma_start +
                (sigma_min - sigma_start) * static_cast<float>(i) /
                static_cast<float>(num_steps);
        }
    } else {
        for (int i = 0; i < num_steps; ++i) {
            const float f = (num_steps > 1)
                ? static_cast<float>(i) / static_cast<float>(num_steps - 1)
                : 0.0f;
            s.sigmas[static_cast<size_t>(i)] = sigma_start + (sigma_min - sigma_start) * f;
        }
    }
    if (std::abs(shift - 1.0f) > 1e-6f) {
        for (int i = 0; i < num_steps; ++i) {
            const float sig = s.sigmas[static_cast<size_t>(i)];
            s.sigmas[static_cast<size_t>(i)] =
                shift * sig / (1.0f + (shift - 1.0f) * sig);
        }
    }
    return s;
}

// ═══════════════════════════════════════════════════════════════════════════
//  run_sfx — the denoising pipeline (ported from audiocore session.cpp)
// ═══════════════════════════════════════════════════════════════════════════

bool MossSfxV2Session::run_sfx(
    const std::string& prompt, const std::string& negative_prompt,
    float duration_sec, int num_steps, float guidance_scale, uint32_t seed,
    std::vector<float>* pcm_out, std::string* error) {

    const auto& cfg = assets_->config;
    const DitConfig& dc = cfg.dit;

    if (!dit_runner_ || !vae_runner_) {
        if (error) *error = "MOSS-SFX-v2: runners not prepared";
        return false;
    }

    // ── 1. Text encoder → context embeddings ─────────────────────────────
    // CRITICAL: WanPrompter always pads the tokenized prompt to 512 positions.
    // The DiT cross-attention expects 512 KV pairs. With only the real token
    // count the velocity is ~200× too large, causing trajectory divergence.
    // No-TE path (Phase 1): zero context padded to 512 — matches audiocore's
    // no-TE fallback. A real Qwen3 TE is wired in a later phase.
    constexpr int32_t kTextLen = 512;
    const int32_t T_cond = kTextLen;
    const int32_t T_uncond = kTextLen;
    std::vector<float> cond_emb(static_cast<size_t>(kTextLen) * dc.text_dim, 0.0f);
    std::vector<float> uncond_emb(static_cast<size_t>(kTextLen) * dc.text_dim, 0.0f);
    std::fprintf(stderr,
        "[moss_sfx_v2] TE: no-TE path (zero context, T=%d) prompt=\"%s\"\n",
        T_cond, prompt.c_str());

    // ── 2. Output duration → T_latent ────────────────────────────────────
    // Match Python: always denoise the max_inference_seconds=30 latent, then
    // crop. The DiT was trained on 30s segments.
    const int up = cfg.vae.hop_length;   // 960
    constexpr int32_t kMaxInferenceSeconds = 30;
    const int32_t T_latent =
        (cfg.sample_rate * kMaxInferenceSeconds) / up;
    const int32_t num_samples = static_cast<int32_t>(duration_sec * cfg.sample_rate);

    std::fprintf(stderr,
        "[moss_sfx_v2] dur=%.1fs samples=%d T_latent=%d (full=%ds)\n",
        duration_sec, num_samples, T_latent, kMaxInferenceSeconds);

    // ── 3. Noise latent ∼ N(0, I)  [T_latent, in_dim] ────────────────────
    std::mt19937_64 rng(static_cast<uint64_t>(seed != 0 ? seed : 42));
    std::normal_distribution<float> ndist(0.0f, 1.0f);
    std::vector<float> x_t(static_cast<size_t>(T_latent) * dc.in_dim);
    for (auto& v : x_t) v = ndist(rng);

    // ── 4. Flow-matching schedule ────────────────────────────────────────
    const int steps = (num_steps > 0) ? num_steps : 50;
    const float gs = (guidance_scale > 0.0f) ? guidance_scale : 5.0f;
    FlowSchedule sched = build_schedule(
        steps,
        static_cast<float>(cfg.scheduler.shift),
        cfg.scheduler.sigma_min,
        cfg.scheduler.extra_one_step);

    // ── 5. Velocity buffer + denoising loop ──────────────────────────────
    std::vector<float> v_out(static_cast<size_t>(T_latent) * dc.out_dim);
    auto t0 = ggml_time_us();
    double t_fwd_us = 0;
    for (int step = 0; step < sched.n_steps; ++step) {
        const float sigma = sched.sigmas[static_cast<size_t>(step)];
        const float sigma_next = (step + 1 < sched.n_steps)
            ? sched.sigmas[static_cast<size_t>(step + 1)]
            : 0.0f;
        const float dt = sigma_next - sigma;

        const auto f0 = ggml_time_us();
        if (!dit_runner_->forward_cfg(
                x_t.data(), &sigma,
                cond_emb.data(), T_cond,
                uncond_emb.data(), T_uncond,
                gs,
                /*B=*/1, T_latent,
                v_out.data(), error)) {
            return false;
        }
        const double fwd_us = static_cast<double>(ggml_time_us() - f0);
        t_fwd_us += fwd_us;

        for (size_t i = 0; i < x_t.size(); ++i) {
            x_t[i] += dt * v_out[i];
        }

        if (step % 10 == 0 || step == sched.n_steps - 1) {
            double v_sq = 0.0, x_sq = 0.0;
            for (size_t i = 0; i < x_t.size(); ++i) {
                v_sq += static_cast<double>(v_out[i]) * v_out[i];
                x_sq += static_cast<double>(x_t[i]) * x_t[i];
            }
            std::fprintf(stderr,
                "[moss_sfx_v2] step %d/%d σ=%.4f v_rms=%.2f x_rms=%.2f fwd=%.2fs\n",
                step, sched.n_steps, sigma,
                std::sqrt(v_sq / x_t.size()),
                std::sqrt(x_sq / x_t.size()),
                fwd_us / 1e6);
        }
    }
    const double t_tot = static_cast<double>(ggml_time_us() - t0) / 1e6;
    std::fprintf(stderr,
        "[moss_sfx_v2] loop: %.2fs total, %.2fs in forward (%.1f%%) avg %.3fs/step\n",
        t_tot, t_fwd_us / 1e6, 100.0 * t_fwd_us / (t_tot * 1e6),
        t_tot / sched.n_steps);

    // ── 6. VAE decode → PCM mono ─────────────────────────────────────────
    const int32_t T_audio = T_latent * up;
    pcm_out->resize(static_cast<size_t>(T_audio));
    constexpr int32_t kVaeChunkLatent = 500;  // ≤10s/chunk fits 24 GB VRAM
    {
        auto v0 = std::chrono::steady_clock::now();
        const int n_chunks =
            (T_latent + kVaeChunkLatent - 1) / kVaeChunkLatent;
        std::fprintf(stderr,
            "[moss_sfx_v2] VAE decode: %d chunk(s) of ≤%d latent frames\n",
            n_chunks, kVaeChunkLatent);
        for (int c = 0; c < n_chunks; ++c) {
            const int t_start = c * kVaeChunkLatent;
            const int t_end = std::min(t_start + kVaeChunkLatent,
                                       static_cast<int>(T_latent));
            const int chunk_T = t_end - t_start;
            std::vector<float> chunk_in(
                static_cast<size_t>(chunk_T) * dc.in_dim);
            for (int t = 0; t < chunk_T; ++t) {
                std::memcpy(
                    chunk_in.data() + static_cast<size_t>(t) * dc.in_dim,
                    x_t.data() + static_cast<size_t>(t_start + t) * dc.in_dim,
                    static_cast<size_t>(dc.in_dim) * sizeof(float));
            }
            std::vector<float> chunk_out(
                static_cast<size_t>(chunk_T) * up);
            if (!vae_runner_->decode(chunk_in.data(), /*B=*/1, chunk_T,
                                      chunk_out.data(), error)) {
                return false;
            }
            std::memcpy(
                pcm_out->data() + static_cast<size_t>(t_start) * up,
                chunk_out.data(),
                static_cast<size_t>(chunk_T) * up * sizeof(float));
            std::fprintf(stderr,
                "[moss_sfx_v2]   chunk %d/%d: T=%d → %d audio\n",
                c + 1, n_chunks, chunk_T, chunk_T * up);
        }
        auto v1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(v1 - v0).count();
        std::fprintf(stderr,
            "[moss_sfx_v2] VAE decode total: %.0fms (%d latent → %d audio)\n",
            ms, T_latent, T_audio);
    }

    if (static_cast<int32_t>(pcm_out->size()) > num_samples) {
        pcm_out->resize(static_cast<size_t>(num_samples));
    }

    // ── 7. Post-processing: DC removal + gentle normalization ────────────
    {
        auto& pcm = *pcm_out;
        const size_t n = pcm.size();
        if (n > 0) {
            double sum = 0.0;
            float peak = 0.0f;
            for (size_t i = 0; i < n; ++i) sum += static_cast<double>(pcm[i]);
            const float dc = static_cast<float>(sum / static_cast<double>(n));
            for (size_t i = 0; i < n; ++i) {
                pcm[i] -= dc;
                peak = std::max(peak, std::fabs(pcm[i]));
            }
            if (peak > 1e-6f && peak < 0.5f) {
                const float gain = 0.9f / peak;
                for (size_t i = 0; i < n; ++i) pcm[i] *= gain;
            }
            std::fprintf(stderr,
                "[moss_sfx_v2] post: DC=%.5f peak=%.5f %s\n",
                dc, peak, (peak > 1e-6f && peak < 0.5f) ? "normalized" : "");
        }
    }

    std::fprintf(stderr,
        "[moss_sfx_v2] done: %zu samples (%.2fs @ %dHz)\n",
        pcm_out->size(),
        static_cast<double>(pcm_out->size()) / cfg.sample_rate,
        cfg.sample_rate);
    (void)negative_prompt;  // reserved for the real-TE CFG path
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
//  run — parse request → run_sfx → TaskResult
// ═══════════════════════════════════════════════════════════════════════════

runtime::TaskResult MossSfxV2Session::run(const runtime::TaskRequest& request) {
    require_prepared("MOSS-SFX-v2 run");

    // Prompt text: text_input first, else "prompt"/"text" option.
    std::string prompt;
    if (request.text_input && !request.text_input->text.empty()) {
        prompt = request.text_input->text;
    } else {
        const auto p = runtime::find_option(request.options, {"prompt", "text"});
        if (p) prompt = *p;
    }
    const auto negative_prompt =
        runtime::find_option(request.options, {"negative_prompt", "negative"}).value_or("");

    const float duration_sec =
        runtime::parse_finite_float_option(request.options, {"duration_sec", "duration_seconds"})
            .value_or(10.0f);
    const int num_steps =
        runtime::parse_int_option(request.options, {"num_inference_steps", "steps"}).value_or(50);
    const float guidance_scale =
        runtime::parse_finite_float_option(request.options, {"guidance_scale"}).value_or(5.0f);
    const uint32_t seed =
        runtime::parse_u32_option(request.options, {"seed"}).value_or(42);

    if (duration_sec <= 0.0f) {
        throw std::runtime_error("MOSS-SFX-v2: duration_sec must be positive");
    }

    std::vector<float> pcm;
    std::string err;
    if (!run_sfx(prompt, negative_prompt, duration_sec, num_steps,
                 guidance_scale, seed, &pcm, &err)) {
        throw std::runtime_error("MOSS-SFX-v2 generation failed: " + err);
    }

    runtime::TaskResult out;
    runtime::AudioBuffer audio;
    audio.sample_rate = assets_->config.sample_rate;
    audio.channels = 1;
    audio.samples = std::move(pcm);
    out.audio_output = std::move(audio);
    return out;
}

}  // namespace engine::models::moss_sfx_v2
