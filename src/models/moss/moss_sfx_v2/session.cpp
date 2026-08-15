// session.cpp — MOSS-SoundEffect-v2 offline generation session.
//
// SfxSession run_sfx flow-matching loop on audiocpp's RuntimeSessionBase. The
// DiT + VAE graph logic is verbatim from the reference implementation; only the
// request/response plumbing + weight front-end differ.

#include "engine/models/moss/moss_sfx_v2/session.h"

#include "engine/framework/runtime/options.h"
#include "engine/framework/tokenizers/llama_bpe.h"

#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::moss_sfx_v2 {

namespace runtime = engine::runtime;
namespace fs = std::filesystem;

// Build the Qwen3 BPE tokenizer for the text encoder from the HF tokenizer
// directory discovered by assets.cpp (vocab.json + merges.txt +
// tokenizer_config.json). Returns nullptr if the files are not present.
std::shared_ptr<tokenizers::LlamaBpeTokenizer> build_te_tokenizer(
    const fs::path& tokenizer_dir, const TextEncoderConfig& /*te_cfg*/) {
    if (tokenizer_dir.empty()) return nullptr;
    auto exists = [](const fs::path& p) {
        return !p.empty() && fs::exists(p) && fs::is_regular_file(p);
    };
    fs::path vocab = tokenizer_dir / "vocab.json";
    fs::path merges = tokenizer_dir / "merges.txt";
    fs::path tcfg = tokenizer_dir / "tokenizer_config.json";
    if (!exists(vocab) || !exists(merges) || !exists(tcfg)) return nullptr;

    tokenizers::LlamaBpeTokenizerSpec spec;
    spec.vocab_path            = vocab;
    spec.merges_path           = merges;
    spec.tokenizer_config_path = tcfg;
    spec.pre_type              = tokenizers::LlamaBpePreTokenizer::Qwen2;
    // tokenizer.json optional — the Qwen2 pre-tokenizer reconstructs from
    // vocab+merges+config here (matches qwen3_tts/tokenizer_text.cpp).
    try {
        return tokenizers::load_llama_bpe_tokenizer(spec);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[moss_sfx_v2] TE tokenizer load failed: %s\n", e.what());
        return nullptr;
    }
}

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

    // Optional Qwen3 text-encoder sidecar. Built only when the TE GGUF is
    // present AND the HF tokenizer files (vocab/merges/tokenizer_config) are
    // discoverable. When absent, run_sfx falls back to a zero context — the
    // same path a missing/broken TE effectively exercises.
    if (assets_->te_weights) {
        te_weights_ = load_weights_into_context(
            *assets_->te_weights, /*prefix=*/"",
            /*capacity_tensors=*/1024, &err);
        if (!te_weights_) {
            std::fprintf(stderr,
                "[moss_sfx_v2] TE weight bridge failed (falling back to zero context): %s\n",
                err.c_str());
        } else {
            std::shared_ptr<tokenizers::LlamaBpeTokenizer> tokenizer =
                build_te_tokenizer(assets_->tokenizer_dir, cfg.te);
            if (tokenizer) {
                text_encoder_ = std::make_unique<TextEncoderRunner>(
                    te_weights_->ctx, cfg.te, std::move(tokenizer));
                std::fprintf(stderr,
                    "[moss_sfx_v2] TE enabled (Qwen3: %d layers, hidden=%d, heads=%d/%d kv, hd=%d)\n",
                    cfg.te.n_layers, cfg.te.hidden, cfg.te.n_heads, cfg.te.n_kv_heads, cfg.te.head_dim);
            } else {
                std::fprintf(stderr,
                    "[moss_sfx_v2] TE tokenizer files not found at \"%s\" (zero-context fallback)\n",
                    assets_->tokenizer_dir.string().c_str());
            }
        }
    }

    mark_prepared();
}

// ═══════════════════════════════════════════════════════════════════════════
//  Flow-matching Euler schedule (build_schedule)
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
//  run_sfx — the denoising pipeline
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
    //
    // MossSoundEffectPipeline appends a duration suffix before encoding
    // (append_duration_suffix=True): f"{prompt} duration: {seconds:.1f}s".
    // The unconditional (empty negative_prompt) branch yields n_real=0, i.e.
    // an all-zero embedding — so uncond is always zero by construction.
    constexpr int32_t kTextLen = 512;
    const int32_t T_cond = kTextLen;
    const int32_t T_uncond = kTextLen;
    std::vector<float> cond_emb(static_cast<size_t>(kTextLen) * dc.text_dim, 0.0f);
    std::vector<float> uncond_emb(static_cast<size_t>(kTextLen) * dc.text_dim, 0.0f);

    const float dur_rounded = std::round(duration_sec * 10.0f) / 10.0f;
    char dur_suffix[48];
    std::snprintf(dur_suffix, sizeof(dur_suffix), " duration: %.1fs", dur_rounded);
    const std::string cond_prompt = prompt + dur_suffix;

    if (text_encoder_) {
        int n_real = text_encoder_->encode(cond_prompt, cond_emb.data(), error);
        if (n_real < 0) {
            if (error) *error = std::string("MOSS-SFX-v2: TE encode failed: ") + *error;
            return false;
        }
        std::fprintf(stderr,
            "[moss_sfx_v2] TE: cond prompt=\"%s\" n_real=%d/%d\n",
            cond_prompt.c_str(), n_real, kTextLen);
    } else {
        std::fprintf(stderr,
            "[moss_sfx_v2] TE: no-TE path (zero context, T=%d) prompt=\"%s\"\n",
            T_cond, cond_prompt.c_str());
    }

    // Debug hook: dump the cond embedding and exit before the DiT loop, for
    // byte-level parity verification against the Python reference
    // (encprompt_artifacts/cond_emb.f32). Opt-in via AUDIOCORE_DUMP_TE_EMB.
    if (const char* dp = std::getenv("AUDIOCORE_DUMP_TE_EMB")) {
        const std::string path = std::string(dp);
        std::FILE* f = std::fopen(path.c_str(), "wb");
        if (f) {
            std::fwrite(cond_emb.data(), sizeof(float), cond_emb.size(), f);
            std::fclose(f);
            std::fprintf(stderr,
                "[moss_sfx_v2] dumped cond_emb [%zu floats] -> %s\n",
                cond_emb.size(), path.c_str());
        }
        if (error) *error = "TE-only dump completed";
        return false;  // early exit (skip DiT/VAE)
    }

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
    constexpr int32_t kVaeChunkLatent = 128;  // VAE decode chunk (latent frames). With the
                                            // gallocr allocator, peak VRAM is liveness-based
                                            // (chunk size no longer drives OOM), so this only
                                            // trades graph-build count vs per-chunk latency:
                                            // 12 chunks per 30s latent. 500 works too; 128 is
                                            // the value proven with gallocr (2026-08-06).
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
