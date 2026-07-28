// text_encoder.cpp — Qwen3-1.7B forward for MOSS-SoundEffect-v2.
//
// Raw ggml + ggml_gallocr on CUDA (the DiTRunner idiom). One graph build +
// compute per encode() call; the TE runs once per generation so rebuild cost
// is irrelevant. Output is the last-layer post-norm hidden states (matches
// HF Qwen3 `outputs.hidden_states[-1]`), zero-padded to 512 positions.

#include "engine/models/moss/moss_sfx_v2/text_encoder.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace engine::models::moss_sfx_v2 {

namespace {

// Shorthand: fetch a named weight tensor out of the (migrated) ext context.
inline ggml_tensor* W(ggml_context* ctx, const char* name) {
    return ggml_get_tensor(ctx, name);
}

// RMSNorm over ne[0] (hidden), then scale by the [hidden] weight vector.
// ggml_mul broadcasts the [hidden,1] view across all trailing dims. The norm
// weights ship as F16; activations are F32 (mul_mat output) — cast the weight
// up so ggml_mul sees matching types.
ggml_tensor* rms_norm_scaled(ggml_context* ctx, ggml_tensor* x,
                             ggml_tensor* w, float eps) {
    ggml_tensor* n = ggml_rms_norm(ctx, x, eps);
    ggml_tensor* wf = (w->type == GGML_TYPE_F32) ? w : ggml_cast(ctx, w, GGML_TYPE_F32);
    ggml_tensor* wb = ggml_reshape_2d(ctx, wf, wf->ne[0], 1);
    return ggml_mul(ctx, n, wb);
}

// Per-head Q/K RMSNorm over head_dim (Qwen3 applies QKNorm to each head's
// hd-vector independently). t: [hd, n_heads_or_kv, T] contiguous. Collapsing
// to [hd, n*T] preserves the per-(head,pos) hd vectors as rows for rms_norm,
// because ggml reshape reinterprets ne[] without moving data.
ggml_tensor* qk_norm(ggml_context* ctx, ggml_tensor* t, int hd, ggml_tensor* w) {
    const int64_t n_heads = t->ne[1];
    const int64_t T = t->ne[2];
    ggml_tensor* flat = ggml_reshape_2d(ctx, t, hd, n_heads * T);
    flat = ggml_rms_norm(ctx, flat, 1e-6f);
    ggml_tensor* wf = (w->type == GGML_TYPE_F32) ? w : ggml_cast(ctx, w, GGML_TYPE_F32);
    ggml_tensor* wb = ggml_reshape_2d(ctx, wf, hd, 1);
    flat = ggml_mul(ctx, flat, wb);
    return ggml_reshape_3d(ctx, flat, hd, n_heads, T);
}

// GQA broadcast: [hd, n_kv, T] -> [hd, n_heads, T] with each kv head repeated
// `rep` CONSECUTIVE times (kv0,kv0,kv1,kv1,...) so query head h maps to kv
// head h/rep. The reshape+repeat must put the REPEAT axis inner (faster) and
// the kv axis outer (slower): collapsing [rep, n_kv] into rep*n_kv yields flat
// head index r + rep*kv, i.e. consecutive heads share a kv head. (Putting kv
// inner instead gives kv + n_kv*r, which only agrees with the correct ordering
// at the boundary heads 0 and n_heads-1 — the exact "head 0 & 15 match, middle
// scrambled" parity signature this fix resolves.)
ggml_tensor* gqa_repeat(ggml_context* ctx, ggml_tensor* t, int rep) {
    const int64_t hd   = t->ne[0];
    const int64_t n_kv = t->ne[1];
    const int64_t T    = t->ne[2];
    ggml_tensor* t4  = ggml_reshape_4d(ctx, t, hd, 1, n_kv, T);      // [hd, 1, n_kv, T]
    ggml_tensor* tgt = ggml_new_tensor_4d(ctx, t->type, hd, rep, n_kv, T);
    t4 = ggml_repeat(ctx, t4, tgt);                                   // [hd, rep, n_kv, T]
    return ggml_reshape_3d(ctx, t4, hd, rep * n_kv, T);               // [hd, rep*n_kv, T]
}

}  // namespace

TextEncoderRunner::TextEncoderRunner(
    ggml_context* te_ctx, const TextEncoderConfig& cfg,
    std::shared_ptr<tokenizers::LlamaBpeTokenizer> tokenizer)
    : ext_ctx_(te_ctx), cfg_(cfg), tokenizer_(std::move(tokenizer)) {}

TextEncoderRunner::~TextEncoderRunner() {
    if (migrated_buf_) ggml_backend_buffer_free(migrated_buf_);
}

bool TextEncoderRunner::ensure_backend(std::string* error) {
    if (backend_initialized_) return true;

    bp_ = std::make_unique<ggml_utils::BackendPair>(ggml_utils::backend_init("MSE2-TE"));
    if (!bp_ || !bp_->backend) {
        if (error) *error = "TE: no backend available";
        return false;
    }
    cuda_backend_ = bp_->backend;

    if (bp_->has_gpu) {
        migrated_buf_ = ggml_utils::migrate_ctx_to_backend(ext_ctx_, cuda_backend_, "MSE2-TE");
        if (!migrated_buf_) {
            if (error) *error = "TE: failed to migrate weights to GPU";
            return false;
        }
    }

    backend_initialized_ = true;
    return true;
}

int TextEncoderRunner::encode(const std::string& text, float* out, std::string* error) {
    if (!ext_ctx_) { if (error) *error = "TE: no weight context"; return -1; }
    if (!tokenizer_) { if (error) *error = "TE: no tokenizer"; return -1; }

    // ── Tokenize (Qwen3 BPE; add_special_tokens has no effect for Qwen —
    //    no BOS, EOS not appended for plain text, matching the reference ids).
    std::vector<int32_t> ids = tokenizer_->encode(text);
    if (ids.empty()) {
        // Empty prompt -> n_real=0 -> all-zero embedding (matches Python
        // WanPrompter where prompt_emb[i, 0:] = 0 for the empty/uncond branch).
        std::memset(out, 0, sizeof(float) * cfg_.max_seq * cfg_.hidden);
        return 0;
    }
    if (static_cast<int>(ids.size()) > cfg_.max_seq) {
        ids.resize(cfg_.max_seq);  // truncate (WanPrompter truncation=True)
    }
    const int T = static_cast<int>(ids.size());
    if (std::getenv("AUDIOCORE_TE_DEBUG")) {
        std::fprintf(stderr, "[TE-dbg] token_ids (%d):", T);
        for (int i = 0; i < T; ++i) std::fprintf(stderr, " %d", ids[static_cast<size_t>(i)]);
        std::fprintf(stderr, "\n");
    }

    if (!ensure_backend(error)) return -1;

    const int hd = cfg_.head_dim;
    const int nh = cfg_.n_heads;
    const int nkv = cfg_.n_kv_heads;
    const int rep = nh / nkv;  // GQA group size
    const int hidden = cfg_.hidden;

    // ── Build graph in a scratch (no_alloc) context ──
    // Only tensor/cgraph metadata lives here; ggml_gallocr places data on CUDA.
    constexpr size_t kArenaBytes = 8 * 1024 * 1024;
    std::vector<uint8_t> arena(kArenaBytes);
    struct ggml_init_params gparams;
    gparams.mem_size    = kArenaBytes;
    gparams.mem_buffer  = arena.data();
    gparams.no_alloc    = true;
    ggml_context* ctx = ggml_init(gparams);
    if (!ctx) { if (error) *error = "TE: ggml_init failed"; return -1; }

    // Inputs (data uploaded after gallocr assigns their buffers).
    ggml_tensor* ids_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_tensor* pos_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);


    // Token embedding: token_embd.weight [hidden, vocab] -> [hidden, T].
    // Cast to F32 up front: the projection mul_mats output F32 and the residual
    // adds require matching types, so we keep the whole activation stream in F32
    // (mul_mat handles F16 weights × F32 activations natively).
    ggml_tensor* x = ggml_get_rows(ctx, W(ext_ctx_, "token_embd.weight"), ids_t);
    x = ggml_cast(ctx, x, GGML_TYPE_F32);

    char nm[80];
    for (int layer = 0; layer < cfg_.n_layers; ++layer) {
        ggml_tensor* h_in = x;

        // ── Attention ──
        std::snprintf(nm, sizeof(nm), "blk.%d.attn_norm.weight", layer);
        ggml_tensor* hn = rms_norm_scaled(ctx, h_in, W(ext_ctx_, nm), cfg_.rms_eps);

        std::snprintf(nm, sizeof(nm), "blk.%d.attn_q.weight", layer);
        ggml_tensor* q = ggml_mul_mat(ctx, W(ext_ctx_, nm), hn);   // [hidden, T]
        std::snprintf(nm, sizeof(nm), "blk.%d.attn_k.weight", layer);
        ggml_tensor* k = ggml_mul_mat(ctx, W(ext_ctx_, nm), hn);   // [nkv*hd, T]
        std::snprintf(nm, sizeof(nm), "blk.%d.attn_v.weight", layer);
        ggml_tensor* v = ggml_mul_mat(ctx, W(ext_ctx_, nm), hn);   // [nkv*hd, T]

        q = ggml_reshape_3d(ctx, q, hd, nh, T);
        k = ggml_reshape_3d(ctx, k, hd, nkv, T);
        v = ggml_reshape_3d(ctx, v, hd, nkv, T);

        // Qwen3 per-head QK-norm (over head_dim), applied before RoPE.
        std::snprintf(nm, sizeof(nm), "blk.%d.attn_q_norm.weight", layer);
        q = qk_norm(ctx, q, hd, W(ext_ctx_, nm));
        std::snprintf(nm, sizeof(nm), "blk.%d.attn_k_norm.weight", layer);
        k = qk_norm(ctx, k, hd, W(ext_ctx_, nm));

        // GQA: expand kv heads to match query heads.
        k = gqa_repeat(ctx, k, rep);
        v = gqa_repeat(ctx, v, rep);

        // NeoX RoPE (mode 2), n_dims=hd, freq_base=1e6.
        q = ggml_rope_ext(ctx, q, pos_t, nullptr,
                          hd, /*mode=*/2, /*n_ctx_orig=*/0,
                          cfg_.rope_base, /*freq_scale=*/1.0f,
                          /*ext_factor=*/0.0f, /*attn_factor=*/1.0f,
                          /*beta_fast=*/0.0f, /*beta_slow=*/0.0f);
        k = ggml_rope_ext(ctx, k, pos_t, nullptr,
                          hd, 2, 0, cfg_.rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // Manual causal attention, batched over heads via ne[2].
        // q,k permuted to [hd, T, nh];  v permuted to [T, hd, nh].
        q = ggml_permute(ctx, q, 0, 2, 1, 3);   // [hd, nh, T] -> [hd, T, nh]
        k = ggml_permute(ctx, k, 0, 2, 1, 3);   // [hd, T, nh]
        // v: ggml_permute(t, a0,a1,a2,a3) maps source axis i -> result
        // position a_i, so [hd,nh,T] -> [T,hd,nh] is (1,2,0,3). That view has
        // nb[0] > nb[1] (ggml flags it "transposed"); cont() fixes the strides.
        v = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));   // [hd,nh,T] -> [T,hd,nh]

        ggml_tensor* kq = ggml_mul_mat(ctx, k, q);              // [T, T, nh]
        kq = ggml_scale(ctx, kq, 1.0f / std::sqrt(static_cast<float>(hd)));
        kq = ggml_diag_mask_inf(ctx, kq, 0);                    // causal (key>query masked)
        kq = ggml_soft_max(ctx, kq);
        ggml_tensor* kqv = ggml_mul_mat(ctx, v, kq);            // [hd, T, nh]

        // Back to [hidden, T] = [hd*nh, T] for the output projection.
        ggml_tensor* attn = ggml_permute(ctx, kqv, 0, 2, 1, 3);  // [hd, nh, T]
        attn = ggml_cont(ctx, attn);
        attn = ggml_reshape_2d(ctx, attn, hd * nh, T);           // [hidden, T]

        std::snprintf(nm, sizeof(nm), "blk.%d.attn_output.weight", layer);
        attn = ggml_mul_mat(ctx, W(ext_ctx_, nm), attn);         // [hidden, T]
        x = ggml_add(ctx, h_in, attn);                           // residual

        // ── SwiGLU FFN: down(silu(gate(x_norm)) * up(x_norm)) ──
        ggml_tensor* ffn_in = x;
        std::snprintf(nm, sizeof(nm), "blk.%d.ffn_norm.weight", layer);
        ggml_tensor* fn = rms_norm_scaled(ctx, ffn_in, W(ext_ctx_, nm), cfg_.rms_eps);
        std::snprintf(nm, sizeof(nm), "blk.%d.ffn_gate.weight", layer);
        ggml_tensor* gate = ggml_mul_mat(ctx, W(ext_ctx_, nm), fn);
        std::snprintf(nm, sizeof(nm), "blk.%d.ffn_up.weight", layer);
        ggml_tensor* up = ggml_mul_mat(ctx, W(ext_ctx_, nm), fn);
        ggml_tensor* act = ggml_mul(ctx, ggml_silu(ctx, gate), up);
        std::snprintf(nm, sizeof(nm), "blk.%d.ffn_down.weight", layer);
        ggml_tensor* ffn_out = ggml_mul_mat(ctx, W(ext_ctx_, nm), act);
        x = ggml_add(ctx, ffn_in, ffn_out);
    }  // for layer

    // Qwen3Model.forward applies self.norm(hidden_states) AFTER the decoder
    // loop, and HF's output_hidden_states captures the result. So
    // outputs.hidden_states[-1] is the NORMED last-layer output, not raw.
    // MossSoundEffectPipeline returns this normed output as cond_emb.
    // Verified empirically: model.norm(raw_l27_out) == hidden_states[-1]
    // (cos 1.000000), and matches the Python reference cond_emb.f32.
    ggml_tensor* out_t = rms_norm_scaled(ctx, x,
        W(ext_ctx_, "output_norm.weight"), cfg_.rms_eps);
    ggml_set_output(out_t);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out_t);

    // Allocate on the CUDA backend (lifecycle-based; weights already migrated).
    ggml_gallocr_t gallocr = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(cuda_backend_));
    if (!gallocr || !ggml_gallocr_alloc_graph(gallocr, gf)) {
        if (error) *error = "TE: gallocr_alloc_graph failed";
        if (gallocr) ggml_gallocr_free(gallocr);
        ggml_free(ctx);
        return -1;
    }

    // Upload inputs. gallocr may skip allocating leaf tensors for very small
    // graphs (<~3 nodes). Pre-allocate them explicitly on the CUDA backend so
    // they always have a buffer, regardless of graph size.
    if (!ids_t->buffer) {
        ggml_backend_buffer_t ibuf = ggml_backend_alloc_buffer(cuda_backend_,
            static_cast<size_t>(T) * sizeof(int32_t));
        void* base = ggml_backend_buffer_get_base(ibuf);
        ggml_backend_tensor_alloc(ibuf, ids_t, base);
    }
    ggml_backend_tensor_set(ids_t, ids.data(), 0,
                            static_cast<size_t>(T) * sizeof(int32_t));
    std::vector<int32_t> pos(static_cast<size_t>(T));
    if (!pos_t->buffer) {
        ggml_backend_buffer_t pbuf = ggml_backend_alloc_buffer(cuda_backend_,
            static_cast<size_t>(T) * sizeof(int32_t));
        void* base = ggml_backend_buffer_get_base(pbuf);
        ggml_backend_tensor_alloc(pbuf, pos_t, base);
    }
    for (int i = 0; i < T; ++i) pos[static_cast<size_t>(i)] = i;
    ggml_backend_tensor_set(pos_t, pos.data(), 0,
                            static_cast<size_t>(T) * sizeof(int32_t));

    // Compute.
    const auto t0 = ggml_time_us();
    const int rc = ggml_backend_graph_compute(cuda_backend_, gf);
    const auto t1 = ggml_time_us();
    std::fprintf(stderr, "[moss_sfx_v2] TE fwd %d tokens %.1f ms\n",
                 T, static_cast<double>(t1 - t0) / 1000.0);
    if (rc != 0) {
        if (error) *error = "TE: graph_compute failed";
        ggml_gallocr_free(gallocr);
        ggml_free(ctx);
        return -1;
    }

    // Read [hidden, T] (ne0=hidden contiguous == PyTorch [T, hidden] row-major).
    // Zero-pad the tail (positions >= T) to max_seq and memcpy the real block.
    std::memset(out, 0, sizeof(float) * cfg_.max_seq * cfg_.hidden);
    ggml_backend_tensor_get(out_t, out, 0,
                            static_cast<size_t>(hidden) * static_cast<size_t>(T) * sizeof(float));

    ggml_gallocr_free(gallocr);
    ggml_free(ctx);
    return T;
}

}  // namespace engine::models::moss_sfx_v2
