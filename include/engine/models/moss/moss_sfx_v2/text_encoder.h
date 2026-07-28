// text_encoder.h — Qwen3 text encoder for MOSS-SoundEffect-v2.
//
// The DiT (WanAudioModel) cross-attends over Qwen3-1.7B last-layer hidden
// states. WanPrompter.encode_prompt pads the tokenized prompt to 512 positions,
// runs the TE, and zeros the tail beyond the real token count (seq_lens). The
// DiT then projects the [512, 2048] context through its own text_embedding
// (Linear 2048->1536 + GELU(tanh) + Linear 1536->1536).
//
// This runner ports the Qwen3 forward (token embedding -> 28 x [RMSNorm ->
// GQA + per-head QK-norm + NeoX RoPE + manual causal attention + RMSNorm ->
// SwiGLU FFN] -> final RMSNorm) onto raw ggml + ggml_gallocr on CUDA, mirroring
// DiTRunner's backend idiom. The earlier llama.cpp qwen3::Runner path crashed
// in ggml_cuda_op_mul (binbcast.cu stride assert) — so
// this is a from-scratch ggml port validated against the upstream Python
// (MOSS-TTS/moss_soundeffect_v2/diffsynth/models/qwen3_text_encoder.py).

#ifndef ENGINE_MODELS_MOSS_SFX_V2_TEXT_ENCODER_H
#define ENGINE_MODELS_MOSS_SFX_V2_TEXT_ENCODER_H

#include "engine/models/moss/moss_sfx_v2/backend_helper.h"
#include "engine/framework/tokenizers/llama_bpe.h"

#include <cstdint>
#include <memory>
#include <string>

struct ggml_context;
struct ggml_tensor;
typedef struct ggml_backend * ggml_backend_t;
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;

namespace engine::models::moss_sfx_v2 {

// Qwen3-1.7B config. Defaults match the TE GGUF KV (qwen3.*); assets.cpp
// overrides from the file so this stays in sync with the actual weights.
struct TextEncoderConfig {
    int32_t n_layers   = 28;
    int32_t hidden     = 2048;
    int32_t n_heads    = 16;
    int32_t n_kv_heads = 8;
    int32_t head_dim   = 128;       // == rope.dimension_count == hidden / n_heads
    int32_t ffn_dim    = 6144;
    int32_t vocab_size = 151936;
    float   rms_eps    = 1e-6f;
    float   rope_base  = 1.0e6f;
    int32_t max_seq    = 512;       // WanPrompter pads to text_len = 512
};

// Runs the Qwen3 text encoder on CUDA. encode() returns the last-layer
// hidden states written into `out` as [max_seq, hidden] f32 with the tail
// (positions >= n_real) zeroed — byte-identical to WanPrompter.encode_prompt's
// PyTorch [1, 512, 2048] output (ggml ne0=hidden-contiguous == PyTorch
// row-major), so the bytes drop straight into the DiT ctx_emb tensor.
class TextEncoderRunner {
public:
    TextEncoderRunner(ggml_context* te_ctx, const TextEncoderConfig& cfg,
                      std::shared_ptr<tokenizers::LlamaBpeTokenizer> tokenizer);
    ~TextEncoderRunner();

    TextEncoderRunner(const TextEncoderRunner&) = delete;
    TextEncoderRunner& operator=(const TextEncoderRunner&) = delete;

    // Encode `text`, writing max_seq*hidden f32 values into `out`.
    // Returns the real (non-padded) token count, or -1 on error.
    int encode(const std::string& text, float* out, std::string* error);

    const TextEncoderConfig& config() const { return cfg_; }

private:
    // Lazy GPU init on first encode(): pick best backend, migrate the TE
    // weight tensors in ext_ctx_ to GPU (same pattern as DiTRunner).
    bool ensure_backend(std::string* error);

    ggml_context*                                    ext_ctx_;
    TextEncoderConfig                                cfg_;
    std::shared_ptr<tokenizers::LlamaBpeTokenizer>   tokenizer_;

    std::unique_ptr<ggml_utils::BackendPair> bp_;
    ggml_backend_t        cuda_backend_       = nullptr;
    ggml_backend_buffer_t migrated_buf_       = nullptr;
    bool                  backend_initialized_ = false;
};

}  // namespace engine::models::moss_sfx_v2

#endif  // ENGINE_MODELS_MOSS_SFX_V2_TEXT_ENCODER_H
