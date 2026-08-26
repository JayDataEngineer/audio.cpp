#!/usr/bin/env bash
# Build the merged ACE-Step 1.5 SFT GGUF package (ace-step-1.5-sft-q8_0.gguf)
# reproducibly from pinned upstream sources.
#
# Every input is content-pinned by SHA-256; a mismatch aborts the build.
# The DiT component needs its two over-GGML_MAX_NAME tensor names shortened
# (tools/gguf_rename.py mirrors the converter's own `_audiocpp.<i>` +
# `audiocpp.tensor_names` metadata convention) and its embedded silence_latent
# split out into its own spec namespace. The converter then merges the five
# component files with `--type orig` (byte-faithful: no re-quantization, so the
# package keeps the author's exact tensor types — q8_0 DiT bulk, bf16 VAE,
# f32 silence — matching the audio-cpp turbo/base package layout).
#
# Usage:
#   scripts/build_ace_step_sft_gguf.sh [<workdir>]
#
# Env overrides:
#   AUDIOCPP_CONVERTER   path to audiocpp_gguf (default: build/bin/audiocpp_gguf
#                        relative to the repo root — build with
#                        `cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build`)
#   AUDIOCPP_SFT_CACHE   dir to download sources into (default: <workdir>/downloads)
#   AUDIOCPP_SFT_ROOT    dir with the sidecar layout (default: <workdir>/root)
#
# Output: <workdir>/ace-step-1.5-sft-q8_0.gguf  (1663 tensors, embedded spec
#         family "ace_step" + all 14 sidecars) — prints its SHA-256 on success.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${1:-"$REPO_ROOT/build/ace_step_sft"}"
CACHE="${AUDIOCPP_SFT_CACHE:-"$WORKDIR/downloads"}"
ROOTDIR="${AUDIOCPP_SFT_ROOT:-"$WORKDIR/root"}"
CONVERTER="${AUDIOCPP_CONVERTER:-"$REPO_ROOT/build/bin/audiocpp_gguf"}"
RENAMER="$REPO_ROOT/tools/gguf_rename.py"
SPEC="$REPO_ROOT/model_specs/ace_step_sft.json"

mkdir -p "$CACHE" "$ROOTDIR"

# ---- pinned manifest: <file>|<huggingface repo>|<path in repo>|<sha256> ----
MANIFEST=(
  # DiT (Serveurperso/ACE-Step-1.5-GGUF — audio-cpp component conversions)
  "acestep-v15-sft-Q8_0.gguf|Serveurperso/ACE-Step-1.5-GGUF|acestep-v15-sft-Q8_0.gguf|17f1984e48aaab27b3eb8ccbf0b754a6656e677884c60ea7003845cfc0059b70"
  "acestep-5Hz-lm-1.7B-Q8_0.gguf|Serveurperso/ACE-Step-1.5-GGUF|acestep-5Hz-lm-1.7B-Q8_0.gguf|726f99a82f050b32ebc5c5e36acaa9d7acd06bfe5e579a62b00e0d0d6d0b7ec6"
  "Qwen3-Embedding-0.6B-Q8_0.gguf|Serveurperso/ACE-Step-1.5-GGUF|Qwen3-Embedding-0.6B-Q8_0.gguf|972f23255e46adfe744a0eb9a0039f3c63988f65753b0968d776e8b27168c321"
  # VAE from SAFETENSORS, not the Serveurperso GGUF: the GGUF flattened the
  # snake-alpha tensors to rank-2 (1,2048) but vae_decoder requires rank-3
  # (1,2048,1). The original safetensors preserves rank-3 — same source the
  # audio-cpp turbo/base packages use. (Serveurperso's vae-BF16.gguf is a
  # trap: same BF16 data, wrong rank metadata → "tensor shape mismatch for
  # vae_weights/decoder.block.0.snake1.alpha" at load time.)
  "vae/diffusion_pytorch_model.safetensors|ACE-Step/Ace-Step1.5|vae/diffusion_pytorch_model.safetensors|da17edb604c40deaf09e9b24974e590d1ca83a374070e5d0884cfa4bed9a99b0"
  # LM + text encoder tokenizer sets + vae config (ACE-Step/Ace-Step1.5)
  "acestep-5Hz-lm-1.7B/config.json|ACE-Step/Ace-Step1.5|acestep-5Hz-lm-1.7B/config.json|73515ca4bfd86a5b56f28bf680311d1911e159c85d7f7ceb2188e310007c9010"
  "acestep-5Hz-lm-1.7B/tokenizer_config.json|ACE-Step/Ace-Step1.5|acestep-5Hz-lm-1.7B/tokenizer_config.json|6cd70cdd89425971794f5235562edcc608b0629a6c4686ae51a8b8c8b8ba5e95"
  "acestep-5Hz-lm-1.7B/vocab.json|ACE-Step/Ace-Step1.5|acestep-5Hz-lm-1.7B/vocab.json|ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"
  "acestep-5Hz-lm-1.7B/merges.txt|ACE-Step/Ace-Step1.5|acestep-5Hz-lm-1.7B/merges.txt|8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"
  "acestep-5Hz-lm-1.7B/tokenizer.json|ACE-Step/Ace-Step1.5|acestep-5Hz-lm-1.7B/tokenizer.json|35af56c3f5cb3ea2cc578aa28a8937770981d504f183ac5c8c38baf4bbd4af4d"
  "acestep-5Hz-lm-1.7B/chat_template.jinja|ACE-Step/Ace-Step1.5|acestep-5Hz-lm-1.7B/chat_template.jinja|a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
  "Qwen3-Embedding-0.6B/config.json|ACE-Step/Ace-Step1.5|Qwen3-Embedding-0.6B/config.json|bb23c1607cfe059a58d8f0196cf1cebb52082b1056b8e358a579da80a5759420"
  "Qwen3-Embedding-0.6B/tokenizer_config.json|ACE-Step/Ace-Step1.5|Qwen3-Embedding-0.6B/tokenizer_config.json|443bfa629eb16387a12edbf92a76f6a6f10b2af3b53d87ba1550adfcf45f7fa0"
  "Qwen3-Embedding-0.6B/vocab.json|ACE-Step/Ace-Step1.5|Qwen3-Embedding-0.6B/vocab.json|ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"
  "Qwen3-Embedding-0.6B/merges.txt|ACE-Step/Ace-Step1.5|Qwen3-Embedding-0.6B/merges.txt|8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"
  "Qwen3-Embedding-0.6B/tokenizer.json|ACE-Step/Ace-Step1.5|Qwen3-Embedding-0.6B/tokenizer.json|def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a"
  "Qwen3-Embedding-0.6B/chat_template.jinja|ACE-Step/Ace-Step1.5|Qwen3-Embedding-0.6B/chat_template.jinja|87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5"
  "vae/config.json|ACE-Step/Ace-Step1.5|vae/config.json|14e019904df567f26df750317a70e2bd08f9f8f3c40ff4a24c97d1cd3f20ccd2"
  # authoritative SFT DiT config (ACE-Step/acestep-v15-sft)
  "acestep-v15-sft/config.json|ACE-Step/acestep-v15-sft|config.json|9bb4f832f2e5e6c8bf7bddab3c6a6da1b13c01d072efbf0ed3c830536c473359"
)

fetch() { # <file> <repo> <path> <sha256>  ->  <cache>/<file>
  local file="$1" repo="$2" path="$3" want="$4"
  local dst="$CACHE/$file"
  if [[ -f "$dst" ]] && [[ "$(sha256sum "$dst" | cut -d' ' -f1)" == "$want" ]]; then
    echo "ok   cached  $file"
    return
  fi
  echo "get  $repo/$path"
  local url="https://huggingface.co/$repo/resolve/main/$path"
  curl -fL --retry 3 --progress-bar -o "$dst" "$url"
  local got
  got="$(sha256sum "$dst" | cut -d' ' -f1)"
  if [[ "$got" != "$want" ]]; then
    echo "FAIL sha256 mismatch for $file: want $want got $got" >&2
    rm -f "$dst"
    exit 1
  fi
}

for entry in "${MANIFEST[@]}"; do
  IFS='|' read -r file repo path want <<<"$entry"
  fetch "$file" "$repo" "$path" "$want"
done

# ---- assemble the sidecar root exactly as the spec's model: refs require ----
for entry in "${MANIFEST[@]}"; do
  IFS='|' read -r file repo path want <<<"$entry"
  if [[ "$file" != *.gguf ]]; then
    mkdir -p "$ROOTDIR/$(dirname "$file")"
    cp "$CACHE/$file" "$ROOTDIR/$file"
  fi
done

echo "== rename + split the DiT component =="
python3 "$RENAMER" "$CACHE/acestep-v15-sft-Q8_0.gguf" "$WORKDIR/acestep-v15-sft-Q8_0.renamed.gguf" \
  --drop silence_latent --split-silence "$WORKDIR/acestep-v15-sft-silence.gguf"

echo "== strip the LM component's Qwen-style 'model.' prefix =="
# The author's LM GGUF ships Qwen-style `model.`-prefixed names; the loader
# contract (set by the audio-cpp turbo/base packages) is bare — the turbo
# package's lm_weights are `embed_tokens.weight`, `layers.N.…`.
python3 "$RENAMER" "$CACHE/acestep-5Hz-lm-1.7B-Q8_0.gguf" "$WORKDIR/acestep-5Hz-lm-1.7B-Q8_0.stripped.gguf" \
  --strip-prefix model.

echo "== merge with audiocpp_gguf ($CONVERTER) =="
"$CONVERTER" \
  --input "dit_sft_weights=$WORKDIR/acestep-v15-sft-Q8_0.renamed.gguf" \
  --input "dit_sft_silence_latent=$WORKDIR/acestep-v15-sft-silence.gguf" \
  --input "lm_weights=$WORKDIR/acestep-5Hz-lm-1.7B-Q8_0.stripped.gguf" \
  --input "text_encoder_weights=$CACHE/Qwen3-Embedding-0.6B-Q8_0.gguf" \
  --input "vae_weights=$CACHE/vae/diffusion_pytorch_model.safetensors" \
  --model-spec "$SPEC" --root "$ROOTDIR" --type orig --overwrite \
  --output "$WORKDIR/ace-step-1.5-sft-q8_0.gguf"

echo "== validate with the fork's own ggml reader =="
"$CONVERTER" --inspect "$WORKDIR/ace-step-1.5-sft-q8_0.gguf"

OUT="$WORKDIR/ace-step-1.5-sft-q8_0.gguf"
echo "== DONE =="
echo "package:  $OUT"
echo "sha256:   $(sha256sum "$OUT" | cut -d' ' -f1)"
echo "size:     $(du -h "$OUT" | cut -f1)"
