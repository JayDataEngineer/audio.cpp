# ACE-Step 1.5 SFT variant — conversion & integration notes

Status: **DONE** (2026-08-13). SFT generates music end-to-end through the
audiocpp engine, proven in-container on RTX 4090. This document records the
full conversion pathway and every bug found and fixed along the way, so it
never has to be reverse-engineered again.

## TL;DR

```
scripts/build_ace_step_sft_gguf.sh
# → downloads 17 pinned sources, renames + splits, merges with --type orig
# → ace-step-1.5-sft-q8_0.gguf  (sha256 7ff0b40e..., 1663 tensors, 5.4 GB)
# copy to: /mnt/data/models/audio/acestep-cpp-converted/
```

## Goal

Make the SFT quality tier (acestep-v15-sft, 50-step default / max 200,
CFG 7.0 per the ACE-Step 1.5 Musician's Guide) loadable through the existing
audiocpp engine as a third DiT variant alongside turbo and base. The
integration target is the **ComfyUI audiocore extension nodes**
(`vendor/audiocore-comfyui`) — NOT any other backend.

## Sourcing reality (what exists upstream, what doesn't)

| Source | SFT? | Format | Usable? |
|---|---|---|---|
| `audio-cpp/audio.cpp-gguf` | **NO** — turbo/base only (4 files) | merged audiocpp GGUF | canonical, but no SFT |
| `Serveurperso/ACE-Step-1.5-GGUF` (author) | YES | separate-component GGUFs: DiT + LM + TE + VAE | **NOT directly** — see bugs below |
| `ACE-Step/Ace-Step1.5` (official) | **NO** — turbo only | safetensors | no SFT weights (but VAE safetensors IS used) |
| `ACE-Step/acestep-v15-xl-sft` | XL only | safetensors | wrong arch (XL ≠ v15) |

The author's GGUFs are the ONLY public SFT DiT source. They are valid
**converter inputs** after three metadata fixes (below). The VAE is sourced
from the official `ACE-Step/Ace-Step1.5` safetensors (NOT the author's GGUF —
see bug #3).

## Architecture: what the SFT DiT actually is

SFT (and base) DiT is a **strict superset of turbo** (678 vs 520 tensors;
every turbo tensor name also exists in SFT):

- turbo: decoder-only — `decoder.*` + `condition_embedder` + `cross_attn`
  (distilled variant), `is_turbo: true` in config.
- SFT/base: the full two-brain architecture —
  `encoder.lyric_encoder.layers.0-7`, `encoder.timbre_encoder.layers.0-3`,
  `tokenizer.attention_pooler.*`, `detokenizer.layers.*` (+ `null_condition_emb`,
  loaded only when `is_turbo == false`), plus the shared `decoder.*`.
- The author's SFT GGUF **includes the `silence_latent` tensor** — the
  converter splits it into its own namespace (`dit_sft_silence_latent`).

## The fork's GGUF format quirks (bit-level)

1. **Namespace separator is `/`**, not `.` — merged tensor names look like
   `dit_sft_weights/decoder.condition_embedder.bias` (note the inner `.`).
2. **Dims are int64 (8 bytes each)** in tensor info, unlike standard
   gguf.cpp v2 (uint32). A naive reader that uses 4-byte dims desyncs.
3. **GGML_MAX_NAME = 64 is a HARD limit** on both read and write.
   Merged name = `dit_sft_weights/` (16 chars) + suffix must be ≤ 64 →
   suffix ≤ 48 chars. The author's suffixes run up to 62 chars; 156 of 678
   overflow once prefixed. The `_renamed.<i>` convention + `audiocpp.tensor_names`
   KV (in `tools/gguf_rename.py`) resolves this: over-long names are replaced
   with `_renamed.<index>` and the original name is preserved in the KV array.
4. **The authoritative shape mechanism** (THE root cause of most bugs —
   see below): `audiocpp.tensor_ranks` (int32 array, atype=5) and
   `audiocpp.tensor_shapes` (int64 array, atype=11) KV arrays. When present,
   `GgufTensorSource` uses these INSTEAD of raw ggml dims. The converter
   writes these from the INPUT source's shape metadata. If the input has
   corrupted rank metadata, the corruption propagates into the merged output.
5. Merged files are **single-variant**. Variant selection happens by filename
   substring in `loader.cpp` (`selection_from_request`): `"base"` →
   `acestep-v15-base`, `"turbo"` → `acestep-v15-turbo`, `"sft"` →
   `acestep-v15-sft`.

## The three bugs in Serveurperso's GGUFs (and their fixes)

### Bug 1: 30-tensor rank loss (DiT component)

**Symptom**: `tensor shape mismatch for dit_sft_weights/detokenizer.special_tokens`

**Root cause**: The author's GGUF writer stripped leading-1 batch dimensions
from 30 tensors (25 `scale_shift_table`, `detokenizer.special_tokens`, 2
`special_token`, `null_condition_emb`, `silence_latent`). The ggml function
`ggml_n_dims()` drops trailing-1 dims, so the raw GGML dims are unrecoverable
from the file alone.

**Fix**: `tools/gguf_rename.py` now injects `audiocpp.tensor_ranks` +
`audiocpp.tensor_shapes` KVs into the input GGUF before merge. The converter
reads these as authoritative and propagates correct shapes. Data bytes are
unchanged — metadata-only fix. The renamer knows which tensors need rank
fixes via `_RANK_FIX_SUFFIXES` and `_RANK_FIX_EXACT`.

### Bug 2: silence_latent dims swapped

**Symptom**: `tensor shape mismatch for dit_sft_silence_latent/silence_latent`

**Root cause**: `silence_latent` in the author's GGUF has dims stored in
PyTorch order (64, 15000) instead of ggml reversed order (15000, 64). The
generic reversal in `compute_exact_shapes` produced wrong shape (1, 15000, 64)
instead of correct (1, 64, 15000).

**Fix**: `_SHAPE_OVERRIDES = {'silence_latent': [1, 64, 15000]}` in
`gguf_rename.py` — hardcoded because the dims are swapped in the source.

### Bug 3: VAE rank-2 (snake1.alpha flattened)

**Symptom**: `tensor shape mismatch for vae_weights/decoder.block.0.snake1.alpha`

**Root cause**: The author's `vae-BF16.gguf` stores `snake1.alpha` as
ne=(1,2048,1) rank-3, but `ggml_n_dims()` normalizes it to rank-2 (1,2048).
The converter then writes rank-2 to the authoritative shapes KV. The VAE
decoder requires rank-3.

**Fix**: Source the VAE from safetensors instead of GGUF:
`vae/diffusion_pytorch_model.safetensors` from `ACE-Step/Ace-Step1.5`
(sha256 `da17edb6...`). Safetensors preserves PyTorch rank-3 shapes directly.
This is the same VAE source the turbo/base packages use.

## The C++ patches (loader + assets + diffusion)

Three files patched in this fork:

1. **`src/models/ace_step/loader.cpp`** — `selection_from_request` gains an
   `else if (model_name.find("sft") != std::string::npos)` arm that sets
   `dit_model_path = "acestep-v15-sft"`. Load-option docs updated to list
   `acestep-v15-turbo|acestep-v15-base|acestep-v15-sft`.

2. **`src/models/ace_step/assets.cpp`** — `dit_resource_id` gains an sft arm:
   `"acestep-v15-sft"` → `"dit_sft_"` resource prefix. Also contains the
   `ace_step.vae_path` runtime override (for optional ScragVAE — NOT an
   embedded default; the stock VAE must work embedded, like turbo).

3. **`src/models/ace_step/diffusion.cpp`** — L1480:
   `reference_skips_wavelet_dcw = assets_->selection.dit_model_path != "acestep-v15-turbo"`
   (SFT is base-class, skips wavelet DCW — only turbo runs it).

The C++ engine defaults `guidance_scale` to 7.0 (`request_parser.cpp:161`)
and gates CFG on `!config.is_turbo && options.guidance_scale > 1.0F`
(`diffusion.cpp:1479`). Turbo ignores guidance_scale entirely (distilled).
Steps: `num_inference_steps <= 0` means "use engine default" (turbo capped
at 20, SFT uses config default).

## The conversion (single command)

```bash
scripts/build_ace_step_sft_gguf.sh
```

What it does (all inputs SHA-256 pinned, mismatch aborts):
1. Downloads 17 files (4 GGUF components + safetensors VAE + 12 config/tokenizer sidecars).
2. Renames the DiT GGUF: shortens over-64-char names (`_renamed.<i>` convention),
   injects `audiocpp.tensor_ranks` + `audiocpp.tensor_shapes` KVs (fixes bugs 1+2),
   splits `silence_latent` into its own component GGUF.
3. Strips the LM GGUF's Qwen-style `model.` prefix (loader contract = bare names).
4. Merges 5 components with `audiocpp_gguf --type orig` (byte-faithful: no
   re-quantization — preserves the author's q8_0 DiT, bf16 VAE, f32 silence).

**Output**: `ace-step-1.5-sft-q8_0.gguf` — 1663 tensors, 5.4 GB,
sha256 `7ff0b40e1d43ccac4429b6b860450281b08a4b9fdb3d3d2a709a27e621a474f5`.

## Deployment

```bash
# 1. Copy the merged GGUF:
cp build/ace_step_sft/ace-step-1.5-sft-q8_0.gguf \
   /mnt/data/models/audio/acestep-cpp-converted/

# 2. Rebuild the .so IN THE CONTAINER (host glibc ≠ container glibc):
scripts/rebuild_audiocore_native_in_container.sh
# → builds in inference-comfyui, deploys to /usr/local/lib/libaudiocore_native.so
# → survives docker restart, NOT container recreation (rerun after image rebuild)
```

Verify the patches are baked in:
```bash
docker exec inference-comfyui strings /usr/local/lib/libaudiocore_native.so | grep -c acestep-v15-sft
# → 3 (loader + assets + diffusion)
```

## ComfyUI node integration

The SFT variant is exposed through the existing audiocore custom nodes
(`vendor/audiocore-comfyui`):

**Variant selection**: On the `LoadAudiocoreModel` node, set `family=ace_step`
and `extras={"variant":"sft"}`. The Python layer (`core.py::_resolve_model_file`)
substring-matches the variant on the GGUF filename, and the C++ engine
re-detects "sft" from the filename for its internal variant selection.
Empty extras = the spec default (turbo, the largest GGUF in the dir).

**guidance_scale default fix**: The `AudiocoreMusic` node's `guidance_scale`
default was changed from 1.0 → 7.0. The old default of 1.0 overrode the C++
engine's correct default of 7.0, silently disabling CFG for SFT/base
(`diffusion.cpp:1479: use_diffusion_cfg = !is_turbo && guidance_scale > 1.0`).
For turbo this was harmless (turbo ignores CFG entirely), but for SFT it
produced degraded output. 7.0 is correct for base/SFT and harmless for turbo.

**n_diffusion_steps=0**: The node default of 0 means "engine decides" —
`schemas.py` drops 0 before serialization (`build_music_request`), so the
engine applies its own per-variant default. Positive values pass through.

## Proof (in-container, 2026-08-13)

```bash
docker exec inference-comfyui /tmp/audiocpp-fork-src/build-incontainer/bin/audiocpp_cli \
  --task gen --family ace_step \
  --model /mnt/data/models/audio/acestep-cpp-converted/ace-step-1.5-sft-q8_0.gguf \
  --backend cuda --task-route text2music \
  --text "uplifting cinematic orchestral with driving percussion" \
  --duration-seconds 8 --num-inference-steps 16 --guidance-scale 7.0 \
  --out /tmp/sft_incontainer_proof.wav
# → EXIT=0, 1.5 MB WAV (48 kHz stereo 16-bit, 8.0 s), 4 CUDA graph warmups
```

## Tests

`tests/unit/test_audiocore_param_contract.py::TestAceStepSftDefaults` pins:
- `guidance_scale=7.0` reaches the engine body root (not the old 1.0)
- `guidance_scale` absent when unset (engine default takes effect)
- `n_diffusion_steps=0` is dropped (engine picks per-variant default)
- Explicit step counts forward unchanged

## Reference: GGUF type enum (this fork)

Non-standard vs upstream ggml. atype=5 is INT32 (4B), atype=11 is INT64 (8B).
Used for `audiocpp.tensor_ranks` (atype=5) and `audiocpp.tensor_shapes` (atype=11).
