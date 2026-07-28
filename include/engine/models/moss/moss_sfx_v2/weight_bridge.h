// weight_bridge.h — populate a no_alloc ggml_context from audiocpp's TensorSource.
//
// The ported DiTRunner / VAERunner read weights by name via
//   ggml_get_tensor(ext_ctx, "moss_sfx_v2.xxx.weight")
// exactly as audiocore's versions did. audiocore's GgufReader handed them a
// context whose tensors were mmap'd straight out of the GGUF. audiocpp's
// TensorSource is the equivalent front end: it exposes per-tensor raw bytes
// (in the stored dtype) + PyTorch-convention shape metadata.
//
// This bridge mirrors the audiocore loader's materialize step: for every
// tensor in `source` whose name starts with `prefix`, allocate a matching
// ggml_tensor in `ctx`, copy the raw bytes in, and set its name so the
// runners can find it. The owned buffer storage keeps the bytes alive for the
// lifetime of the context.

#ifndef ENGINE_MODELS_MOSS_SFX_V2_WEIGHT_BRIDGE_H
#define ENGINE_MODELS_MOSS_SFX_V2_WEIGHT_BRIDGE_H

#include "engine/framework/assets/tensor_source.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ggml_context;

namespace engine::models::moss_sfx_v2 {

// Owning handle for a context + the raw byte buffers that back its tensors.
// Destruction frees the ggml_context; the buffers die with this object, so the
// context must outlive any runner that references it (it does — runners are
// members of the session, destroyed before this handle).
struct LoadedWeights {
    ggml_context* ctx = nullptr;
    std::vector<std::vector<std::byte>> owned_buffers;
    size_t tensors_loaded = 0;

    LoadedWeights() = default;
    ~LoadedWeights();
    LoadedWeights(LoadedWeights&&) noexcept = default;
    LoadedWeights& operator=(LoadedWeights&&) noexcept = default;
    LoadedWeights(const LoadedWeights&) = delete;
    LoadedWeights& operator=(const LoadedWeights&) = delete;
};

// Append every tensor in `source` whose name starts with `prefix` into a
// freshly-created no_alloc ggml_context sized for `capacity_tensors` slots.
// `prefix` may be empty to load every tensor. Returns nullptr + sets *error on
// hard failure. On success the returned handle owns the context and the
// materialized byte buffers; tensor->data pointers alias those buffers.
//
// Tensors are reconstructed with their native ggml dtype (BF16/F32/...) and
// PyTorch-convention shape reversed back to ggml ne[], so the raw GGUF bytes
// copy in verbatim — byte-identical to audiocore's mmap'd reader.
std::unique_ptr<LoadedWeights> load_weights_into_context(
    const assets::TensorSource& source,
    const std::string& prefix,
    size_t capacity_tensors,
    std::string* error);

}  // namespace engine::models::moss_sfx_v2

#endif  // ENGINE_MODELS_MOSS_SFX_V2_WEIGHT_BRIDGE_H
