// weight_bridge.cpp — see weight_bridge.h.
//
// Mirrors audiocore's GgufReader::materialize step: take audiocpp's TensorSource
// (which exposes raw GGUF bytes + PyTorch-convention shape metadata per tensor)
// and populate a no_alloc ggml_context whose tensors carry the same names, dtype,
// and ne[] layout the ported DiTRunner/VAERunner expect from ggml_get_tensor().

#include "engine/models/moss/moss_sfx_v2/weight_bridge.h"

#include "ggml.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace engine::models::moss_sfx_v2 {

LoadedWeights::~LoadedWeights() {
    if (ctx != nullptr) {
        ggml_free(ctx);
        ctx = nullptr;
    }
    // owned_buffers freed by vector destructor
}

std::unique_ptr<LoadedWeights> load_weights_into_context(
    const assets::TensorSource& source,
    const std::string& prefix,
    size_t capacity_tensors,
    std::string* error) {

    auto result = std::make_unique<LoadedWeights>();

    // ── 1. Enumerate tensors, filter by prefix ───────────────────────────
    const auto all = source.tensors();
    std::vector<const assets::TensorMetadata*> selected;
    selected.reserve(all.size());
    for (const auto& m : all) {
        if (prefix.empty() ||
            m.name.rfind(prefix, 0) == 0) {
            selected.push_back(&m);
        }
    }
    if (selected.empty()) {
        if (error) *error = "no tensors matching prefix '" + prefix + "'";
        return nullptr;
    }

    const size_t cap = std::max(capacity_tensors, selected.size());
    ggml_init_params gip = {};
    gip.mem_size   = ggml_tensor_overhead() * (cap + 1);
    gip.mem_buffer = nullptr;
    gip.no_alloc   = true;  // we provide data pointers ourselves
    result->ctx = ggml_init(gip);
    if (result->ctx == nullptr) {
        if (error) *error = "ggml_init failed for weight bridge";
        return nullptr;
    }

    // ── 2. Materialize each tensor into an owned buffer ──────────────────
    size_t loaded = 0;
    result->owned_buffers.reserve(selected.size());
    for (const auto* meta : selected) {
        assets::RawTensorData raw;
        try {
            raw = source.require_tensor_data(meta->name);
        } catch (const std::exception& e) {
            std::fprintf(stderr,
                "[moss_sfx_v2] weight_bridge: require_tensor_data FAILED for "
                "%s: %s\n", meta->name.c_str(), e.what());
            continue;
        }

        const ggml_type gtype = assets::ggml_type_for_tensor_dtype(raw.metadata.dtype);

        // PyTorch shape → ggml ne[]. shape[0] is outermost (rows); ne[0] is
        // innermost (fastest). So ne[i] = shape[rank-1-i]. ggml supports up to
        // GGML_MAX_DIMS (4) dimensions.
        const int rank = static_cast<int>(raw.metadata.shape.size());
        if (rank <= 0 || rank > GGML_MAX_DIMS) {
            std::fprintf(stderr,
                "[moss_sfx_v2] weight_bridge: %s has unsupported rank %d\n",
                meta->name.c_str(), rank);
            continue;
        }
        int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
        for (int i = 0; i < rank; ++i) {
            ne[i] = raw.metadata.shape[static_cast<size_t>(rank - 1 - i)];
            if (ne[i] <= 0) ne[i] = 1;
        }

        ggml_tensor* gt = ggml_new_tensor(result->ctx, gtype, rank, ne);
        if (gt == nullptr) {
            std::fprintf(stderr,
                "[moss_sfx_v2] weight_bridge: ggml_new_tensor failed for %s\n",
                meta->name.c_str());
            continue;
        }
        ggml_set_name(gt, meta->name.c_str());

        // Copy raw bytes into an owned buffer; alias it via tensor->data.
        // The bytes are in the source dtype's native layout, which for GGUF
        // is row-major = ggml ne[0]-major, so they copy in verbatim.
        result->owned_buffers.emplace_back(std::move(raw.bytes));
        const auto& buf = result->owned_buffers.back();
        const size_t want_bytes = ggml_nbytes(gt);
        if (buf.size() < want_bytes) {
            std::fprintf(stderr,
                "[moss_sfx_v2] weight_bridge: %s byte size %zu < ggml_nbytes %zu"
                " (dtype=%s rank=%d ne=[%lld,%lld,%lld,%lld])\n",
                meta->name.c_str(), buf.size(), want_bytes,
                raw.metadata.dtype.c_str(), rank,
                (long long)ne[0], (long long)ne[1],
                (long long)ne[2], (long long)ne[3]);
            // Still proceed — partial buffer is better than skipping; the
            // runner will read within bounds since ne[] matches the metadata.
        }
        gt->data = const_cast<std::byte*>(buf.data());
        ++loaded;
    }

    result->tensors_loaded = loaded;
    std::fprintf(stderr,
        "[moss_sfx_v2] weight_bridge: loaded %zu/%zu tensors (prefix='%s')\n",
        loaded, selected.size(), prefix.c_str());
    return result;
}

}  // namespace engine::models::moss_sfx_v2
