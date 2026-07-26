#include "engine/models/inflect_tts/runtime.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/models/inflect_tts/frontend.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace engine::models::inflect_tts {
namespace {

using Clock = std::chrono::steady_clock;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct GgmlGallocrDeleter {
    void operator()(ggml_gallocr * gallocr) const noexcept {
        if (gallocr != nullptr) {
            ggml_gallocr_free(gallocr);
        }
    }
};

std::unique_ptr<ggml_context, GgmlContextDeleter> make_context(size_t bytes = 96ull * 1024ull * 1024ull) {
    ggml_init_params params{bytes, nullptr, true};
    auto * ctx = ggml_init(params);
    if (ctx == nullptr) {
        throw std::runtime_error("failed to initialize Inflect graph context");
    }
    return std::unique_ptr<ggml_context, GgmlContextDeleter>(ctx);
}

class GraphRunner {
public:
    explicit GraphRunner(core::ExecutionContext & execution_context)
        : execution_context_(execution_context),
          gallocr_(
              ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_context.backend())),
              GgmlGallocrDeleter{}) {
        if (gallocr_ == nullptr) {
            throw std::runtime_error("failed to create Inflect graph allocator");
        }
    }

    ggml_cgraph * allocate(ggml_context * ctx, ggml_tensor * output) {
        auto * graph = ggml_new_graph_custom(ctx, 65536, false);
        ggml_build_forward_expand(graph, output);
        if (!ggml_gallocr_alloc_graph(gallocr_.get(), graph)) {
            throw std::runtime_error("failed to allocate Inflect graph");
        }
        core::prepare_host_graph_plan(execution_context_, graph, plan_);
        return graph;
    }

    void compute(ggml_cgraph * graph, const char * label) {
        const ggml_status status = core::compute_graph(execution_context_, graph, plan_, label);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error(std::string("Inflect graph compute failed: ") + label);
        }
    }

private:
    core::ExecutionContext & execution_context_;
    std::unique_ptr<ggml_gallocr, GgmlGallocrDeleter> gallocr_;
    core::HostGraphPlan plan_;
};

core::TensorValue ensure_contiguous(core::ModuleBuildContext & ctx, const core::TensorValue & value) {
    if (core::has_backend_addressable_layout(value.tensor)) {
        return value;
    }
    return core::wrap_tensor(ggml_cont(ctx.ggml, value.tensor), value.shape, value.type);
}

core::TensorValue add_tensors(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & lhs,
    const core::TensorValue & rhs) {
    const auto left = ensure_contiguous(ctx, lhs);
    const auto right = ensure_contiguous(ctx, rhs);
    return core::wrap_tensor(ggml_add(ctx.ggml, left.tensor, right.tensor), lhs.shape, GGML_TYPE_F32);
}

core::TensorValue sub_tensors(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & lhs,
    const core::TensorValue & rhs) {
    const auto left = ensure_contiguous(ctx, lhs);
    const auto right = ensure_contiguous(ctx, rhs);
    return core::wrap_tensor(ggml_sub(ctx.ggml, left.tensor, right.tensor), lhs.shape, GGML_TYPE_F32);
}

core::TensorValue mul_tensors(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & lhs,
    const core::TensorValue & rhs) {
    const auto left = ensure_contiguous(ctx, lhs);
    const auto right = ensure_contiguous(ctx, rhs);
    return core::wrap_tensor(ggml_mul(ctx.ggml, left.tensor, right.tensor), lhs.shape, GGML_TYPE_F32);
}

struct GraphUploads {
    std::vector<std::pair<ggml_tensor *, std::vector<float>>> f32;
    std::vector<std::pair<ggml_tensor *, std::vector<int32_t>>> i32;
};

core::TensorValue make_f32_input(
    core::ModuleBuildContext & ctx,
    const core::TensorShape & shape,
    std::vector<float> values,
    GraphUploads & uploads) {
    auto tensor = core::make_tensor(ctx, GGML_TYPE_F32, shape);
    uploads.f32.emplace_back(tensor.tensor, std::move(values));
    return tensor;
}

core::TensorValue make_i32_input(
    core::ModuleBuildContext & ctx,
    const core::TensorShape & shape,
    std::vector<int32_t> values,
    GraphUploads & uploads) {
    auto tensor = core::make_tensor(ctx, GGML_TYPE_I32, shape);
    uploads.i32.emplace_back(tensor.tensor, std::move(values));
    return tensor;
}

std::vector<float> padded_relative_table(
    const std::vector<float> & source,
    int64_t frames,
    int64_t head_dim) {
    constexpr int64_t kWindow = 4;
    constexpr int64_t kSourceRows = kWindow * 2 + 1;
    const int64_t rows = frames * 2 - 1;
    std::vector<float> out(static_cast<size_t>(rows * head_dim), 0.0f);
    if (frames >= kWindow + 1) {
        const int64_t target_start = frames - (kWindow + 1);
        for (int64_t row = 0; row < kSourceRows; ++row) {
            std::copy_n(
                source.begin() + static_cast<std::ptrdiff_t>(row * head_dim),
                static_cast<size_t>(head_dim),
                out.begin() + static_cast<std::ptrdiff_t>((target_start + row) * head_dim));
        }
        return out;
    }
    const int64_t source_start = (kWindow + 1) - frames;
    for (int64_t row = 0; row < rows; ++row) {
        std::copy_n(
            source.begin() + static_cast<std::ptrdiff_t>((source_start + row) * head_dim),
            static_cast<size_t>(head_dim),
            out.begin() + static_cast<std::ptrdiff_t>(row * head_dim));
    }
    return out;
}

std::vector<float> transpose_relative_table(
    const std::vector<float> & values,
    int64_t rows,
    int64_t head_dim) {
    std::vector<float> out(static_cast<size_t>(rows * head_dim));
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t dim = 0; dim < head_dim; ++dim) {
            out[static_cast<size_t>(dim * rows + row)] =
                values[static_cast<size_t>(row * head_dim + dim)];
        }
    }
    return out;
}

class WeightMap {
public:
    WeightMap(
        const assets::TensorSource & source,
        core::ExecutionContext & execution_context,
        assets::TensorStorageType weight_type)
        : store_(
              execution_context.backend(),
              execution_context.backend_type(),
              "inflect_tts.weights",
              64ull * 1024ull * 1024ull) {
        const auto tensors = source.tensors();
        values_.reserve(tensors.size());
        for (const auto & tensor : tensors) {
            if (tensor.name.find("emb_rel_") != std::string::npos) {
                relative_values_.emplace(tensor.name, source.require_f32(tensor.name, tensor.shape));
            }
            values_.emplace(
                tensor.name,
                store_.load_tensor(source, tensor.name, weight_type, tensor.shape));
        }
        store_.upload();
        source.release_storage();
    }

    const core::TensorValue & at(const std::string & name) const {
        const auto it = values_.find(name);
        if (it == values_.end()) {
            throw std::runtime_error("Inflect weights missing tensor: " + name);
        }
        return it->second;
    }

    const std::vector<float> & relative(const std::string & name) const {
        const auto it = relative_values_.find(name);
        if (it == relative_values_.end()) {
            throw std::runtime_error("Inflect weights missing relative tensor: " + name);
        }
        return it->second;
    }

private:
    core::BackendWeightStore store_;
    std::unordered_map<std::string, core::TensorValue> values_;
    std::unordered_map<std::string, std::vector<float>> relative_values_;
};

modules::Conv1dWeights conv1d_weights(const WeightMap & weights, const std::string & prefix, bool bias = true) {
    modules::Conv1dWeights out;
    out.weight = weights.at(prefix + ".weight");
    if (bias) {
        out.bias = weights.at(prefix + ".bias");
    }
    return out;
}

modules::NormWeights norm_weights(const WeightMap & weights, const std::string & prefix) {
    return {weights.at(prefix + ".gamma"), weights.at(prefix + ".beta")};
}

core::TensorValue add_channel_bias(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & output,
    const core::TensorValue & bias,
    int64_t out_channels) {
    core::validate_shape(bias, core::TensorShape::from_dims({out_channels}), "bias");
    auto bias_value = bias.type == GGML_TYPE_F32
        ? bias
        : core::wrap_tensor(ggml_cast(ctx.ggml, bias.tensor, GGML_TYPE_F32), bias.shape, GGML_TYPE_F32);
    bias_value = core::reshape_tensor(ctx, bias_value, core::TensorShape::from_dims({1, out_channels, 1}));
    bias_value = core::wrap_tensor(ggml_repeat(ctx.ggml, bias_value.tensor, output.tensor), output.shape, GGML_TYPE_F32);
    return add_tensors(ctx, output, bias_value);
}

core::TensorValue conv1d(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const WeightMap & weights,
    const std::string & prefix,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel,
    int padding,
    int dilation = 1,
    bool bias = true) {
    auto output = modules::Conv1dModule({in_channels, out_channels, kernel, 1, padding, dilation, false})
        .build(ctx, input, conv1d_weights(weights, prefix, false));
    if (bias) {
        output = add_channel_bias(ctx, output, weights.at(prefix + ".bias"), out_channels);
    }
    return output;
}

core::TensorValue relative_position_to_absolute_position(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t length) {
    auto x = ensure_contiguous(ctx, input);
    x = core::wrap_tensor(
        ggml_pad(ctx.ggml, x.tensor, 1, 0, 0, 0),
        core::TensorShape::from_dims({length, 2 * length}),
        GGML_TYPE_F32);
    x = core::reshape_tensor(ctx, ensure_contiguous(ctx, x), core::TensorShape::from_dims({length * 2 * length}));
    x = core::wrap_tensor(
        ggml_pad(ctx.ggml, x.tensor, static_cast<int>(length - 1), 0, 0, 0),
        core::TensorShape::from_dims({length * 2 * length + length - 1}),
        GGML_TYPE_F32);
    x = core::reshape_tensor(ctx, ensure_contiguous(ctx, x), core::TensorShape::from_dims({length + 1, 2 * length - 1}));
    x = modules::SliceModule({0, 0, length}).build(ctx, x);
    return modules::SliceModule({1, length - 1, length}).build(ctx, x);
}

core::TensorValue absolute_position_to_relative_position(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t length) {
    auto x = ensure_contiguous(ctx, input);
    x = core::wrap_tensor(
        ggml_pad(ctx.ggml, x.tensor, static_cast<int>(length - 1), 0, 0, 0),
        core::TensorShape::from_dims({length, 2 * length - 1}),
        GGML_TYPE_F32);
    x = core::reshape_tensor(ctx, ensure_contiguous(ctx, x), core::TensorShape::from_dims({length * (2 * length - 1)}));
    x = core::wrap_tensor(
        ggml_pad_ext(ctx.ggml, x.tensor, static_cast<int>(length), 0, 0, 0, 0, 0, 0, 0),
        core::TensorShape::from_dims({length * (2 * length - 1) + length}),
        GGML_TYPE_F32);
    x = core::reshape_tensor(ctx, ensure_contiguous(ctx, x), core::TensorShape::from_dims({length, 2 * length}));
    return modules::SliceModule({1, 1, 2 * length - 1}).build(ctx, x);
}

core::TensorValue channel_layer_norm(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const WeightMap & weights,
    const std::string & prefix,
    int64_t channels) {
    auto x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, input);
    x = modules::LayerNormModule({channels, 1e-5f, true, true}).build(ctx, x, norm_weights(weights, prefix));
    return modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
}

core::TensorValue maskless_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const WeightMap & weights,
    const std::string & prefix,
    const InflectConfig & config,
    GraphUploads & uploads) {
    const int64_t channels = config.hidden_channels;
    const int64_t head_dim = channels / config.n_heads;
    const int64_t frames = input.shape.dims[2];
    auto q = conv1d(ctx, input, weights, prefix + ".conv_q", channels, channels, 1, 0);
    auto k = conv1d(ctx, input, weights, prefix + ".conv_k", channels, channels, 1, 0);
    auto v = conv1d(ctx, input, weights, prefix + ".conv_v", channels, channels, 1, 0);
    const int64_t relative_rows = frames * 2 - 1;
    const auto relative_k_rows = padded_relative_table(weights.relative(prefix + ".emb_rel_k"), frames, head_dim);
    const auto relative_v_rows = padded_relative_table(weights.relative(prefix + ".emb_rel_v"), frames, head_dim);
    const auto relative_k = make_f32_input(
        ctx,
        core::TensorShape::from_dims({head_dim, relative_rows}),
        transpose_relative_table(relative_k_rows, relative_rows, head_dim),
        uploads);
    const auto relative_v = make_f32_input(
        ctx,
        core::TensorShape::from_dims({relative_rows, head_dim}),
        relative_v_rows,
        uploads);
    core::TensorValue merged;
    const modules::MatMulModule matmul;
    for (int64_t head = 0; head < config.n_heads; ++head) {
        auto qh = modules::SliceModule({1, head * head_dim, head_dim}).build(ctx, q);
        auto kh = modules::SliceModule({1, head * head_dim, head_dim}).build(ctx, k);
        auto vh = modules::SliceModule({1, head * head_dim, head_dim}).build(ctx, v);
        qh = core::reshape_tensor(ctx, ensure_contiguous(ctx, qh), core::TensorShape::from_dims({head_dim, frames}));
        kh = core::reshape_tensor(ctx, ensure_contiguous(ctx, kh), core::TensorShape::from_dims({head_dim, frames}));
        vh = core::reshape_tensor(ctx, ensure_contiguous(ctx, vh), core::TensorShape::from_dims({head_dim, frames}));
        auto q_td = ensure_contiguous(ctx, modules::TransposeModule({{1, 0, 2, 3}, 2}).build(ctx, qh));
        auto scores = matmul.build(ctx, q_td, kh);
        scores = core::wrap_tensor(
            ggml_scale(ctx.ggml, scores.tensor, 1.0f / std::sqrt(static_cast<float>(head_dim))),
            scores.shape,
            GGML_TYPE_F32);
        auto rel_logits = matmul.build(ctx, q_td, relative_k);
        rel_logits = core::wrap_tensor(
            ggml_scale(ctx.ggml, rel_logits.tensor, 1.0f / std::sqrt(static_cast<float>(head_dim))),
            rel_logits.shape,
            GGML_TYPE_F32);
        auto relative_scores = relative_position_to_absolute_position(ctx, rel_logits, frames);
        scores = add_tensors(ctx, scores, relative_scores);
        auto attn = core::wrap_tensor(
            ggml_soft_max(ctx.ggml, ensure_contiguous(ctx, scores).tensor),
            scores.shape,
            GGML_TYPE_F32);
        auto vh_td = ensure_contiguous(ctx, modules::TransposeModule({{1, 0, 2, 3}, 2}).build(ctx, vh));
        auto context = matmul.build(ctx, ensure_contiguous(ctx, attn), vh_td);
        auto relative_weights = absolute_position_to_relative_position(ctx, attn, frames);
        auto relative_context = matmul.build(ctx, ensure_contiguous(ctx, relative_weights), relative_v);
        context = add_tensors(ctx, context, relative_context);
        context = modules::TransposeModule({{1, 0, 2, 3}, 2}).build(ctx, context);
        context = core::reshape_tensor(ctx, ensure_contiguous(ctx, context), core::TensorShape::from_dims({1, head_dim, frames}));
        merged = merged.valid() ? modules::ConcatModule({1}).build(ctx, merged, context) : context;
    }
    return conv1d(ctx, merged, weights, prefix + ".conv_o", channels, channels, 1, 0);
}

core::TensorValue feed_forward(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const WeightMap & weights,
    const std::string & prefix,
    const InflectConfig & config) {
    auto x = conv1d(ctx, input, weights, prefix + ".conv_1", config.hidden_channels, config.filter_channels, config.kernel_size, 1);
    x = modules::ReluModule{}.build(ctx, x);
    return conv1d(ctx, x, weights, prefix + ".conv_2", config.filter_channels, config.hidden_channels, config.kernel_size, 1);
}

core::TensorValue text_encoder(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & ids,
    const WeightMap & weights,
    const InflectConfig & config,
    GraphUploads & uploads) {
    const int64_t frames = ids.shape.dims[0];
    auto x = modules::EmbeddingModule({config.vocab_size, config.hidden_channels}).build(ctx, ids, weights.at("enc_p.emb.weight"));
    x = core::wrap_tensor(
        ggml_scale(ctx.ggml, x.tensor, std::sqrt(static_cast<float>(config.hidden_channels))),
        x.shape,
        GGML_TYPE_F32);
    x = modules::TransposeModule({{1, 0, 2, 3}, 2}).build(ctx, x);
    x = core::reshape_tensor(ctx, ensure_contiguous(ctx, x), core::TensorShape::from_dims({1, config.hidden_channels, frames}));
    for (int64_t layer = 0; layer < config.n_layers; ++layer) {
        const std::string base = "enc_p.encoder";
        auto y = maskless_attention(ctx, x, weights, base + ".attn_layers." + std::to_string(layer), config, uploads);
        x = channel_layer_norm(
            ctx,
            add_tensors(ctx, x, y),
            weights,
            base + ".norm_layers_1." + std::to_string(layer),
            config.hidden_channels);
        y = feed_forward(ctx, x, weights, base + ".ffn_layers." + std::to_string(layer), config);
        x = channel_layer_norm(
            ctx,
            add_tensors(ctx, x, y),
            weights,
            base + ".norm_layers_2." + std::to_string(layer),
            config.hidden_channels);
    }
    return x;
}

core::TensorValue duration_predictor(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const WeightMap & weights,
    const InflectConfig & config) {
    auto x = conv1d(ctx, input, weights, "dp.conv_1", config.hidden_channels, config.duration_channels, 3, 1);
    x = modules::ReluModule{}.build(ctx, x);
    x = channel_layer_norm(ctx, x, weights, "dp.norm_1", config.duration_channels);
    x = conv1d(ctx, x, weights, "dp.conv_2", config.duration_channels, config.duration_channels, 3, 1);
    x = modules::ReluModule{}.build(ctx, x);
    x = channel_layer_norm(ctx, x, weights, "dp.norm_2", config.duration_channels);
    return conv1d(ctx, x, weights, "dp.proj", config.duration_channels, 1, 1, 0);
}

core::TensorValue prior_graph_output(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & ids,
    const WeightMap & weights,
    const InflectConfig & config,
    GraphUploads & uploads) {
    auto x = text_encoder(ctx, ids, weights, config, uploads);
    auto stats = conv1d(ctx, x, weights, "enc_p.proj", config.hidden_channels, config.inter_channels * 2, 1, 0);
    auto logw = duration_predictor(ctx, x, weights, config);
    return ensure_contiguous(ctx, modules::ConcatModule({1}).build(ctx, stats, logw));
}

core::TensorValue fused_tanh_sigmoid(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t half_channels) {
    auto a = modules::SliceModule({1, 0, half_channels}).build(ctx, input);
    auto b = modules::SliceModule({1, half_channels, half_channels}).build(ctx, input);
    a = modules::TanhModule{}.build(ctx, a);
    b = modules::SigmoidModule{}.build(ctx, b);
    return mul_tensors(ctx, a, b);
}

core::TensorValue wn(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const WeightMap & weights,
    const std::string & prefix,
    int64_t channels) {
    auto x = input;
    core::TensorValue output;
    for (int64_t layer = 0; layer < 4; ++layer) {
        const int dilation = 1;
        auto x_in = conv1d(
            ctx,
            x,
            weights,
            prefix + ".in_layers." + std::to_string(layer),
            channels,
            channels * 2,
            5,
            2,
            dilation);
        auto acts = fused_tanh_sigmoid(ctx, x_in, channels);
        const int64_t res_skip_channels = layer < 3 ? channels * 2 : channels;
        auto res_skip = conv1d(
            ctx,
            acts,
            weights,
            prefix + ".res_skip_layers." + std::to_string(layer),
            channels,
            res_skip_channels,
            1,
            0);
        if (layer < 3) {
            auto res = modules::SliceModule({1, 0, channels}).build(ctx, res_skip);
            auto skip = modules::SliceModule({1, channels, channels}).build(ctx, res_skip);
            x = add_tensors(ctx, x, res);
            output = output.valid() ? add_tensors(ctx, output, skip) : skip;
        } else {
            output = output.valid() ? add_tensors(ctx, output, res_skip) : res_skip;
        }
    }
    return output;
}

core::TensorValue residual_coupling_reverse(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const WeightMap & weights,
    int64_t flow_index,
    const InflectConfig & config) {
    const int64_t half = config.inter_channels / 2;
    const std::string prefix = "flow.flows." + std::to_string(flow_index);
    auto x0 = modules::SliceModule({1, 0, half}).build(ctx, input);
    auto x1 = modules::SliceModule({1, half, half}).build(ctx, input);
    auto h = conv1d(ctx, x0, weights, prefix + ".pre", half, config.hidden_channels, 1, 0);
    h = wn(ctx, h, weights, prefix + ".enc", config.hidden_channels);
    auto m = conv1d(ctx, h, weights, prefix + ".post", config.hidden_channels, half, 1, 0);
    x1 = sub_tensors(ctx, x1, m);
    return modules::ConcatModule({1}).build(ctx, x0, x1);
}

core::TensorValue flip_channels(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & indices) {
    const int64_t channels = input.shape.dims[1];
    const int64_t frames = input.shape.dims[2];
    const auto matrix = core::reshape_tensor(ctx, ensure_contiguous(ctx, input), core::TensorShape::from_dims({channels, frames}));
    auto gathered = core::wrap_tensor(
        ggml_get_rows(ctx.ggml, matrix.tensor, indices.tensor),
        core::TensorShape::from_dims({channels, frames}),
        input.type);
    return core::reshape_tensor(ctx, ensure_contiguous(ctx, gathered), input.shape);
}

core::TensorValue flow_reverse(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const WeightMap & weights,
    const InflectConfig & config,
    GraphUploads & uploads) {
    auto x = input;
    auto index_values = std::vector<int32_t>{};
    index_values.reserve(static_cast<size_t>(config.inter_channels));
    for (int64_t channel = config.inter_channels - 1; channel >= 0; --channel) {
        index_values.push_back(static_cast<int32_t>(channel));
    }
    const auto reverse_indices = make_i32_input(
        ctx,
        core::TensorShape::from_dims({config.inter_channels}),
        std::move(index_values),
        uploads);
    for (int64_t flow = 3; flow >= 0; --flow) {
        x = flip_channels(ctx, x, reverse_indices);
        x = residual_coupling_reverse(ctx, x, weights, flow * 2, config);
    }
    return x;
}

ggml_tensor * named(ggml_tensor * tensor, const char * name) {
    return ggml_set_name(tensor, name);
}

ggml_tensor * contiguous_if_needed(ggml_context * ctx, ggml_tensor * tensor) {
    return core::has_backend_addressable_layout(tensor) ? tensor : ggml_cont(ctx, tensor);
}

ggml_tensor * weight_2d(
    ggml_context * ctx,
    const core::TensorValue & value,
    int64_t ne0,
    int64_t ne1,
    const std::string & name) {
    return named(ggml_reshape_2d(ctx, value.tensor, ne0, ne1), name.c_str());
}

ggml_tensor * weight_3d(
    ggml_context * ctx,
    const core::TensorValue & value,
    int64_t ne0,
    int64_t ne1,
    int64_t ne2,
    const std::string & name) {
    return named(ggml_reshape_3d(ctx, value.tensor, ne0, ne1, ne2), name.c_str());
}

ggml_tensor * conv1d_2d(
    ggml_context * ctx,
    ggml_tensor * x,
    const WeightMap & weights,
    const std::string & prefix,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel,
    int padding,
    int dilation,
    bool bias = true) {
    if (kernel == 1 && padding == 0 && dilation == 1) {
        auto * weight = weight_2d(ctx, weights.at(prefix + ".weight"), in_channels, out_channels, prefix + ".weight");
        auto * x_cf = contiguous_if_needed(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
        auto * y_cf = ggml_mul_mat(ctx, weight, x_cf);
        auto * y = ggml_reshape_2d(
            ctx,
            contiguous_if_needed(ctx, ggml_permute(ctx, y_cf, 1, 0, 2, 3)),
            x->ne[0],
            out_channels);
        if (bias) {
            y = ggml_add(ctx, y, weight_2d(ctx, weights.at(prefix + ".bias"), 1, out_channels, prefix + ".bias"));
        }
        return named(y, prefix.c_str());
    }
    auto * weight = weight_3d(ctx, weights.at(prefix + ".weight"), kernel, in_channels, out_channels, prefix + ".weight");
    auto * x3 = ggml_reshape_3d(ctx, contiguous_if_needed(ctx, x), x->ne[0], x->ne[1], 1);
    auto * y3 = ggml_conv_1d(ctx, weight, x3, 1, padding, dilation);
    auto * y = ggml_reshape_2d(ctx, y3, y3->ne[0], y3->ne[1]);
    if (bias) {
        y = ggml_add(ctx, y, weight_2d(ctx, weights.at(prefix + ".bias"), 1, out_channels, prefix + ".bias"));
    }
    return named(y, prefix.c_str());
}

ggml_tensor * conv_transpose1d_2d(
    core::ModuleBuildContext & ctx,
    ggml_tensor * x,
    const WeightMap & weights,
    const std::string & prefix,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel,
    int stride,
    int padding) {
    if (ctx.backend_type == core::BackendType::Cpu && padding > 0) {
        auto * input = contiguous_if_needed(ctx.ggml, x);
        auto * weight = weights.at(prefix + ".weight").tensor;
        if (weight->type != GGML_TYPE_F32) {
            weight = ggml_cast(ctx.ggml, weight, GGML_TYPE_F32);
        }
        const int64_t full_frames = (x->ne[0] - 1) * stride + kernel;
        auto * output = ggml_conv_transpose_1d(ctx.ggml, weight, input, stride, 0, 1);
        output = ggml_reshape_2d(ctx.ggml, output, full_frames, out_channels);
        output = ggml_view_2d(
            ctx.ggml,
            output,
            full_frames - 2 * padding,
            out_channels,
            ggml_row_size(output->type, full_frames),
            ggml_row_size(output->type, padding));
        output = contiguous_if_needed(ctx.ggml, output);
        output = ggml_add(ctx.ggml, output, weight_2d(ctx.ggml, weights.at(prefix + ".bias"), 1, out_channels, prefix + ".bias"));
        return named(output, prefix.c_str());
    }
    const auto input = core::wrap_tensor(
        ggml_reshape_3d(ctx.ggml, contiguous_if_needed(ctx.ggml, x), x->ne[0], in_channels, 1),
        core::TensorShape::from_dims({1, in_channels, x->ne[0]}),
        GGML_TYPE_F32);
    const auto output = modules::ConvTranspose1dModule({
        in_channels,
        out_channels,
        kernel,
        stride,
        padding,
        1,
        true,
    }).build(ctx, input, {weights.at(prefix + ".weight"), weights.at(prefix + ".bias")});
    return named(
        ggml_reshape_2d(ctx.ggml, output.tensor, output.shape.dims[2], out_channels),
        prefix.c_str());
}

ggml_tensor * decoder_resblock_2d(
    ggml_context * ctx,
    ggml_tensor * input,
    const WeightMap & weights,
    int64_t index,
    int64_t channels,
    int64_t kernel) {
    auto * x = input;
    const int dilations[] = {1, 3, 5};
    for (int64_t layer = 0; layer < 3; ++layer) {
        auto * y = ggml_leaky_relu(ctx, contiguous_if_needed(ctx, x), 0.1f, false);
        y = conv1d_2d(
            ctx,
            y,
            weights,
            "dec.resblocks." + std::to_string(index) + ".convs1." + std::to_string(layer),
            channels,
            channels,
            kernel,
            static_cast<int>((kernel * dilations[layer] - dilations[layer]) / 2),
            dilations[layer]);
        y = ggml_leaky_relu(ctx, contiguous_if_needed(ctx, y), 0.1f, false);
        y = conv1d_2d(
            ctx,
            y,
            weights,
            "dec.resblocks." + std::to_string(index) + ".convs2." + std::to_string(layer),
            channels,
            channels,
            kernel,
            static_cast<int>((kernel - 1) / 2),
            1);
        x = ggml_add(ctx, x, y);
    }
    return x;
}

core::TensorValue decoder(
    core::ModuleBuildContext & ctx,
    ggml_tensor * input,
    const WeightMap & weights,
    const InflectConfig & config) {
    auto * x = conv1d_2d(ctx.ggml, input, weights, "dec.conv_pre", config.inter_channels, 320, 7, 3, 1);
    const int channels[] = {160, 80, 40, 20};
    const int in_channels[] = {320, 160, 80, 40};
    const int strides[] = {8, 8, 2, 2};
    const int kernels[] = {16, 16, 4, 4};
    const int res_kernels[] = {3, 7, 11};
    for (int stage = 0; stage < 4; ++stage) {
        x = ggml_leaky_relu(ctx.ggml, contiguous_if_needed(ctx.ggml, x), 0.1f, false);
        x = conv_transpose1d_2d(
            ctx,
            x,
            weights,
            "dec.ups." + std::to_string(stage),
            in_channels[stage],
            channels[stage],
            kernels[stage],
            strides[stage],
            (kernels[stage] - strides[stage]) / 2);
        ggml_tensor * sum = nullptr;
        for (int block = 0; block < 3; ++block) {
            auto * y = decoder_resblock_2d(ctx.ggml, x, weights, stage * 3 + block, channels[stage], res_kernels[block]);
            sum = sum == nullptr ? y : ggml_add(ctx.ggml, sum, y);
        }
        x = ggml_scale(ctx.ggml, sum, 1.0f / 3.0f);
    }
    x = ggml_leaky_relu(ctx.ggml, contiguous_if_needed(ctx.ggml, x), 0.01f, false);
    x = conv1d_2d(ctx.ggml, x, weights, "dec.conv_post", 20, 1, 7, 3, 1, false);
    x = ggml_tanh(ctx.ggml, x);
    x = contiguous_if_needed(ctx.ggml, x);
    return core::wrap_tensor(
        ggml_reshape_3d(ctx.ggml, x, x->ne[0], 1, 1),
        core::TensorShape::from_dims({1, 1, x->ne[0]}),
        GGML_TYPE_F32);
}

core::TensorValue vocoder_graph_output(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & z,
    const WeightMap & weights,
    const InflectConfig & config,
    GraphUploads & uploads) {
    auto x_value = flow_reverse(ctx, z, weights, config, uploads);
    auto * x = ggml_reshape_2d(
        ctx.ggml,
        contiguous_if_needed(ctx.ggml, x_value.tensor),
        x_value.shape.dims[2],
        x_value.shape.dims[1]);
    return decoder(ctx, x, weights, config);
}

std::vector<float> get_tensor_f32(ggml_tensor * tensor) {
    std::vector<float> values(static_cast<size_t>(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
    return values;
}

void set_tensor_i32(ggml_tensor * tensor, const std::vector<int32_t> & values) {
    ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(int32_t));
}

void set_tensor_f32(ggml_tensor * tensor, const std::vector<float> & values) {
    ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(float));
}

std::vector<float> expand_and_sample(
    const std::vector<float> & prior,
    int64_t text_frames,
    const InflectConfig & config,
    const InflectSynthesisOptions & options,
    int64_t & latent_frames) {
    const int64_t channels = config.inter_channels;
    const int64_t row_stride = text_frames;
    std::vector<int64_t> durations(static_cast<size_t>(text_frames));
    latent_frames = 0;
    const float length_scale = 1.0f / options.speed;
    for (int64_t t = 0; t < text_frames; ++t) {
        const float logw = prior[static_cast<size_t>((channels * 2) * row_stride + t)];
        const int64_t duration = std::max<int64_t>(0, static_cast<int64_t>(std::ceil(std::exp(logw) * length_scale)));
        durations[static_cast<size_t>(t)] = duration;
        latent_frames += duration;
    }
    latent_frames = std::max<int64_t>(1, latent_frames);
    latent_frames = std::min<int64_t>(latent_frames, config.max_len);
    std::mt19937 rng(options.seed == 0 ? std::random_device{}() : options.seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> z(static_cast<size_t>(channels * latent_frames), 0.0f);
    int64_t out_t = 0;
    for (int64_t t = 0; t < text_frames && out_t < latent_frames; ++t) {
        for (int64_t repeat = 0; repeat < durations[static_cast<size_t>(t)] && out_t < latent_frames; ++repeat, ++out_t) {
            for (int64_t c = 0; c < channels; ++c) {
                const float mean = prior[static_cast<size_t>(c * row_stride + t)];
                const float logs = prior[static_cast<size_t>((channels + c) * row_stride + t)];
                z[static_cast<size_t>(c * latent_frames + out_t)] =
                    mean + normal(rng) * std::exp(logs) * options.variation;
            }
        }
    }
    return z;
}

}  // namespace

class InflectRuntime::Impl {
public:
    Impl(
        std::shared_ptr<const InflectAssets> assets,
        core::ExecutionContext & execution_context,
        assets::TensorStorageType weight_type)
        : assets_(std::move(assets)),
          execution_context_(execution_context),
          weights_(*assets_->weights, execution_context_, weight_type),
          prior_runner_(execution_context_),
          vocoder_runner_(execution_context_) {}

    runtime::AudioBuffer synthesize(const std::string & text, const InflectSynthesisOptions & options) {
        auto chunk_audio = synthesize_chunk(text, options);
        runtime::AudioBuffer output;
        output.sample_rate = assets_->config.sample_rate;
        output.channels = 1;
        output.samples = std::move(chunk_audio);
        return output;
    }

private:
    std::vector<float> synthesize_chunk(const std::string & text, const InflectSynthesisOptions & options) {
        const auto ids = encode_inflect_text(text, *assets_->english_g2p);
        const int64_t text_frames = static_cast<int64_t>(ids.size());
        std::unique_ptr<ggml_context, GgmlContextDeleter> prior_ctx;
        core::ModuleBuildContext prior_build{};
        core::TensorValue ids_value;
        GraphUploads uploads;
        core::TensorValue prior;
        const double prior_build_ms = engine::debug::measure_ms([&]() {
            prior_ctx = make_context();
            prior_build = core::ModuleBuildContext{prior_ctx.get(), "inflect.prior", execution_context_.backend_type()};
            ids_value = core::make_tensor(prior_build, GGML_TYPE_I32, core::TensorShape::from_dims({text_frames}));
            prior = prior_graph_output(prior_build, ids_value, weights_, assets_->config, uploads);
        });
        ggml_cgraph * prior_graph = nullptr;
        const double prior_alloc_ms = engine::debug::measure_ms([&]() {
            prior_graph = prior_runner_.allocate(prior_ctx.get(), prior.tensor);
        });
        const double prior_upload_ms = engine::debug::measure_ms([&]() {
            set_tensor_i32(ids_value.tensor, ids);
            for (const auto & upload : uploads.f32) {
                set_tensor_f32(upload.first, upload.second);
            }
        });
        const double prior_ms = engine::debug::measure_ms([&]() {
            prior_runner_.compute(prior_graph, "inflect_tts.prior");
        });
        int64_t latent_frames = 0;
        double prior_read_ms = 0.0;
        double sample_ms = 0.0;
        auto z = std::vector<float>{};
        prior_read_ms = engine::debug::measure_ms([&]() {
            auto prior_values = get_tensor_f32(prior.tensor);
            sample_ms = engine::debug::measure_ms([&]() {
                z = expand_and_sample(prior_values, text_frames, assets_->config, options, latent_frames);
            });
        });

        std::unique_ptr<ggml_context, GgmlContextDeleter> vocoder_ctx;
        core::ModuleBuildContext vocoder_build{};
        GraphUploads vocoder_uploads;
        core::TensorValue z_value;
        core::TensorValue audio;
        const double vocoder_build_ms = engine::debug::measure_ms([&]() {
            vocoder_ctx = make_context(256ull * 1024ull * 1024ull);
            vocoder_build = core::ModuleBuildContext{
                vocoder_ctx.get(), "inflect.vocoder", execution_context_.backend_type()};
            z_value = core::make_tensor(
                vocoder_build,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({1, assets_->config.inter_channels, latent_frames}));
            audio = vocoder_graph_output(vocoder_build, z_value, weights_, assets_->config, vocoder_uploads);
        });
        ggml_cgraph * vocoder_graph = nullptr;
        const double vocoder_alloc_ms = engine::debug::measure_ms([&]() {
            vocoder_graph = vocoder_runner_.allocate(vocoder_ctx.get(), audio.tensor);
        });
        const double z_upload_ms = engine::debug::measure_ms([&]() {
            set_tensor_f32(z_value.tensor, z);
            for (const auto & upload : vocoder_uploads.i32) {
                set_tensor_i32(upload.first, upload.second);
            }
        });
        const double vocoder_ms = engine::debug::measure_ms([&]() {
            vocoder_runner_.compute(vocoder_graph, "inflect_tts.vocoder");
        });
        std::vector<float> samples;
        const double audio_read_ms = engine::debug::measure_ms([&]() {
            samples = get_tensor_f32(audio.tensor);
        });
        engine::debug::timing_log_scalar("inflect_tts.prior_build_ms", prior_build_ms);
        engine::debug::timing_log_scalar("inflect_tts.prior_alloc_ms", prior_alloc_ms);
        engine::debug::timing_log_scalar("inflect_tts.prior_upload_ms", prior_upload_ms);
        engine::debug::timing_log_scalar("inflect_tts.prior_compute_ms", prior_ms);
        engine::debug::timing_log_scalar("inflect_tts.prior_read_sample_ms", prior_read_ms);
        engine::debug::timing_log_scalar("inflect_tts.sample_ms", sample_ms);
        engine::debug::timing_log_scalar("inflect_tts.vocoder_build_ms", vocoder_build_ms);
        engine::debug::timing_log_scalar("inflect_tts.vocoder_alloc_ms", vocoder_alloc_ms);
        engine::debug::timing_log_scalar("inflect_tts.z_upload_ms", z_upload_ms);
        engine::debug::timing_log_scalar("inflect_tts.vocoder_compute_ms", vocoder_ms);
        engine::debug::timing_log_scalar("inflect_tts.audio_read_ms", audio_read_ms);
        return samples;
    }

    std::shared_ptr<const InflectAssets> assets_;
    core::ExecutionContext & execution_context_;
    WeightMap weights_;
    GraphRunner prior_runner_;
    GraphRunner vocoder_runner_;
};

InflectRuntime::InflectRuntime(
    std::shared_ptr<const InflectAssets> assets,
    engine::core::ExecutionContext & execution_context,
    assets::TensorStorageType weight_type)
    : impl_(std::make_shared<Impl>(std::move(assets), execution_context, weight_type)) {}

runtime::AudioBuffer InflectRuntime::synthesize(
    const std::string & text,
    const InflectSynthesisOptions & options) {
    const auto start = Clock::now();
    auto audio = impl_->synthesize(text, options);
    engine::debug::timing_log_scalar("inflect_tts.generate_ms", engine::debug::elapsed_ms(start));
    return audio;
}

}  // namespace engine::models::inflect_tts
