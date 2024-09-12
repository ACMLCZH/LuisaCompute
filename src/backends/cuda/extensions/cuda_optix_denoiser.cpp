#include "cuda_optix_denoiser.h"
#include <luisa/core/logging.h>
namespace luisa::compute {

OptixDenoiser::OptixDenoiser(CUDADevice *device, CUDAStream *stream) noexcept
    : _device(device), _stream(stream) {}

OptixDenoiser::~OptixDenoiser() noexcept { reset(); }

optix::Image2D OptixDenoiser::build_Image2D(const DenoiserExt::Image &img) noexcept {
    optix::Image2D image;
    image.data = reinterpret_cast<CUDABuffer *>(img.buffer_handle)->device_address();
    image.width = img.width;
    image.height = img.height;
    image.pixelStrideInBytes = img.pixel_stride;
    image.rowStrideInBytes = img.row_stride;
    image.format = get_format(img.format);
    return image;
};

optix::Image2D OptixDenoiser::create_Image2D(const DenoiserExt::Image &img) noexcept {
    optix::Image2D image;
    LUISA_CHECK_CUDA(cuMemAllocAsync(&image.data, img.size_bytes, _stream->handle()));
    image.width = img.width;
    image.height = img.height;
    image.pixelStrideInBytes = img.pixel_stride;
    image.rowStrideInBytes = img.row_stride;
    image.format = get_format(img.format);
    return image;
}

optix::Image2D OptixDenoiser::create_internal(const DenoiserExt::Image &img, const optix::DenoiserSizes &denoiser_sizes) {
    optix::Image2D image;
    unsigned int pixel_stride = denoiser_sizes.internalGuideLayerPixelSizeInBytes;
    LUISA_CHECK_CUDA(cuMemAllocAsync(
        &image.data, pixel_stride * img.width * img.height, _stream->handle()));
    image.width = img.width;
    image.height = img.height;
    image.pixelStrideInBytes = pixel_stride;
    image.rowStrideInBytes = pixel_stride * img.width;
    image.format = optix::PIXEL_FORMAT_INTERNAL_GUIDE_LAYER;
    return image;
};

optix::DenoiserModelKind OptixDenoiser::get_model_kind() noexcept {
    switch ((_has_aov ? 1u : 0u) | (_has_upscale ? 2u : 0u) | (_has_temporal ? 4u : 0u)) {
        case 0: return optix::DenoiserModelKind::DENOISER_MODEL_KIND_HDR;
        case 1: return optix::DenoiserModelKind::DENOISER_MODEL_KIND_AOV;
        case 2: return optix::DenoiserModelKind::DENOISER_MODEL_KIND_UPSCALE2X;
        case 4: return optix::DenoiserModelKind::DENOISER_MODEL_KIND_TEMPORAL;
        case 5: return optix::DenoiserModelKind::DENOISER_MODEL_KIND_TEMPORAL_AOV;
        case 6: return optix::DenoiserModelKind::DENOISER_MODEL_KIND_TEMPORAL_UPSCALE2X;
    }
};

void OptixDenoiser::reset() noexcept {
    LUISA_CHECK_OPTIX(optix::api().denoiserDestroy(_denoiser));
    LUISA_CHECK_CUDA(cuMemFreeAsync(_params.hdrIntensity, _stream->handle()));
    LUISA_CHECK_CUDA(cuMemFreeAsync(_params.hdrAverageColor, _stream->handle()));
    LUISA_CHECK_CUDA(cuMemFreeAsync(_scratch, _stream->handle()));
    LUISA_CHECK_CUDA(cuMemFreeAsync(_state, _stream->handle()));
    LUISA_CHECK_CUDA(cuMemFreeAsync(_guideLayer.previousOutputInternalGuideLayer.data, _stream->handle()));
    LUISA_CHECK_CUDA(cuMemFreeAsync(_guideLayer.outputInternalGuideLayer.data, _stream->handle()));
    for (auto &l: _layers) {
        if (l.output.data != l.previousOutput.data) {
            LUISA_CHECK_CUDA(cuMemFreeAsync(l.previousOutput.data, _stream->handle()));
        }
    }
    _denoier = nullptr;
    _params = {};
    _layers = {};
    _guideLayer = {};
    _scratch = 0;
    _state = 0;
}

void OptixDenoiser::init(const DenoiserExt::DenoiserInput &input) noexcept {
    LUISA_ASSERT(input.layers.size() > 0, "input is empty!");
    
    auto get_format = [](DenoiserExt::ImageFormat fmt) noexcept {
        switch (fmt) {
            case FLOAT1: return optix::PIXEL_FORMAT_FLOAT1;
            case FLOAT2: return optix::PIXEL_FORMAT_FLOAT2;
            case FLOAT3: return optix::PIXEL_FORMAT_FLOAT3;
            case FLOAT4: return optix::PIXEL_FORMAT_FLOAT4;
            case HALF1: return optix::PIXEL_FORMAT_HALF1;
            case HALF2: return optix::PIXEL_FORMAT_HALF2;
            case HALF3: return optix::PIXEL_FORMAT_HALF3;
            case HALF4: return optix::PIXEL_FORMAT_HALF4;
            default: LUISA_ERROR_WITH_LOCATION("Invalid image format: {}.", (int)fmt);
        }
    };

    auto get_aov_type = [](DenoiserExt::ImageAOVType type) noexcept {
        switch (type) {
            case BEAUTY: return optix::DenoiserAOVType::DENOISER_AOV_TYPE_BEAUTY;
            case DIFFUSE: return optix::DenoiserAOVType::DENOISER_AOV_TYPE_DIFFUSE;
            case SPECULAR: return optix::DenoiserAOVType::DENOISER_AOV_TYPE_SPECULAR;
            case REFLECTION: return optix::DenoiserAOVType::DENOISER_AOV_TYPE_REFLECTION;
            case REFRACTION: return optix::DenoiserAOVType::DENOISER_AOV_TYPE_REFRACTION;
        }
    };

    reset();
    auto optix_ctx = _device->handle().optix_context();
    _has_aov = input.layers.size() > 1;
    _has_upscale = input.upscale;
    _has_temporal = input.temporal;
    auto out_scale = input.upscale ? 2u : 1u;
    optix::DenoiserModelKind model_kind = get_model_kind();
    optix::DenoiserOptions options = {};
    if (input.prefilter_mode != DenoiserExt::PrefilterMode::NONE) {
        bool guide_flow = false;
        bool guide_flowtrust = false;
        for (auto &f : input.features) {
            if (f.type == DenoiserExt::ImageFeatureType::ALBEDO) {
                LUISA_ASSERT(!options.guideAlbedo, "Albedo feature already set.");
                options.guideAlbedo = true;
                _guideLayer.albedo = build_Image2D(f.image);
            } else if (f.type == DenoiserExt::ImageFeatureType::NORMAL) {
                LUISA_ASSERT(!options.guideNormal, "Normal feature already set.");
                options.guideNormal = true;
                _guideLayer.normal = build_Image2D(f.image);
            } else if (f.type == DenoiserExt::ImageFeatureType::FLOW && input.temporal) {
                LUISA_ASSERT(!guide_flow, "Flow feature already set.");
                _guideLayer.flow = build_Image2D(f.image);
                guide_flow = true;
            } else if (f.type == DenoiserExt::ImageFeatureType::FLOWTRUST && input.temporal) {
                LUISA_ASSERT(!guide_flowtrust, "Flow trust feature already set.");
                _guideLayer.flowTrustworthiness = build_Image2D(f.image);
                guide_flowtrust = true;
            }
        }
    }
    LUISA_CHECK_OPTIX(optix::api().denoiserCreate(optix_ctx, model_kind, &options, &_denoiser));

    optix::DenoiserSizes denoiser_sizes;
    LUISA_CHECK_OPTIX(optix::api().denoiserComputeMemoryResources(
        _denoiser, input.width * out_scale, input.height * out_scale, &denoiser_sizes));
    _scratch_size = static_cast<uint32_t>(denoiser_sizes.withoutOverlapScratchSizeInBytes);
    _state_size = static_cast<uint32_t>(denoiser_sizes.stateSizeInBytes);
    _overlap = 0u;

    // denoiser params
    if (_has_aov) {
        LUISA_CHECK_CUDA(cuMemAllocAsync(
            &_params.hdrAverageColor, 3 * sizeof(float), _stream->handle()));
    } 
    if (_has_temporal) {
        LUISA_CHECK_CUDA(cuMemAllocAsync(
            &_params.hdrIntensity, sizeof(float), _stream->handle()));
        _params.temporalModeUsePreviousLayers = 1;
    }
    LUISA_CHECK_CUDA(cuMemAllocAsync(&_scratch, _scratch_size, _stream->handle()));
    LUISA_CHECK_CUDA(cuMemAllocAsync(&_state, _state_size, _stream->handle()));

    // build layers
    for (auto &l : input.layers) {
        optix::DenoiserLayer layer = {};
        layer.input = build_Image2D(l.input);
        layer.output = build_Image2D(l.output);
        if (_has_temporal) {
            if (_has_aov) {
                layer.previousOutput = create_Image2D(l.output);
                if (!_has_upscale) {          // First frame initializaton.
                    LUISA_CHECK_CUDA(cuMemcpyAsync(layer.previousOutput.data, layer.input.data, l.input.size_bytes, _stream->handle()));
                    LUISA_CHECK_CUDA(cuMemcpyAsync(layer.output.data, layer.input.data, l.input.size_bytes, _stream->handle()));
                }
            } else {
                layer.previousOutput = build_Image2D(l.output);
                if (!_has_upscale) {
                    LUISA_CHECK_CUDA(cuMemcpyAsync(layer.previousOutput.data, layer.input.data, l.input.size_bytes, _stream->handle()));
                }
            }
        }
        layer.type = get_aov_type(l.aov_type);
        _layers.push_back(layer);
    }
    if (_has_temporal && _has_aov) {
        _guideLayer.previousOutputInternalGuideLayer = create_internal(input.layers[0].output, denoiser_sizes);
        _guideLayer.outputInternalGuideLayer = create_internal(input.layers[0].output, denoiser_sizes);
    }

    LUISA_CHECK_OPTIX(optix::api().denoiserSetup(
        _denoiser, _stream->handle(),
        input.width + 2 * _overlap, input.height + 2 * _overlap,
        _state, _state_size, _scratch, _scratch_size
    ));
}

    // auto cuda_stream = reinterpret_cast<CUDAStream *>(stream.handle())->handle();
    // auto optix_ctx = _device->handle().optix_context();
    // optix::DenoiserParams _params = {};
    //_params.denoiseAlpha = _mode.alphamode ? optix::DENOISER_ALPHA_MODE_ALPHA_AS_AOV : optix::DENOISER_ALPHA_MODE_COPY;
    // _params.hdrIntensity = _intensity;
    // _params.hdrAverageColor = _avg_color;
    // _params.blendFactor = 0.0f;
    // _params.temporalModeUsePreviousLayers = 0;
    // LUISA_ASSERT(data.beauty != nullptr && *data.beauty, "input image(beauty) is invalid!");
    // _layers[0].input.data = reinterpret_cast<CUDABuffer *>(data.beauty->handle())->device_address();

    // if (_mode.temporal)
    //     _guideLayer.flow.data = reinterpret_cast<CUDABuffer *>(data.flow->handle())->device_address();

    // if (data.albedo)
    //     _guideLayer.albedo.data = reinterpret_cast<CUDABuffer *>(data.albedo->handle())->device_address();

    // if (data.normal)
    //     _guideLayer.normal.data = reinterpret_cast<CUDABuffer *>(data.normal->handle())->device_address();
    // for (size_t i = 0; i < data.aov_size; i++)
    //     _layers[i + 1].input.data = reinterpret_cast<CUDABuffer *>(data.aovs[i]->handle())->device_address();

void OptixDenoiser::execute_denoise() noexcept {
    // Swap previous output
    if (_has_temporal && _has_aov) {
        auto &img = _layers[0].output;
        size_t img_size = img.width * img.height * img.pixelStrideInBytes;
        
        LUISA_CHECK_CUDA(cuMemcpyAsync(
            _guideLayer.previousOutputInternalGuideLayer.data,
            _guideLayer.outputInternalGuideLayer.data,
            img_size, _stream->handle()));
        for (size_t i = 0u; i < _layers.size(); ++i) {
            LUISA_CHECK_CUDA(cuMemcpyAsync(
                _layers[i].previousOutput.data, _layers[i].output.data, img_size, _stream->handle()));
        }
    }

    if (_params.hdrIntensity) {
        LUISA_CHECK_OPTIX(optix::api().denoiserComputeIntensity(
            _denoiser, _stream->handle(), &_layers[0].input,
            _params.hdrIntensity, _scratch, _scratch_size));
    }
    if (_params.hdrAverageColor) {
        LUISA_CHECK_OPTIX(optix::api().denoiserComputeAverageColor(
            _denoiser, _stream->handle(), &_layers[0].input,
            _params.hdrAverageColor, _scratch, _scratch_size));
    }
    LUISA_CHECK_OPTIX(optix::api().denoiserInvoke(
        _denoiser, _stream->handle(), &_params,
        _state, _state_size,
        &_guideLayer, _layers.data(), static_cast<unsigned int>(_layers.size()), 0, 0,
        _scratch, _scratch_size));
}

// void CUDAOldDenoiserExt::denoise(Stream &stream, uint2 resolution, Buffer<float> const &image, Buffer<float> &output,
//                                  Buffer<float> const &normal, Buffer<float> const &albedo, Buffer<float> **aovs, uint aov_size) noexcept {
//     DenoiserMode mode{};
//     mode.alphamode = 0;
//     mode.kernel_pred = 0;
//     mode.temporal = 0;
//     mode.upscale = 0;

//     DenoiserInput data{};
//     data.beauty = &image;
//     data.normal = &normal;
//     data.albedo = &albedo;
//     data.flow = nullptr;
//     data.flowtrust = nullptr;
//     data.aovs = aovs;
//     data.aov_size = aov_size;
//     _device->with_handle([&] {
//         _init(stream, mode, data, resolution);
//         _process(stream, data);
//         _get_result(stream, output, -1);
//         _destroy(stream);
//     });
// }

}// namespace luisa::compute
