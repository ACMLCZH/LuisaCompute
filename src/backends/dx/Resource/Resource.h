#pragma once
#include <DXRuntime/Device.h>
namespace lc::dx {
class Resource : public vstd::ISelfPtr {
public:
    enum class Tag : uint8_t {
        None,
        UploadBuffer,
        ReadbackBuffer,
        DefaultBuffer,
        RenderTexture,
        DescriptorHeap,
        BindlessArray,
        Mesh,
        SwapChain,
        DepthBuffer,
        ExternalBuffer,
        ExternalTexture,
        ExternalDepth
    };

protected:
    Device *device;

public:
    static uint64 GetTextureSize(
        Device *device,
        uint width,
        uint height,
        GFXFormat Format,
        TextureDimension type,
        uint depthCount,
        uint mipCount);
    static uint64 GetTexturePixelSize(
        GFXFormat format);
    static bool IsBCtex(GFXFormat format);
    Device *GetDevice() const { return device; }
    Resource(Device *device)
        : device(device) {}
    Resource(Resource &&) = default;
    Resource(Resource const &) = delete;
    virtual Tag GetTag() const = 0;
    virtual ~Resource() = default;
    virtual ID3D12Resource *GetResource() const { return nullptr; }
    virtual bool IsNonSimulResource() const { return false; }
    virtual D3D12_RESOURCE_STATES GetInitState() const { return D3D12_RESOURCE_STATE_COMMON; }
    virtual D3D12_RESOURCE_STATES GetNonSimulCurrentState() const {
        static_assert(sizeof(D3D12_RESOURCE_STATES) == 4);
        return static_cast<D3D12_RESOURCE_STATES>(std::numeric_limits<uint32_t>::max());
    }
    virtual void SetNonSimulCurrentState(D3D12_RESOURCE_STATES state) const {}
};
}// namespace lc::dx