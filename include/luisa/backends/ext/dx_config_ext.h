#pragma once

#include <luisa/runtime/context.h>
#include <d3d12.h>
#include <dxgi1_2.h>
#include <luisa/vstl/common.h>
#include <luisa/runtime/rhi/device_interface.h>

namespace luisa::compute {
struct DirectXHeap {
    uint64_t handle;
    ID3D12Heap *heap;
    size_t offset;
};
class DirectXAllocator {
public:
    [[nodiscard]] virtual DirectXHeap AllocateBufferHeap(
        luisa::string_view name,
        uint64_t sizeBytes,
        D3D12_HEAP_TYPE heapType,
        D3D12_HEAP_FLAGS extraFlags) const noexcept = 0;
    [[nodiscard]] virtual DirectXHeap AllocateTextureHeap(
        vstd::string_view name,
        size_t sizeBytes,
        bool isRenderTexture,
        D3D12_HEAP_FLAGS extraFlags) const noexcept = 0;
    [[nodiscard]] virtual void DeAllocateHeap(uint64_t handle) const noexcept = 0;
};
struct DirectXDeviceConfigExt : public DeviceConfigExt, public vstd::IOperatorNewBase {

    struct ExternalDevice {
        ID3D12Device *device;
        IDXGIAdapter1 *adapter;
        IDXGIFactory2 *factory;
    };

    virtual vstd::optional<ExternalDevice> CreateExternalDevice() noexcept { return {}; }
    // Called during create_device
    virtual void ReadbackDX12Device(
        ID3D12Device *device,
        IDXGIAdapter1 *adapter,
        IDXGIFactory2 *factory,
        DirectXAllocator const*allocator,
        ID3D12DescriptorHeap *shaderDescriptor,
        ID3D12DescriptorHeap *samplerDescriptor) noexcept {}

    // plugin resources
    virtual ID3D12CommandQueue *CreateQueue(D3D12_COMMAND_LIST_TYPE type) noexcept { return nullptr; }

    virtual ID3D12GraphicsCommandList *BorrowCommandList(D3D12_COMMAND_LIST_TYPE type) noexcept { return nullptr; }

    // Custom callback
    // return true if this callback is implemented
    virtual bool ExecuteCommandList(
        ID3D12CommandQueue *queue,
        ID3D12GraphicsCommandList *cmdList) noexcept { return false; }

    virtual bool SignalFence(
        ID3D12CommandQueue *queue,
        ID3D12Fence *fence, uint64_t fenceIndex) noexcept { return false; }

    ~DirectXDeviceConfigExt() noexcept override = default;
};

}// namespace luisa::compute
