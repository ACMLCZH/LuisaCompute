#include <Resource/UploadBuffer.h>
namespace lc::dx {
UploadBuffer::UploadBuffer(
    Device *device,
    uint64 byteSize,
    GpuAllocator *allocator)
    : Buffer(device),
      allocHandle(allocator),
      byteSize(byteSize) {
    if (allocator) {
        ID3D12Heap *heap;
        uint64 offset;
        allocHandle.allocateHandle = allocator->AllocateBufferHeap(
            device, "upload buffer", byteSize, D3D12_HEAP_TYPE_UPLOAD, &heap, &offset);
        auto buffer = CD3DX12_RESOURCE_DESC::Buffer(byteSize);
        ThrowIfFailed(device->device->CreatePlacedResource(
            heap, offset,
            &buffer,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&allocHandle.resource)));
    } else {
        auto prop = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
        auto buffer = CD3DX12_RESOURCE_DESC::Buffer(byteSize);
        ThrowIfFailed(device->device->CreateCommittedResource(
            &prop,
            D3D12_HEAP_FLAG_NONE,
            &buffer,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&allocHandle.resource)));
    }
}
UploadBuffer::~UploadBuffer() {
}
void UploadBuffer::CopyData(uint64 offset, vstd::span<uint8_t const> data) const {
    void *mappedPtr;
    D3D12_RANGE range;
    range.Begin = offset;
    range.End = offset + data.size();
    ThrowIfFailed(allocHandle.resource->Map(0, &range, reinterpret_cast<void **>(&mappedPtr)));
    auto disp = vstd::scope_exit([&] {
        allocHandle.resource->Unmap(0, &range);
    });
    memcpy(reinterpret_cast<uint8_t *>(mappedPtr) + offset, data.data(), data.size());
}

}// namespace lc::dx
