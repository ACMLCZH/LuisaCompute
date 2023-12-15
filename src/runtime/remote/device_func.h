#pragma once
#include <stdint.h>
namespace luisa::compute {
enum class DeviceFunc : uint32_t {
    CreateBufferAst,
    CreateBufferIR,
    DestroyBuffer,
    CreateTexture,
    DestroyTexture,
    CreateBindlessArray,
    DestroyBindlessArray,
    CreateStream,
    DestroyStream,
    Dispatch,
    CreateSwapChain,
    CreateShaderAst,
    CreateShaderIR,
    CreateShaderIRV2,
    LoadShader,
    ShaderArgUsage,
    DestroyShader,
    CreateEvent,
    DestroyEvent,
    SignalEvent,
    WaitEvent,
    SyncEvent,
    CreateSwapchain,
    DestroySwapchain,
    CreateMesh,
    DestroyMesh,
    CreateProcedrualPrim,
    DestroyProcedrualPrim,
    CreateAccel,
    DestroyAccel,
    CreateSparseBuffer,
    DestroySparseBuffer,
    CreateSparseTexture,
    DestroySparseTexture,
    AllocSparseBufferHeap,
    DeAllocSparseBufferHeap,
    AllocSparseTextureHeap,
    DeAllocSparseTextureHeap,
    UpdateSparseResource,
};
}// namespace luisa::compute