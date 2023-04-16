#pragma once
#include <core/basic_traits.h>
namespace lc::hlsl {
using namespace luisa;
enum class ShaderVariableType : uint8_t {
    ConstantBuffer,
    SRVDescriptorHeap,
    UAVDescriptorHeap,
    CBVDescriptorHeap,
    SampDescriptorHeap,
    StructuredBuffer,
    RWStructuredBuffer,
    ConstantValue
};
struct Property {
    ShaderVariableType type;
    uint spaceIndex;
    uint registerIndex;
    uint arrSize;
};
}// namespace lc::hlsl