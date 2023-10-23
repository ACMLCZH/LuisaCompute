#pragma once

#include <type_traits>
#include <array>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/string.h>

#include <luisa/ir_v2/ir_v2_fwd.h>

#ifndef LC_IR_API
#ifdef LC_IR_EXPORT_DLL
#define LC_IR_API __declspec(dllexport)
#else
#define LC_IR_API __declspec(dllimport)
#endif
#endif

namespace luisa::compute::ir_v2 {
template<class T>
struct Slice {
    T *data = nullptr;
    size_t len = 0;
    constexpr Slice() noexcept = default;
    constexpr Slice(T *data, size_t len) noexcept : data(data), len(len) {}
#ifndef BINDGEN
    // construct from array
    template<size_t N>
    constexpr Slice(T (&arr)[N]) noexcept : data(arr), len(N) {}
    // construct from std::array
    template<size_t N>
    constexpr Slice(std::array<T, N> &arr) noexcept : data(arr.data()), len(N) {}
    // construct from luisa::vector
    constexpr Slice(luisa::vector<T> &vec) noexcept : data(vec.data()), len(vec.size()) {}
    // construct from luisa::span
    constexpr Slice(luisa::span<T> &span) noexcept : data(span.data()), len(span.size()) {}
    // construct from luisa::string
    constexpr Slice(luisa::string &str) noexcept : data(str.data()), len(str.size()) {
        static_assert(std::is_same_v<T, char> || std::is_same_v<T, const char>);
    }
    luisa::vector<T> to_vector() const noexcept {
        return luisa::vector<T>(data, data + len);
    }
    luisa::string to_string() const noexcept {
        static_assert(std::is_same_v<T, char> || std::is_same_v<T, const char>);
        return luisa::string(data, len);
    }
#endif
};
}// namespace luisa::compute::ir_v2

namespace luisa::compute::ir_v2 {
struct IrV2BindingTable {
    ZeroFn *(*Func_as_ZeroFn)(Func *self);
    OneFn *(*Func_as_OneFn)(Func *self);
    AssumeFn *(*Func_as_AssumeFn)(Func *self);
    UnreachableFn *(*Func_as_UnreachableFn)(Func *self);
    ThreadIdFn *(*Func_as_ThreadIdFn)(Func *self);
    BlockIdFn *(*Func_as_BlockIdFn)(Func *self);
    WarpSizeFn *(*Func_as_WarpSizeFn)(Func *self);
    WarpLaneIdFn *(*Func_as_WarpLaneIdFn)(Func *self);
    DispatchIdFn *(*Func_as_DispatchIdFn)(Func *self);
    DispatchSizeFn *(*Func_as_DispatchSizeFn)(Func *self);
    PropagateGradientFn *(*Func_as_PropagateGradientFn)(Func *self);
    OutputGradientFn *(*Func_as_OutputGradientFn)(Func *self);
    RequiresGradientFn *(*Func_as_RequiresGradientFn)(Func *self);
    BackwardFn *(*Func_as_BackwardFn)(Func *self);
    GradientFn *(*Func_as_GradientFn)(Func *self);
    AccGradFn *(*Func_as_AccGradFn)(Func *self);
    DetachFn *(*Func_as_DetachFn)(Func *self);
    RayTracingInstanceTransformFn *(*Func_as_RayTracingInstanceTransformFn)(Func *self);
    RayTracingInstanceVisibilityMaskFn *(*Func_as_RayTracingInstanceVisibilityMaskFn)(Func *self);
    RayTracingInstanceUserIdFn *(*Func_as_RayTracingInstanceUserIdFn)(Func *self);
    RayTracingSetInstanceTransformFn *(*Func_as_RayTracingSetInstanceTransformFn)(Func *self);
    RayTracingSetInstanceOpacityFn *(*Func_as_RayTracingSetInstanceOpacityFn)(Func *self);
    RayTracingSetInstanceVisibilityFn *(*Func_as_RayTracingSetInstanceVisibilityFn)(Func *self);
    RayTracingSetInstanceUserIdFn *(*Func_as_RayTracingSetInstanceUserIdFn)(Func *self);
    RayTracingTraceClosestFn *(*Func_as_RayTracingTraceClosestFn)(Func *self);
    RayTracingTraceAnyFn *(*Func_as_RayTracingTraceAnyFn)(Func *self);
    RayTracingQueryAllFn *(*Func_as_RayTracingQueryAllFn)(Func *self);
    RayTracingQueryAnyFn *(*Func_as_RayTracingQueryAnyFn)(Func *self);
    RayQueryWorldSpaceRayFn *(*Func_as_RayQueryWorldSpaceRayFn)(Func *self);
    RayQueryProceduralCandidateHitFn *(*Func_as_RayQueryProceduralCandidateHitFn)(Func *self);
    RayQueryTriangleCandidateHitFn *(*Func_as_RayQueryTriangleCandidateHitFn)(Func *self);
    RayQueryCommittedHitFn *(*Func_as_RayQueryCommittedHitFn)(Func *self);
    RayQueryCommitTriangleFn *(*Func_as_RayQueryCommitTriangleFn)(Func *self);
    RayQueryCommitdProceduralFn *(*Func_as_RayQueryCommitdProceduralFn)(Func *self);
    RayQueryTerminateFn *(*Func_as_RayQueryTerminateFn)(Func *self);
    LoadFn *(*Func_as_LoadFn)(Func *self);
    CastFn *(*Func_as_CastFn)(Func *self);
    BitCastFn *(*Func_as_BitCastFn)(Func *self);
    AddFn *(*Func_as_AddFn)(Func *self);
    SubFn *(*Func_as_SubFn)(Func *self);
    MulFn *(*Func_as_MulFn)(Func *self);
    DivFn *(*Func_as_DivFn)(Func *self);
    RemFn *(*Func_as_RemFn)(Func *self);
    BitAndFn *(*Func_as_BitAndFn)(Func *self);
    BitOrFn *(*Func_as_BitOrFn)(Func *self);
    BitXorFn *(*Func_as_BitXorFn)(Func *self);
    ShlFn *(*Func_as_ShlFn)(Func *self);
    ShrFn *(*Func_as_ShrFn)(Func *self);
    RotRightFn *(*Func_as_RotRightFn)(Func *self);
    RotLeftFn *(*Func_as_RotLeftFn)(Func *self);
    EqFn *(*Func_as_EqFn)(Func *self);
    NeFn *(*Func_as_NeFn)(Func *self);
    LtFn *(*Func_as_LtFn)(Func *self);
    LeFn *(*Func_as_LeFn)(Func *self);
    GtFn *(*Func_as_GtFn)(Func *self);
    GeFn *(*Func_as_GeFn)(Func *self);
    MatCompMulFn *(*Func_as_MatCompMulFn)(Func *self);
    NegFn *(*Func_as_NegFn)(Func *self);
    NotFn *(*Func_as_NotFn)(Func *self);
    BitNotFn *(*Func_as_BitNotFn)(Func *self);
    AllFn *(*Func_as_AllFn)(Func *self);
    AnyFn *(*Func_as_AnyFn)(Func *self);
    SelectFn *(*Func_as_SelectFn)(Func *self);
    ClampFn *(*Func_as_ClampFn)(Func *self);
    LerpFn *(*Func_as_LerpFn)(Func *self);
    StepFn *(*Func_as_StepFn)(Func *self);
    SaturateFn *(*Func_as_SaturateFn)(Func *self);
    SmoothStepFn *(*Func_as_SmoothStepFn)(Func *self);
    AbsFn *(*Func_as_AbsFn)(Func *self);
    MinFn *(*Func_as_MinFn)(Func *self);
    MaxFn *(*Func_as_MaxFn)(Func *self);
    ReduceSumFn *(*Func_as_ReduceSumFn)(Func *self);
    ReduceProdFn *(*Func_as_ReduceProdFn)(Func *self);
    ReduceMinFn *(*Func_as_ReduceMinFn)(Func *self);
    ReduceMaxFn *(*Func_as_ReduceMaxFn)(Func *self);
    ClzFn *(*Func_as_ClzFn)(Func *self);
    CtzFn *(*Func_as_CtzFn)(Func *self);
    PopCountFn *(*Func_as_PopCountFn)(Func *self);
    ReverseFn *(*Func_as_ReverseFn)(Func *self);
    IsInfFn *(*Func_as_IsInfFn)(Func *self);
    IsNanFn *(*Func_as_IsNanFn)(Func *self);
    AcosFn *(*Func_as_AcosFn)(Func *self);
    AcoshFn *(*Func_as_AcoshFn)(Func *self);
    AsinFn *(*Func_as_AsinFn)(Func *self);
    AsinhFn *(*Func_as_AsinhFn)(Func *self);
    AtanFn *(*Func_as_AtanFn)(Func *self);
    Atan2Fn *(*Func_as_Atan2Fn)(Func *self);
    AtanhFn *(*Func_as_AtanhFn)(Func *self);
    CosFn *(*Func_as_CosFn)(Func *self);
    CoshFn *(*Func_as_CoshFn)(Func *self);
    SinFn *(*Func_as_SinFn)(Func *self);
    SinhFn *(*Func_as_SinhFn)(Func *self);
    TanFn *(*Func_as_TanFn)(Func *self);
    TanhFn *(*Func_as_TanhFn)(Func *self);
    ExpFn *(*Func_as_ExpFn)(Func *self);
    Exp2Fn *(*Func_as_Exp2Fn)(Func *self);
    Exp10Fn *(*Func_as_Exp10Fn)(Func *self);
    LogFn *(*Func_as_LogFn)(Func *self);
    Log2Fn *(*Func_as_Log2Fn)(Func *self);
    Log10Fn *(*Func_as_Log10Fn)(Func *self);
    PowiFn *(*Func_as_PowiFn)(Func *self);
    PowfFn *(*Func_as_PowfFn)(Func *self);
    SqrtFn *(*Func_as_SqrtFn)(Func *self);
    RsqrtFn *(*Func_as_RsqrtFn)(Func *self);
    CeilFn *(*Func_as_CeilFn)(Func *self);
    FloorFn *(*Func_as_FloorFn)(Func *self);
    FractFn *(*Func_as_FractFn)(Func *self);
    TruncFn *(*Func_as_TruncFn)(Func *self);
    RoundFn *(*Func_as_RoundFn)(Func *self);
    FmaFn *(*Func_as_FmaFn)(Func *self);
    CopysignFn *(*Func_as_CopysignFn)(Func *self);
    CrossFn *(*Func_as_CrossFn)(Func *self);
    DotFn *(*Func_as_DotFn)(Func *self);
    OuterProductFn *(*Func_as_OuterProductFn)(Func *self);
    LengthFn *(*Func_as_LengthFn)(Func *self);
    LengthSquaredFn *(*Func_as_LengthSquaredFn)(Func *self);
    NormalizeFn *(*Func_as_NormalizeFn)(Func *self);
    FaceforwardFn *(*Func_as_FaceforwardFn)(Func *self);
    DistanceFn *(*Func_as_DistanceFn)(Func *self);
    ReflectFn *(*Func_as_ReflectFn)(Func *self);
    DeterminantFn *(*Func_as_DeterminantFn)(Func *self);
    TransposeFn *(*Func_as_TransposeFn)(Func *self);
    InverseFn *(*Func_as_InverseFn)(Func *self);
    WarpIsFirstActiveLaneFn *(*Func_as_WarpIsFirstActiveLaneFn)(Func *self);
    WarpFirstActiveLaneFn *(*Func_as_WarpFirstActiveLaneFn)(Func *self);
    WarpActiveAllEqualFn *(*Func_as_WarpActiveAllEqualFn)(Func *self);
    WarpActiveBitAndFn *(*Func_as_WarpActiveBitAndFn)(Func *self);
    WarpActiveBitOrFn *(*Func_as_WarpActiveBitOrFn)(Func *self);
    WarpActiveBitXorFn *(*Func_as_WarpActiveBitXorFn)(Func *self);
    WarpActiveCountBitsFn *(*Func_as_WarpActiveCountBitsFn)(Func *self);
    WarpActiveMaxFn *(*Func_as_WarpActiveMaxFn)(Func *self);
    WarpActiveMinFn *(*Func_as_WarpActiveMinFn)(Func *self);
    WarpActiveProductFn *(*Func_as_WarpActiveProductFn)(Func *self);
    WarpActiveSumFn *(*Func_as_WarpActiveSumFn)(Func *self);
    WarpActiveAllFn *(*Func_as_WarpActiveAllFn)(Func *self);
    WarpActiveAnyFn *(*Func_as_WarpActiveAnyFn)(Func *self);
    WarpActiveBitMaskFn *(*Func_as_WarpActiveBitMaskFn)(Func *self);
    WarpPrefixCountBitsFn *(*Func_as_WarpPrefixCountBitsFn)(Func *self);
    WarpPrefixSumFn *(*Func_as_WarpPrefixSumFn)(Func *self);
    WarpPrefixProductFn *(*Func_as_WarpPrefixProductFn)(Func *self);
    WarpReadLaneAtFn *(*Func_as_WarpReadLaneAtFn)(Func *self);
    WarpReadFirstLaneFn *(*Func_as_WarpReadFirstLaneFn)(Func *self);
    SynchronizeBlockFn *(*Func_as_SynchronizeBlockFn)(Func *self);
    AtomicExchangeFn *(*Func_as_AtomicExchangeFn)(Func *self);
    AtomicCompareExchangeFn *(*Func_as_AtomicCompareExchangeFn)(Func *self);
    AtomicFetchAddFn *(*Func_as_AtomicFetchAddFn)(Func *self);
    AtomicFetchSubFn *(*Func_as_AtomicFetchSubFn)(Func *self);
    AtomicFetchAndFn *(*Func_as_AtomicFetchAndFn)(Func *self);
    AtomicFetchOrFn *(*Func_as_AtomicFetchOrFn)(Func *self);
    AtomicFetchXorFn *(*Func_as_AtomicFetchXorFn)(Func *self);
    AtomicFetchMinFn *(*Func_as_AtomicFetchMinFn)(Func *self);
    AtomicFetchMaxFn *(*Func_as_AtomicFetchMaxFn)(Func *self);
    BufferWriteFn *(*Func_as_BufferWriteFn)(Func *self);
    BufferReadFn *(*Func_as_BufferReadFn)(Func *self);
    BufferSizeFn *(*Func_as_BufferSizeFn)(Func *self);
    ByteBufferWriteFn *(*Func_as_ByteBufferWriteFn)(Func *self);
    ByteBufferReadFn *(*Func_as_ByteBufferReadFn)(Func *self);
    ByteBufferSizeFn *(*Func_as_ByteBufferSizeFn)(Func *self);
    Texture2dReadFn *(*Func_as_Texture2dReadFn)(Func *self);
    Texture2dWriteFn *(*Func_as_Texture2dWriteFn)(Func *self);
    Texture2dSizeFn *(*Func_as_Texture2dSizeFn)(Func *self);
    Texture3dReadFn *(*Func_as_Texture3dReadFn)(Func *self);
    Texture3dWriteFn *(*Func_as_Texture3dWriteFn)(Func *self);
    Texture3dSizeFn *(*Func_as_Texture3dSizeFn)(Func *self);
    BindlessTexture2dSampleFn *(*Func_as_BindlessTexture2dSampleFn)(Func *self);
    BindlessTexture2dSampleLevelFn *(*Func_as_BindlessTexture2dSampleLevelFn)(Func *self);
    BindlessTexture2dSampleGradFn *(*Func_as_BindlessTexture2dSampleGradFn)(Func *self);
    BindlessTexture2dSampleGradLevelFn *(*Func_as_BindlessTexture2dSampleGradLevelFn)(Func *self);
    BindlessTexture2dReadFn *(*Func_as_BindlessTexture2dReadFn)(Func *self);
    BindlessTexture2dSizeFn *(*Func_as_BindlessTexture2dSizeFn)(Func *self);
    BindlessTexture2dSizeLevelFn *(*Func_as_BindlessTexture2dSizeLevelFn)(Func *self);
    BindlessTexture3dSampleFn *(*Func_as_BindlessTexture3dSampleFn)(Func *self);
    BindlessTexture3dSampleLevelFn *(*Func_as_BindlessTexture3dSampleLevelFn)(Func *self);
    BindlessTexture3dSampleGradFn *(*Func_as_BindlessTexture3dSampleGradFn)(Func *self);
    BindlessTexture3dSampleGradLevelFn *(*Func_as_BindlessTexture3dSampleGradLevelFn)(Func *self);
    BindlessTexture3dReadFn *(*Func_as_BindlessTexture3dReadFn)(Func *self);
    BindlessTexture3dSizeFn *(*Func_as_BindlessTexture3dSizeFn)(Func *self);
    BindlessTexture3dSizeLevelFn *(*Func_as_BindlessTexture3dSizeLevelFn)(Func *self);
    BindlessBufferWriteFn *(*Func_as_BindlessBufferWriteFn)(Func *self);
    BindlessBufferReadFn *(*Func_as_BindlessBufferReadFn)(Func *self);
    BindlessBufferSizeFn *(*Func_as_BindlessBufferSizeFn)(Func *self);
    BindlessByteBufferWriteFn *(*Func_as_BindlessByteBufferWriteFn)(Func *self);
    BindlessByteBufferReadFn *(*Func_as_BindlessByteBufferReadFn)(Func *self);
    BindlessByteBufferSizeFn *(*Func_as_BindlessByteBufferSizeFn)(Func *self);
    VecFn *(*Func_as_VecFn)(Func *self);
    Vec2Fn *(*Func_as_Vec2Fn)(Func *self);
    Vec3Fn *(*Func_as_Vec3Fn)(Func *self);
    Vec4Fn *(*Func_as_Vec4Fn)(Func *self);
    PermuteFn *(*Func_as_PermuteFn)(Func *self);
    GetElementPtrFn *(*Func_as_GetElementPtrFn)(Func *self);
    ExtractElementFn *(*Func_as_ExtractElementFn)(Func *self);
    InsertElementFn *(*Func_as_InsertElementFn)(Func *self);
    ArrayFn *(*Func_as_ArrayFn)(Func *self);
    StructFn *(*Func_as_StructFn)(Func *self);
    MatFullFn *(*Func_as_MatFullFn)(Func *self);
    Mat2Fn *(*Func_as_Mat2Fn)(Func *self);
    Mat3Fn *(*Func_as_Mat3Fn)(Func *self);
    Mat4Fn *(*Func_as_Mat4Fn)(Func *self);
    BindlessAtomicExchangeFn *(*Func_as_BindlessAtomicExchangeFn)(Func *self);
    BindlessAtomicCompareExchangeFn *(*Func_as_BindlessAtomicCompareExchangeFn)(Func *self);
    BindlessAtomicFetchAddFn *(*Func_as_BindlessAtomicFetchAddFn)(Func *self);
    BindlessAtomicFetchSubFn *(*Func_as_BindlessAtomicFetchSubFn)(Func *self);
    BindlessAtomicFetchAndFn *(*Func_as_BindlessAtomicFetchAndFn)(Func *self);
    BindlessAtomicFetchOrFn *(*Func_as_BindlessAtomicFetchOrFn)(Func *self);
    BindlessAtomicFetchXorFn *(*Func_as_BindlessAtomicFetchXorFn)(Func *self);
    BindlessAtomicFetchMinFn *(*Func_as_BindlessAtomicFetchMinFn)(Func *self);
    BindlessAtomicFetchMaxFn *(*Func_as_BindlessAtomicFetchMaxFn)(Func *self);
    CallableFn *(*Func_as_CallableFn)(Func *self);
    CpuExtFn *(*Func_as_CpuExtFn)(Func *self);
    ShaderExecutionReorderFn *(*Func_as_ShaderExecutionReorderFn)(Func *self);
    FuncTag (*Func_tag)(Func *self);
    ZeroFn *(*ZeroFn_new)(Pool *pool);
    OneFn *(*OneFn_new)(Pool *pool);
    Slice<const char> (*AssumeFn_msg)(AssumeFn *self);
    void (*AssumeFn_set_msg)(AssumeFn *self, Slice<const char> value);
    AssumeFn *(*AssumeFn_new)(Pool *pool, Slice<const char> msg);
    Slice<const char> (*UnreachableFn_msg)(UnreachableFn *self);
    void (*UnreachableFn_set_msg)(UnreachableFn *self, Slice<const char> value);
    UnreachableFn *(*UnreachableFn_new)(Pool *pool, Slice<const char> msg);
    ThreadIdFn *(*ThreadIdFn_new)(Pool *pool);
    BlockIdFn *(*BlockIdFn_new)(Pool *pool);
    WarpSizeFn *(*WarpSizeFn_new)(Pool *pool);
    WarpLaneIdFn *(*WarpLaneIdFn_new)(Pool *pool);
    DispatchIdFn *(*DispatchIdFn_new)(Pool *pool);
    DispatchSizeFn *(*DispatchSizeFn_new)(Pool *pool);
    PropagateGradientFn *(*PropagateGradientFn_new)(Pool *pool);
    OutputGradientFn *(*OutputGradientFn_new)(Pool *pool);
    RequiresGradientFn *(*RequiresGradientFn_new)(Pool *pool);
    BackwardFn *(*BackwardFn_new)(Pool *pool);
    GradientFn *(*GradientFn_new)(Pool *pool);
    AccGradFn *(*AccGradFn_new)(Pool *pool);
    DetachFn *(*DetachFn_new)(Pool *pool);
    RayTracingInstanceTransformFn *(*RayTracingInstanceTransformFn_new)(Pool *pool);
    RayTracingInstanceVisibilityMaskFn *(*RayTracingInstanceVisibilityMaskFn_new)(Pool *pool);
    RayTracingInstanceUserIdFn *(*RayTracingInstanceUserIdFn_new)(Pool *pool);
    RayTracingSetInstanceTransformFn *(*RayTracingSetInstanceTransformFn_new)(Pool *pool);
    RayTracingSetInstanceOpacityFn *(*RayTracingSetInstanceOpacityFn_new)(Pool *pool);
    RayTracingSetInstanceVisibilityFn *(*RayTracingSetInstanceVisibilityFn_new)(Pool *pool);
    RayTracingSetInstanceUserIdFn *(*RayTracingSetInstanceUserIdFn_new)(Pool *pool);
    RayTracingTraceClosestFn *(*RayTracingTraceClosestFn_new)(Pool *pool);
    RayTracingTraceAnyFn *(*RayTracingTraceAnyFn_new)(Pool *pool);
    RayTracingQueryAllFn *(*RayTracingQueryAllFn_new)(Pool *pool);
    RayTracingQueryAnyFn *(*RayTracingQueryAnyFn_new)(Pool *pool);
    RayQueryWorldSpaceRayFn *(*RayQueryWorldSpaceRayFn_new)(Pool *pool);
    RayQueryProceduralCandidateHitFn *(*RayQueryProceduralCandidateHitFn_new)(Pool *pool);
    RayQueryTriangleCandidateHitFn *(*RayQueryTriangleCandidateHitFn_new)(Pool *pool);
    RayQueryCommittedHitFn *(*RayQueryCommittedHitFn_new)(Pool *pool);
    RayQueryCommitTriangleFn *(*RayQueryCommitTriangleFn_new)(Pool *pool);
    RayQueryCommitdProceduralFn *(*RayQueryCommitdProceduralFn_new)(Pool *pool);
    RayQueryTerminateFn *(*RayQueryTerminateFn_new)(Pool *pool);
    LoadFn *(*LoadFn_new)(Pool *pool);
    CastFn *(*CastFn_new)(Pool *pool);
    BitCastFn *(*BitCastFn_new)(Pool *pool);
    AddFn *(*AddFn_new)(Pool *pool);
    SubFn *(*SubFn_new)(Pool *pool);
    MulFn *(*MulFn_new)(Pool *pool);
    DivFn *(*DivFn_new)(Pool *pool);
    RemFn *(*RemFn_new)(Pool *pool);
    BitAndFn *(*BitAndFn_new)(Pool *pool);
    BitOrFn *(*BitOrFn_new)(Pool *pool);
    BitXorFn *(*BitXorFn_new)(Pool *pool);
    ShlFn *(*ShlFn_new)(Pool *pool);
    ShrFn *(*ShrFn_new)(Pool *pool);
    RotRightFn *(*RotRightFn_new)(Pool *pool);
    RotLeftFn *(*RotLeftFn_new)(Pool *pool);
    EqFn *(*EqFn_new)(Pool *pool);
    NeFn *(*NeFn_new)(Pool *pool);
    LtFn *(*LtFn_new)(Pool *pool);
    LeFn *(*LeFn_new)(Pool *pool);
    GtFn *(*GtFn_new)(Pool *pool);
    GeFn *(*GeFn_new)(Pool *pool);
    MatCompMulFn *(*MatCompMulFn_new)(Pool *pool);
    NegFn *(*NegFn_new)(Pool *pool);
    NotFn *(*NotFn_new)(Pool *pool);
    BitNotFn *(*BitNotFn_new)(Pool *pool);
    AllFn *(*AllFn_new)(Pool *pool);
    AnyFn *(*AnyFn_new)(Pool *pool);
    SelectFn *(*SelectFn_new)(Pool *pool);
    ClampFn *(*ClampFn_new)(Pool *pool);
    LerpFn *(*LerpFn_new)(Pool *pool);
    StepFn *(*StepFn_new)(Pool *pool);
    SaturateFn *(*SaturateFn_new)(Pool *pool);
    SmoothStepFn *(*SmoothStepFn_new)(Pool *pool);
    AbsFn *(*AbsFn_new)(Pool *pool);
    MinFn *(*MinFn_new)(Pool *pool);
    MaxFn *(*MaxFn_new)(Pool *pool);
    ReduceSumFn *(*ReduceSumFn_new)(Pool *pool);
    ReduceProdFn *(*ReduceProdFn_new)(Pool *pool);
    ReduceMinFn *(*ReduceMinFn_new)(Pool *pool);
    ReduceMaxFn *(*ReduceMaxFn_new)(Pool *pool);
    ClzFn *(*ClzFn_new)(Pool *pool);
    CtzFn *(*CtzFn_new)(Pool *pool);
    PopCountFn *(*PopCountFn_new)(Pool *pool);
    ReverseFn *(*ReverseFn_new)(Pool *pool);
    IsInfFn *(*IsInfFn_new)(Pool *pool);
    IsNanFn *(*IsNanFn_new)(Pool *pool);
    AcosFn *(*AcosFn_new)(Pool *pool);
    AcoshFn *(*AcoshFn_new)(Pool *pool);
    AsinFn *(*AsinFn_new)(Pool *pool);
    AsinhFn *(*AsinhFn_new)(Pool *pool);
    AtanFn *(*AtanFn_new)(Pool *pool);
    Atan2Fn *(*Atan2Fn_new)(Pool *pool);
    AtanhFn *(*AtanhFn_new)(Pool *pool);
    CosFn *(*CosFn_new)(Pool *pool);
    CoshFn *(*CoshFn_new)(Pool *pool);
    SinFn *(*SinFn_new)(Pool *pool);
    SinhFn *(*SinhFn_new)(Pool *pool);
    TanFn *(*TanFn_new)(Pool *pool);
    TanhFn *(*TanhFn_new)(Pool *pool);
    ExpFn *(*ExpFn_new)(Pool *pool);
    Exp2Fn *(*Exp2Fn_new)(Pool *pool);
    Exp10Fn *(*Exp10Fn_new)(Pool *pool);
    LogFn *(*LogFn_new)(Pool *pool);
    Log2Fn *(*Log2Fn_new)(Pool *pool);
    Log10Fn *(*Log10Fn_new)(Pool *pool);
    PowiFn *(*PowiFn_new)(Pool *pool);
    PowfFn *(*PowfFn_new)(Pool *pool);
    SqrtFn *(*SqrtFn_new)(Pool *pool);
    RsqrtFn *(*RsqrtFn_new)(Pool *pool);
    CeilFn *(*CeilFn_new)(Pool *pool);
    FloorFn *(*FloorFn_new)(Pool *pool);
    FractFn *(*FractFn_new)(Pool *pool);
    TruncFn *(*TruncFn_new)(Pool *pool);
    RoundFn *(*RoundFn_new)(Pool *pool);
    FmaFn *(*FmaFn_new)(Pool *pool);
    CopysignFn *(*CopysignFn_new)(Pool *pool);
    CrossFn *(*CrossFn_new)(Pool *pool);
    DotFn *(*DotFn_new)(Pool *pool);
    OuterProductFn *(*OuterProductFn_new)(Pool *pool);
    LengthFn *(*LengthFn_new)(Pool *pool);
    LengthSquaredFn *(*LengthSquaredFn_new)(Pool *pool);
    NormalizeFn *(*NormalizeFn_new)(Pool *pool);
    FaceforwardFn *(*FaceforwardFn_new)(Pool *pool);
    DistanceFn *(*DistanceFn_new)(Pool *pool);
    ReflectFn *(*ReflectFn_new)(Pool *pool);
    DeterminantFn *(*DeterminantFn_new)(Pool *pool);
    TransposeFn *(*TransposeFn_new)(Pool *pool);
    InverseFn *(*InverseFn_new)(Pool *pool);
    WarpIsFirstActiveLaneFn *(*WarpIsFirstActiveLaneFn_new)(Pool *pool);
    WarpFirstActiveLaneFn *(*WarpFirstActiveLaneFn_new)(Pool *pool);
    WarpActiveAllEqualFn *(*WarpActiveAllEqualFn_new)(Pool *pool);
    WarpActiveBitAndFn *(*WarpActiveBitAndFn_new)(Pool *pool);
    WarpActiveBitOrFn *(*WarpActiveBitOrFn_new)(Pool *pool);
    WarpActiveBitXorFn *(*WarpActiveBitXorFn_new)(Pool *pool);
    WarpActiveCountBitsFn *(*WarpActiveCountBitsFn_new)(Pool *pool);
    WarpActiveMaxFn *(*WarpActiveMaxFn_new)(Pool *pool);
    WarpActiveMinFn *(*WarpActiveMinFn_new)(Pool *pool);
    WarpActiveProductFn *(*WarpActiveProductFn_new)(Pool *pool);
    WarpActiveSumFn *(*WarpActiveSumFn_new)(Pool *pool);
    WarpActiveAllFn *(*WarpActiveAllFn_new)(Pool *pool);
    WarpActiveAnyFn *(*WarpActiveAnyFn_new)(Pool *pool);
    WarpActiveBitMaskFn *(*WarpActiveBitMaskFn_new)(Pool *pool);
    WarpPrefixCountBitsFn *(*WarpPrefixCountBitsFn_new)(Pool *pool);
    WarpPrefixSumFn *(*WarpPrefixSumFn_new)(Pool *pool);
    WarpPrefixProductFn *(*WarpPrefixProductFn_new)(Pool *pool);
    WarpReadLaneAtFn *(*WarpReadLaneAtFn_new)(Pool *pool);
    WarpReadFirstLaneFn *(*WarpReadFirstLaneFn_new)(Pool *pool);
    SynchronizeBlockFn *(*SynchronizeBlockFn_new)(Pool *pool);
    AtomicExchangeFn *(*AtomicExchangeFn_new)(Pool *pool);
    AtomicCompareExchangeFn *(*AtomicCompareExchangeFn_new)(Pool *pool);
    AtomicFetchAddFn *(*AtomicFetchAddFn_new)(Pool *pool);
    AtomicFetchSubFn *(*AtomicFetchSubFn_new)(Pool *pool);
    AtomicFetchAndFn *(*AtomicFetchAndFn_new)(Pool *pool);
    AtomicFetchOrFn *(*AtomicFetchOrFn_new)(Pool *pool);
    AtomicFetchXorFn *(*AtomicFetchXorFn_new)(Pool *pool);
    AtomicFetchMinFn *(*AtomicFetchMinFn_new)(Pool *pool);
    AtomicFetchMaxFn *(*AtomicFetchMaxFn_new)(Pool *pool);
    BufferWriteFn *(*BufferWriteFn_new)(Pool *pool);
    BufferReadFn *(*BufferReadFn_new)(Pool *pool);
    BufferSizeFn *(*BufferSizeFn_new)(Pool *pool);
    ByteBufferWriteFn *(*ByteBufferWriteFn_new)(Pool *pool);
    ByteBufferReadFn *(*ByteBufferReadFn_new)(Pool *pool);
    ByteBufferSizeFn *(*ByteBufferSizeFn_new)(Pool *pool);
    Texture2dReadFn *(*Texture2dReadFn_new)(Pool *pool);
    Texture2dWriteFn *(*Texture2dWriteFn_new)(Pool *pool);
    Texture2dSizeFn *(*Texture2dSizeFn_new)(Pool *pool);
    Texture3dReadFn *(*Texture3dReadFn_new)(Pool *pool);
    Texture3dWriteFn *(*Texture3dWriteFn_new)(Pool *pool);
    Texture3dSizeFn *(*Texture3dSizeFn_new)(Pool *pool);
    BindlessTexture2dSampleFn *(*BindlessTexture2dSampleFn_new)(Pool *pool);
    BindlessTexture2dSampleLevelFn *(*BindlessTexture2dSampleLevelFn_new)(Pool *pool);
    BindlessTexture2dSampleGradFn *(*BindlessTexture2dSampleGradFn_new)(Pool *pool);
    BindlessTexture2dSampleGradLevelFn *(*BindlessTexture2dSampleGradLevelFn_new)(Pool *pool);
    BindlessTexture2dReadFn *(*BindlessTexture2dReadFn_new)(Pool *pool);
    BindlessTexture2dSizeFn *(*BindlessTexture2dSizeFn_new)(Pool *pool);
    BindlessTexture2dSizeLevelFn *(*BindlessTexture2dSizeLevelFn_new)(Pool *pool);
    BindlessTexture3dSampleFn *(*BindlessTexture3dSampleFn_new)(Pool *pool);
    BindlessTexture3dSampleLevelFn *(*BindlessTexture3dSampleLevelFn_new)(Pool *pool);
    BindlessTexture3dSampleGradFn *(*BindlessTexture3dSampleGradFn_new)(Pool *pool);
    BindlessTexture3dSampleGradLevelFn *(*BindlessTexture3dSampleGradLevelFn_new)(Pool *pool);
    BindlessTexture3dReadFn *(*BindlessTexture3dReadFn_new)(Pool *pool);
    BindlessTexture3dSizeFn *(*BindlessTexture3dSizeFn_new)(Pool *pool);
    BindlessTexture3dSizeLevelFn *(*BindlessTexture3dSizeLevelFn_new)(Pool *pool);
    BindlessBufferWriteFn *(*BindlessBufferWriteFn_new)(Pool *pool);
    BindlessBufferReadFn *(*BindlessBufferReadFn_new)(Pool *pool);
    BindlessBufferSizeFn *(*BindlessBufferSizeFn_new)(Pool *pool);
    BindlessByteBufferWriteFn *(*BindlessByteBufferWriteFn_new)(Pool *pool);
    BindlessByteBufferReadFn *(*BindlessByteBufferReadFn_new)(Pool *pool);
    BindlessByteBufferSizeFn *(*BindlessByteBufferSizeFn_new)(Pool *pool);
    VecFn *(*VecFn_new)(Pool *pool);
    Vec2Fn *(*Vec2Fn_new)(Pool *pool);
    Vec3Fn *(*Vec3Fn_new)(Pool *pool);
    Vec4Fn *(*Vec4Fn_new)(Pool *pool);
    PermuteFn *(*PermuteFn_new)(Pool *pool);
    GetElementPtrFn *(*GetElementPtrFn_new)(Pool *pool);
    ExtractElementFn *(*ExtractElementFn_new)(Pool *pool);
    InsertElementFn *(*InsertElementFn_new)(Pool *pool);
    ArrayFn *(*ArrayFn_new)(Pool *pool);
    StructFn *(*StructFn_new)(Pool *pool);
    MatFullFn *(*MatFullFn_new)(Pool *pool);
    Mat2Fn *(*Mat2Fn_new)(Pool *pool);
    Mat3Fn *(*Mat3Fn_new)(Pool *pool);
    Mat4Fn *(*Mat4Fn_new)(Pool *pool);
    const Type *(*BindlessAtomicExchangeFn_ty)(BindlessAtomicExchangeFn *self);
    void (*BindlessAtomicExchangeFn_set_ty)(BindlessAtomicExchangeFn *self, const Type *value);
    BindlessAtomicExchangeFn *(*BindlessAtomicExchangeFn_new)(Pool *pool, const Type *ty);
    const Type *(*BindlessAtomicCompareExchangeFn_ty)(BindlessAtomicCompareExchangeFn *self);
    void (*BindlessAtomicCompareExchangeFn_set_ty)(BindlessAtomicCompareExchangeFn *self, const Type *value);
    BindlessAtomicCompareExchangeFn *(*BindlessAtomicCompareExchangeFn_new)(Pool *pool, const Type *ty);
    const Type *(*BindlessAtomicFetchAddFn_ty)(BindlessAtomicFetchAddFn *self);
    void (*BindlessAtomicFetchAddFn_set_ty)(BindlessAtomicFetchAddFn *self, const Type *value);
    BindlessAtomicFetchAddFn *(*BindlessAtomicFetchAddFn_new)(Pool *pool, const Type *ty);
    const Type *(*BindlessAtomicFetchSubFn_ty)(BindlessAtomicFetchSubFn *self);
    void (*BindlessAtomicFetchSubFn_set_ty)(BindlessAtomicFetchSubFn *self, const Type *value);
    BindlessAtomicFetchSubFn *(*BindlessAtomicFetchSubFn_new)(Pool *pool, const Type *ty);
    const Type *(*BindlessAtomicFetchAndFn_ty)(BindlessAtomicFetchAndFn *self);
    void (*BindlessAtomicFetchAndFn_set_ty)(BindlessAtomicFetchAndFn *self, const Type *value);
    BindlessAtomicFetchAndFn *(*BindlessAtomicFetchAndFn_new)(Pool *pool, const Type *ty);
    const Type *(*BindlessAtomicFetchOrFn_ty)(BindlessAtomicFetchOrFn *self);
    void (*BindlessAtomicFetchOrFn_set_ty)(BindlessAtomicFetchOrFn *self, const Type *value);
    BindlessAtomicFetchOrFn *(*BindlessAtomicFetchOrFn_new)(Pool *pool, const Type *ty);
    const Type *(*BindlessAtomicFetchXorFn_ty)(BindlessAtomicFetchXorFn *self);
    void (*BindlessAtomicFetchXorFn_set_ty)(BindlessAtomicFetchXorFn *self, const Type *value);
    BindlessAtomicFetchXorFn *(*BindlessAtomicFetchXorFn_new)(Pool *pool, const Type *ty);
    const Type *(*BindlessAtomicFetchMinFn_ty)(BindlessAtomicFetchMinFn *self);
    void (*BindlessAtomicFetchMinFn_set_ty)(BindlessAtomicFetchMinFn *self, const Type *value);
    BindlessAtomicFetchMinFn *(*BindlessAtomicFetchMinFn_new)(Pool *pool, const Type *ty);
    const Type *(*BindlessAtomicFetchMaxFn_ty)(BindlessAtomicFetchMaxFn *self);
    void (*BindlessAtomicFetchMaxFn_set_ty)(BindlessAtomicFetchMaxFn *self, const Type *value);
    BindlessAtomicFetchMaxFn *(*BindlessAtomicFetchMaxFn_new)(Pool *pool, const Type *ty);
    CallableModule *(*CallableFn_module)(CallableFn *self);
    void (*CallableFn_set_module)(CallableFn *self, CallableModule *value);
    CallableFn *(*CallableFn_new)(Pool *pool, CallableModule *module);
    CpuExternFn (*CpuExtFn_f)(CpuExtFn *self);
    void (*CpuExtFn_set_f)(CpuExtFn *self, CpuExternFn value);
    CpuExtFn *(*CpuExtFn_new)(Pool *pool, CpuExternFn f);
    ShaderExecutionReorderFn *(*ShaderExecutionReorderFn_new)(Pool *pool);
    BufferInst *(*Instruction_as_BufferInst)(Instruction *self);
    Texture2dInst *(*Instruction_as_Texture2dInst)(Instruction *self);
    Texture3dInst *(*Instruction_as_Texture3dInst)(Instruction *self);
    BindlessArrayInst *(*Instruction_as_BindlessArrayInst)(Instruction *self);
    AccelInst *(*Instruction_as_AccelInst)(Instruction *self);
    SharedInst *(*Instruction_as_SharedInst)(Instruction *self);
    UniformInst *(*Instruction_as_UniformInst)(Instruction *self);
    ArgumentInst *(*Instruction_as_ArgumentInst)(Instruction *self);
    ConstantInst *(*Instruction_as_ConstantInst)(Instruction *self);
    CallInst *(*Instruction_as_CallInst)(Instruction *self);
    PhiInst *(*Instruction_as_PhiInst)(Instruction *self);
    BasicBlockSentinelInst *(*Instruction_as_BasicBlockSentinelInst)(Instruction *self);
    IfInst *(*Instruction_as_IfInst)(Instruction *self);
    GenericLoopInst *(*Instruction_as_GenericLoopInst)(Instruction *self);
    SwitchInst *(*Instruction_as_SwitchInst)(Instruction *self);
    LocalInst *(*Instruction_as_LocalInst)(Instruction *self);
    BreakInst *(*Instruction_as_BreakInst)(Instruction *self);
    ContinueInst *(*Instruction_as_ContinueInst)(Instruction *self);
    ReturnInst *(*Instruction_as_ReturnInst)(Instruction *self);
    PrintInst *(*Instruction_as_PrintInst)(Instruction *self);
    UpdateInst *(*Instruction_as_UpdateInst)(Instruction *self);
    RayQueryInst *(*Instruction_as_RayQueryInst)(Instruction *self);
    RevAutodiffInst *(*Instruction_as_RevAutodiffInst)(Instruction *self);
    FwdAutodiffInst *(*Instruction_as_FwdAutodiffInst)(Instruction *self);
    InstructionTag (*Instruction_tag)(Instruction *self);
    BufferInst *(*BufferInst_new)(Pool *pool);
    Texture2dInst *(*Texture2dInst_new)(Pool *pool);
    Texture3dInst *(*Texture3dInst_new)(Pool *pool);
    BindlessArrayInst *(*BindlessArrayInst_new)(Pool *pool);
    AccelInst *(*AccelInst_new)(Pool *pool);
    SharedInst *(*SharedInst_new)(Pool *pool);
    UniformInst *(*UniformInst_new)(Pool *pool);
    bool (*ArgumentInst_by_value)(ArgumentInst *self);
    void (*ArgumentInst_set_by_value)(ArgumentInst *self, bool value);
    ArgumentInst *(*ArgumentInst_new)(Pool *pool, bool by_value);
    const Type *(*ConstantInst_ty)(ConstantInst *self);
    Slice<uint8_t> (*ConstantInst_value)(ConstantInst *self);
    void (*ConstantInst_set_ty)(ConstantInst *self, const Type *value);
    void (*ConstantInst_set_value)(ConstantInst *self, Slice<uint8_t> value);
    ConstantInst *(*ConstantInst_new)(Pool *pool, const Type *ty, Slice<uint8_t> value);
    const Func *(*CallInst_func)(CallInst *self);
    Slice<Node *> (*CallInst_args)(CallInst *self);
    void (*CallInst_set_func)(CallInst *self, const Func *value);
    void (*CallInst_set_args)(CallInst *self, Slice<Node *> value);
    CallInst *(*CallInst_new)(Pool *pool, const Func *func, Slice<Node *> args);
    Slice<PhiIncoming> (*PhiInst_incomings)(PhiInst *self);
    void (*PhiInst_set_incomings)(PhiInst *self, Slice<PhiIncoming> value);
    PhiInst *(*PhiInst_new)(Pool *pool, Slice<PhiIncoming> incomings);
    BasicBlockSentinelInst *(*BasicBlockSentinelInst_new)(Pool *pool);
    Node *(*IfInst_cond)(IfInst *self);
    BasicBlock *(*IfInst_true_branch)(IfInst *self);
    BasicBlock *(*IfInst_false_branch)(IfInst *self);
    void (*IfInst_set_cond)(IfInst *self, Node *value);
    void (*IfInst_set_true_branch)(IfInst *self, BasicBlock *value);
    void (*IfInst_set_false_branch)(IfInst *self, BasicBlock *value);
    IfInst *(*IfInst_new)(Pool *pool, Node *cond, BasicBlock *true_branch, BasicBlock *false_branch);
    BasicBlock *(*GenericLoopInst_prepare)(GenericLoopInst *self);
    Node *(*GenericLoopInst_cond)(GenericLoopInst *self);
    BasicBlock *(*GenericLoopInst_body)(GenericLoopInst *self);
    BasicBlock *(*GenericLoopInst_update)(GenericLoopInst *self);
    void (*GenericLoopInst_set_prepare)(GenericLoopInst *self, BasicBlock *value);
    void (*GenericLoopInst_set_cond)(GenericLoopInst *self, Node *value);
    void (*GenericLoopInst_set_body)(GenericLoopInst *self, BasicBlock *value);
    void (*GenericLoopInst_set_update)(GenericLoopInst *self, BasicBlock *value);
    GenericLoopInst *(*GenericLoopInst_new)(Pool *pool, BasicBlock *prepare, Node *cond, BasicBlock *body, BasicBlock *update);
    Node *(*SwitchInst_value)(SwitchInst *self);
    Slice<SwitchCase> (*SwitchInst_cases)(SwitchInst *self);
    BasicBlock *(*SwitchInst_default_)(SwitchInst *self);
    void (*SwitchInst_set_value)(SwitchInst *self, Node *value);
    void (*SwitchInst_set_cases)(SwitchInst *self, Slice<SwitchCase> value);
    void (*SwitchInst_set_default_)(SwitchInst *self, BasicBlock *value);
    SwitchInst *(*SwitchInst_new)(Pool *pool, Node *value, Slice<SwitchCase> cases, BasicBlock *default_);
    Node *(*LocalInst_init)(LocalInst *self);
    void (*LocalInst_set_init)(LocalInst *self, Node *value);
    LocalInst *(*LocalInst_new)(Pool *pool, Node *init);
    BreakInst *(*BreakInst_new)(Pool *pool);
    ContinueInst *(*ContinueInst_new)(Pool *pool);
    Node *(*ReturnInst_value)(ReturnInst *self);
    void (*ReturnInst_set_value)(ReturnInst *self, Node *value);
    ReturnInst *(*ReturnInst_new)(Pool *pool, Node *value);
    Slice<const char> (*PrintInst_fmt)(PrintInst *self);
    Slice<Node *> (*PrintInst_args)(PrintInst *self);
    void (*PrintInst_set_fmt)(PrintInst *self, Slice<const char> value);
    void (*PrintInst_set_args)(PrintInst *self, Slice<Node *> value);
    PrintInst *(*PrintInst_new)(Pool *pool, Slice<const char> fmt, Slice<Node *> args);
    Node *(*UpdateInst_var)(UpdateInst *self);
    Node *(*UpdateInst_value)(UpdateInst *self);
    void (*UpdateInst_set_var)(UpdateInst *self, Node *value);
    void (*UpdateInst_set_value)(UpdateInst *self, Node *value);
    UpdateInst *(*UpdateInst_new)(Pool *pool, Node *var, Node *value);
    Node *(*RayQueryInst_query)(RayQueryInst *self);
    BasicBlock *(*RayQueryInst_on_triangle_hit)(RayQueryInst *self);
    BasicBlock *(*RayQueryInst_on_procedural_hit)(RayQueryInst *self);
    void (*RayQueryInst_set_query)(RayQueryInst *self, Node *value);
    void (*RayQueryInst_set_on_triangle_hit)(RayQueryInst *self, BasicBlock *value);
    void (*RayQueryInst_set_on_procedural_hit)(RayQueryInst *self, BasicBlock *value);
    RayQueryInst *(*RayQueryInst_new)(Pool *pool, Node *query, BasicBlock *on_triangle_hit, BasicBlock *on_procedural_hit);
    BasicBlock *(*RevAutodiffInst_body)(RevAutodiffInst *self);
    void (*RevAutodiffInst_set_body)(RevAutodiffInst *self, BasicBlock *value);
    RevAutodiffInst *(*RevAutodiffInst_new)(Pool *pool, BasicBlock *body);
    BasicBlock *(*FwdAutodiffInst_body)(FwdAutodiffInst *self);
    void (*FwdAutodiffInst_set_body)(FwdAutodiffInst *self, BasicBlock *value);
    FwdAutodiffInst *(*FwdAutodiffInst_new)(Pool *pool, BasicBlock *body);
    BufferBinding *(*Binding_as_BufferBinding)(Binding *self);
    TextureBinding *(*Binding_as_TextureBinding)(Binding *self);
    BindlessArrayBinding *(*Binding_as_BindlessArrayBinding)(Binding *self);
    AccelBinding *(*Binding_as_AccelBinding)(Binding *self);
    BindingTag (*Binding_tag)(Binding *self);
    uint64_t (*BufferBinding_handle)(BufferBinding *self);
    uint64_t (*BufferBinding_offset)(BufferBinding *self);
    uint64_t (*BufferBinding_size)(BufferBinding *self);
    void (*BufferBinding_set_handle)(BufferBinding *self, uint64_t value);
    void (*BufferBinding_set_offset)(BufferBinding *self, uint64_t value);
    void (*BufferBinding_set_size)(BufferBinding *self, uint64_t value);
    BufferBinding *(*BufferBinding_new)(Pool *pool, uint64_t handle, uint64_t offset, uint64_t size);
    uint64_t (*TextureBinding_handle)(TextureBinding *self);
    uint64_t (*TextureBinding_level)(TextureBinding *self);
    void (*TextureBinding_set_handle)(TextureBinding *self, uint64_t value);
    void (*TextureBinding_set_level)(TextureBinding *self, uint64_t value);
    TextureBinding *(*TextureBinding_new)(Pool *pool, uint64_t handle, uint64_t level);
    uint64_t (*BindlessArrayBinding_handle)(BindlessArrayBinding *self);
    void (*BindlessArrayBinding_set_handle)(BindlessArrayBinding *self, uint64_t value);
    BindlessArrayBinding *(*BindlessArrayBinding_new)(Pool *pool, uint64_t handle);
    uint64_t (*AccelBinding_handle)(AccelBinding *self);
    void (*AccelBinding_set_handle)(AccelBinding *self, uint64_t value);
    AccelBinding *(*AccelBinding_new)(Pool *pool, uint64_t handle);
};
extern "C" LC_IR_API IrV2BindingTable lc_ir_v2_binding_table();
}// namespace luisa::compute::ir_v2
