#pragma once
#include "attributes.hpp"

namespace luisa::shader {

using int32 = int;
using int64 = long long;
using uint32 = unsigned int;
using uint64 = unsigned long long;

template<typename T>
trait remove_cvref { using type = T; };
template<typename T>
trait remove_cvref<T &> { using type = T; };
template<typename T>
trait remove_cvref<T const> { using type = T; };
template<typename T>
trait remove_cvref<T volatile> { using type = T; };
template<typename T>
trait remove_cvref<T &&> { using type = T; };
struct [[builtin("half")]] half {
    [[ignore]] half() = default;
    [[ignore]] half(float);
    [[ignore]] half(uint32);
    [[ignore]] half(int32);
private:
    short v;
};
template<typename T, uint64 N>
struct vec;
template<uint64 N>
struct matrix;

template<typename T>
trait is_floatN { static constexpr bool value = false; };
template<> trait is_floatN<float> { static constexpr bool value = true; };
template<> trait is_floatN<half> { static constexpr bool value = true; };
template<> trait is_floatN<double> { static constexpr bool value = true; };
template<uint64 N> trait is_floatN<vec<float, N>> { static constexpr bool value = true; };
template<uint64 N> trait is_floatN<vec<double, N>> { static constexpr bool value = true; };

template<typename T>
concept floatN = is_floatN<typename remove_cvref<T>::type>::value;

template<typename T>
trait vec_dim { static constexpr uint64 value = 1; };

template<typename T, uint64 N>
trait vec_dim<vec<T, N>> { static constexpr uint64 value = N; };
template<typename T>
[[ignore]] constexpr uint64 vec_dim_v = vec_dim<typename remove_cvref<T>::type>::value;

template<uint64 dim, typename T, typename... Ts>
[[ignore]] consteval uint64 sum_dim() {
    constexpr auto new_dim = dim + vec_dim_v<T>;
    if constexpr (sizeof...(Ts) == 0) {
        return new_dim;
    } else {
        return sum_dim<new_dim, Ts...>();
    }
}

template<typename T, uint64 N>
struct [[builtin("vec")]] vec {
    [[ignore]] vec() noexcept = default;
    template<typename... Args>
        requires(sum_dim<0ull, Args...>() == N)
    [[ignore]] explicit vec(Args &&...args) ;
private:
    T v[N];
};

template<typename T>
struct alignas(8) [[builtin("vec")]] vec<T, 2> {
    [[ignore]] vec() noexcept = default;
    template<typename... Args>
        requires(sum_dim<0ull, Args...>() == 2)
    [[ignore]] explicit vec(Args &&...args) ;
private:
    T v[2];
};

template<typename T>
struct alignas(16) [[builtin("vec")]] vec<T, 3> {
    [[ignore]] vec() noexcept = default;
    template<typename... Args>
        requires(sum_dim<0ull, Args...>() == 3)
    [[ignore]] explicit vec(Args &&...args) ;
private:
    T v[4];
};

template<typename T>
struct alignas(16) [[builtin("vec")]] vec<T, 4> {
    [[ignore]] vec() noexcept = default;
    template<typename... Args>
        requires(sum_dim<0ull, Args...>() == 4)
    [[ignore]] explicit vec(Args &&...args) ;
private:
    T v[4];
};

using float2 = vec<float, 2>;
using float3 = vec<float, 3>;
using float4 = vec<float, 4>;
// using double2 = vec<double, 2>;
// using double3 = vec<double, 3>;
// using double4 = vec<double, 4>;
using int2 = vec<int32, 2>;
using int3 = vec<int32, 3>;
using int4 = vec<int32, 4>;
using uint2 = vec<uint32, 2>;
using uint3 = vec<uint32, 3>;
using uint4 = vec<uint32, 4>;
using half2 = vec<half, 2>;
using half3 = vec<half, 3>;
using half4 = vec<half, 4>;
using bool2 = vec<bool, 2>;
using bool3 = vec<bool, 3>;
using bool4 = vec<bool, 4>;
template<>
struct [[builtin("matrix")]] matrix<2> {
    [[ignore]] matrix(float2 col0, float2 col1);
    [[ignore]] matrix(float m00, float m01, float m10, float m11);
    [[ignore]] matrix(matrix<3> float3x3);
    [[ignore]] matrix(matrix<4> float4x4);
private:
    float2 v[2];
};
template<>
struct [[builtin("matrix")]] matrix<3> {
    [[ignore]] matrix(float3 col0, float3 col1, float3 col2);
    [[ignore]] matrix(float m00, float m01, float m02, float m10, float m11, float m12, float m20, float m21, float m22);
    [[ignore]] matrix(matrix<2> float2x2);
    [[ignore]] matrix(matrix<4> float4x4);
private:
    float3 v[3];
};
template<>
struct alignas(16) [[builtin("matrix")]] matrix<4> {
    [[ignore]] matrix(float4 col0, float4 col1, float4 col2, float4 col3);
    [[ignore]] matrix(float m00, float m01, float m02, float m03, float m10, float m11, float m12, float m13, float m20, float m21, float m22, float m23, float m30, float m31, float m32, float m33);
    [[ignore]] matrix(matrix<2> float2x2);
    [[ignore]] matrix(matrix<3> float3x3);
private:
    float4 v[4];
};
using float2x2 = matrix<2>;
using float3x3 = matrix<3>;
using float4x4 = matrix<4>;
}