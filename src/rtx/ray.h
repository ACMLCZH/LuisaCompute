//
// Created by Mike Smith on 2021/6/24.
//

#pragma once

#include <dsl/struct.h>
#include <dsl/syntax.h>

namespace luisa::compute {

struct alignas(16) Ray {
    std::array<float, 3> compressed_origin;
    float compressed_t_min;
    std::array<float, 3> compressed_direction;
    float compressed_t_max;
};

}

LUISA_STRUCT(
    luisa::compute::Ray,
    compressed_origin,
    compressed_t_min,
    compressed_direction,
    compressed_t_max) {

    [[nodiscard]] auto origin() const noexcept { return luisa::compute::def<luisa::float3>(compressed_origin); }
    [[nodiscard]] auto direction() const noexcept { return luisa::compute::def<luisa::float3>(compressed_direction); }
    [[nodiscard]] auto t_min() const noexcept { return compressed_t_min; }
    [[nodiscard]] auto t_max() const noexcept { return compressed_t_max; }
    void set_origin(luisa::compute::Expr<luisa::float3> origin) noexcept { compressed_origin = origin; }
    void set_direction(luisa::compute::Expr<luisa::float3> direction) noexcept { compressed_direction = direction; }
    void set_t_min(luisa::compute::Expr<float> t_min) noexcept { compressed_t_min = t_min; }
    void set_t_max(luisa::compute::Expr<float> t_max) noexcept { compressed_t_max = t_max; }
};

namespace luisa::compute {

[[nodiscard]] Var<Ray> make_ray(
    Expr<float3> origin,
    Expr<float3> direction,
    Expr<float> t_min,
    Expr<float> t_max) noexcept;

[[nodiscard]] Var<Ray> make_ray(
    Expr<float3> origin,
    Expr<float3> direction) noexcept;

// ray from p with surface normal ng, with self intersections avoidance
[[nodiscard]] Var<Ray> make_ray_robust(
    Expr<float3> p,
    Expr<float3> ng,
    Expr<float3> direction,
    Expr<float> t_max) noexcept;

[[nodiscard]] Var<Ray> make_ray_robust(
    Expr<float3> p,
    Expr<float3> ng,
    Expr<float3> direction) noexcept;

}// namespace luisa::compute
