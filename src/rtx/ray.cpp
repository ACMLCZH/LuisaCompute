//
// Created by Mike Smith on 2021/6/24.
//

#include <rtx/ray.h>

namespace luisa::compute {

Expr<float3> origin(Expr<Ray> ray) noexcept {
    static Callable _origin = [](Var<Ray> ray) noexcept {
        return make_float3(ray.origin[0], ray.origin[1], ray.origin[2]);
    };
    return _origin(ray);
}

Expr<float3> direction(Expr<Ray> ray) noexcept {
    static Callable _direction = [](Var<Ray> ray) noexcept {
        return make_float3(ray.direction[0], ray.direction[1], ray.direction[2]);
    };
    return _direction(ray);
}

void set_origin(Ref<Ray> ray, Expr<float3> origin) noexcept {
    Var o = origin;
    ray.origin[0] = o.x;
    ray.origin[1] = o.y;
    ray.origin[2] = o.z;
}

void set_direction(Ref<Ray> ray, Expr<float3> direction) noexcept {
    static Callable _set_direction = [](Ref<Ray> ray, Float3 d) noexcept {
        ray.direction[0] = d.x;
        ray.direction[1] = d.y;
        ray.direction[2] = d.z;
    };
    _set_direction(ray, direction);
}

Expr<Ray> make_ray(Expr<float3> origin, Expr<float3> direction, Expr<float> t_min, Expr<float> t_max) noexcept {
    static Callable _make_ray = [](Float3 origin, Float3 direction, Float t_min, Float t_max) noexcept {
        Var<Ray> ray;
        ray.origin[0] = origin.x;
        ray.origin[1] = origin.y;
        ray.origin[2] = origin.z;
        ray.t_min = t_min;
        ray.direction[0] = direction.x;
        ray.direction[1] = direction.y;
        ray.direction[2] = direction.z;
        ray.t_max = t_max;
        return ray;
    };
    return _make_ray(origin, direction, t_min, t_max);
}

Expr<Ray> make_ray(Expr<float3> origin, Expr<float3> direction) noexcept {
    return make_ray(origin, direction, 0.0f, std::numeric_limits<float>::max());
}

Expr<Ray> make_ray_robust(
    Expr<float3> p, Expr<float3> ng,
    Expr<float3> direction, Expr<float> t_max) noexcept {

    static Callable _make_ray_robust = [](Float3 p, Float3 d, Float3 ng, Float t_max) noexcept {
        constexpr auto origin = 1.0f / 32.0f;
        constexpr auto float_scale = 1.0f / 65536.0f;
        constexpr auto int_scale = 256.0f;
        Var n = sign(dot(ng, d)) * ng;
        Var of_i = make_int3(int_scale * n);
        Var p_i = as<float3>(as<int3>(p) + ite(p < 0.0f, -of_i, of_i));
        Var ro = ite(abs(p) < origin, p + float_scale * n, p_i);
        return make_ray(ro, d, 0.0f, t_max);
    };
    return _make_ray_robust(p, direction, ng, t_max);
}

Expr<Ray> make_ray_robust(Expr<float3> p, Expr<float3> ng, Expr<float3> direction) noexcept {
    return make_ray_robust(p, ng, direction, std::numeric_limits<float>::max());
}

}// namespace luisa::compute
