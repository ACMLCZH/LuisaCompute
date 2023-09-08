#include <luisa/dsl/stmt.h>
#include <luisa/dsl/rtx/accel.h>

namespace luisa::compute {

namespace detail {

Var<TriangleHit> AccelExprProxy::trace_closest(Expr<Ray> ray, Expr<uint> vis_mask, RayCullMode cull_mode) const noexcept {
    return Expr<Accel>{_accel}.trace_closest(ray, vis_mask, cull_mode);
}

Var<bool> AccelExprProxy::trace_any(Expr<Ray> ray, Expr<uint> vis_mask, RayCullMode cull_mode) const noexcept {
    return Expr<Accel>{_accel}.trace_any(ray, vis_mask, cull_mode);
}

RayQueryAll AccelExprProxy::query_all(Expr<Ray> ray, Expr<uint> vis_mask, RayCullMode cull_mode) const noexcept {
    return Expr<Accel>{_accel}.query_all(ray, vis_mask, cull_mode);
}

RayQueryAny AccelExprProxy::query_any(Expr<Ray> ray, Expr<uint> vis_mask, RayCullMode cull_mode) const noexcept {
    return Expr<Accel>{_accel}.query_any(ray, vis_mask, cull_mode);
}

Var<float4x4> AccelExprProxy::instance_transform(Expr<int> instance_id) const noexcept {
    return Expr<Accel>{_accel}.instance_transform(instance_id);
}

Var<float4x4> AccelExprProxy::instance_transform(Expr<uint> instance_id) const noexcept {
    return Expr<Accel>{_accel}.instance_transform(instance_id);
}

void AccelExprProxy::set_instance_transform(Expr<int> instance_id, Expr<float4x4> mat) const noexcept {
    Expr<Accel>{_accel}.set_instance_transform(instance_id, mat);
}

void AccelExprProxy::set_instance_transform(Expr<uint> instance_id, Expr<float4x4> mat) const noexcept {
    Expr<Accel>{_accel}.set_instance_transform(instance_id, mat);
}

void AccelExprProxy::set_instance_visibility(Expr<int> instance_id, Expr<uint> vis_mask) const noexcept {
    Expr<Accel>{_accel}.set_instance_visibility(instance_id, vis_mask);
}

void AccelExprProxy::set_instance_visibility(Expr<uint> instance_id, Expr<uint> vis_mask) const noexcept {
    Expr<Accel>{_accel}.set_instance_visibility(instance_id, vis_mask);
}
void AccelExprProxy::set_instance_opaque(Expr<int> instance_id, Expr<bool> opaque) const noexcept {
    Expr<Accel>{_accel}.set_instance_opaque(instance_id, opaque);
}

void AccelExprProxy::set_instance_opaque(Expr<uint> instance_id, Expr<bool> opaque) const noexcept {
    Expr<Accel>{_accel}.set_instance_opaque(instance_id, opaque);
}

}// namespace detail

Expr<Accel>::Expr(const RefExpr *expr) noexcept
    : _expression{expr} {}

Expr<Accel>::Expr(const Accel &accel) noexcept
    : _expression{detail::FunctionBuilder::current()->accel_binding(
          accel.handle())} {}

Var<TriangleHit> Expr<Accel>::trace_closest(Expr<Ray> ray, Expr<uint> mask, RayCullMode cull_mode) const noexcept {
    CallOp op;
    switch (cull_mode) {
        case RayCullMode::None:
            op = CallOp::RAY_TRACING_TRACE_CLOSEST;
            break;
        case RayCullMode::CullBack:
            op = CallOp::RAY_TRACING_TRACE_CLOSEST_CULL_BACKFACE;
            break;
        case RayCullMode::CullFront:
            op = CallOp::RAY_TRACING_TRACE_CLOSEST_CULL_FRONTFACE;
            break;
    }
    return def<TriangleHit>(
        detail::FunctionBuilder::current()->call(
            Type::of<TriangleHit>(), op,
            {_expression, ray.expression(), mask.expression()}));
}

Var<bool> Expr<Accel>::trace_any(Expr<Ray> ray, Expr<uint> mask, RayCullMode cull_mode) const noexcept {
    CallOp op;
    switch (cull_mode) {
        case RayCullMode::None:
            op = CallOp::RAY_TRACING_TRACE_ANY;
            break;
        case RayCullMode::CullBack:
            op = CallOp::RAY_TRACING_TRACE_ANY_CULL_BACKFACE;
            break;
        case RayCullMode::CullFront:
            op = CallOp::RAY_TRACING_TRACE_ANY_CULL_FRONTFACE;
            break;
    }
    return def<bool>(
        detail::FunctionBuilder::current()->call(
            Type::of<bool>(), CallOp::RAY_TRACING_TRACE_ANY,
            {_expression, ray.expression(), mask.expression()}));
}

RayQueryAll Expr<Accel>::query_all(Expr<Ray> ray, Expr<uint> mask, RayCullMode cull_mode) const noexcept {
    return {_expression, ray.expression(), mask.expression(), cull_mode};
}

RayQueryAny Expr<Accel>::query_any(Expr<Ray> ray, Expr<uint> mask,RayCullMode cull_mode) const noexcept {
    return {_expression, ray.expression(), mask.expression(), cull_mode};
}

Var<float4x4> Expr<Accel>::instance_transform(Expr<uint> instance_id) const noexcept {
    return def<float4x4>(
        detail::FunctionBuilder::current()->call(
            Type::of<float4x4>(), CallOp::RAY_TRACING_INSTANCE_TRANSFORM,
            {_expression, instance_id.expression()}));
}

Var<float4x4> Expr<Accel>::instance_transform(Expr<int> instance_id) const noexcept {
    return def<float4x4>(
        detail::FunctionBuilder::current()->call(
            Type::of<float4x4>(), CallOp::RAY_TRACING_INSTANCE_TRANSFORM,
            {_expression, instance_id.expression()}));
}

void Expr<Accel>::set_instance_transform(Expr<int> instance_id, Expr<float4x4> mat) const noexcept {
    detail::FunctionBuilder::current()->call(
        CallOp::RAY_TRACING_SET_INSTANCE_TRANSFORM,
        {_expression, instance_id.expression(), mat.expression()});
}

void Expr<Accel>::set_instance_visibility(Expr<int> instance_id, Expr<uint> vis_mask) const noexcept {
    detail::FunctionBuilder::current()->call(
        CallOp::RAY_TRACING_SET_INSTANCE_VISIBILITY,
        {_expression, instance_id.expression(), vis_mask.expression()});
}

void Expr<Accel>::set_instance_transform(Expr<uint> instance_id, Expr<float4x4> mat) const noexcept {
    detail::FunctionBuilder::current()->call(
        CallOp::RAY_TRACING_SET_INSTANCE_TRANSFORM,
        {_expression, instance_id.expression(), mat.expression()});
}

void Expr<Accel>::set_instance_visibility(Expr<uint> instance_id, Expr<uint> vis_mask) const noexcept {
    detail::FunctionBuilder::current()->call(
        CallOp::RAY_TRACING_SET_INSTANCE_VISIBILITY,
        {_expression, instance_id.expression(), vis_mask.expression()});
}

void Expr<Accel>::set_instance_opaque(Expr<int> instance_id, Expr<bool> opaque) const noexcept {
    detail::FunctionBuilder::current()->call(
        CallOp::RAY_TRACING_SET_INSTANCE_OPACITY,
        {_expression, instance_id.expression(), opaque.expression()});
}

void Expr<Accel>::set_instance_opaque(Expr<uint> instance_id, Expr<bool> opaque) const noexcept {
    detail::FunctionBuilder::current()->call(
        CallOp::RAY_TRACING_SET_INSTANCE_OPACITY,
        {_expression, instance_id.expression(), opaque.expression()});
}

}// namespace luisa::compute
