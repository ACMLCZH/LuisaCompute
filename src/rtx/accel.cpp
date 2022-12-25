//
// Created by Mike Smith on 2021/6/24.
//

#include <ast/function_builder.h>
#include <runtime/shader.h>
#include <rtx/accel.h>
#include <vstl/pdqsort.h>
#include <core/logging.h>

namespace luisa::compute {

namespace detail {

ShaderInvokeBase &ShaderInvokeBase::operator<<(const Accel &accel) noexcept {
    _command->encode_accel(accel.handle());
    return *this;
}

}// namespace detail

Accel Device::create_accel(AccelBuildOption option) noexcept {
    return _create<Accel>(option);
}

Accel::Accel(DeviceInterface *device, AccelBuildOption const &option) noexcept
    : Resource{device, Resource::Tag::ACCEL, device->create_accel(option.hint, option.allow_compact, option.allow_update)} {}

luisa::unique_ptr<Command> Accel::update(bool build_accel, Accel::BuildRequest request) noexcept {
    if (_mesh_handles.empty()) { LUISA_ERROR_WITH_LOCATION(
        "Building acceleration structure without instances."); }
    // collect modifications
    luisa::vector<Accel::Modification> modifications(_modifications.size());
    std::transform(_modifications.cbegin(), _modifications.cend(), modifications.begin(),
                   [](auto &&pair) noexcept { return pair.second; });
    _modifications.clear();
    pdqsort(modifications.begin(), modifications.end(),
            [](auto &&lhs, auto &&rhs) noexcept { return lhs.index < rhs.index; });
    return AccelBuildCommand::create(handle(), static_cast<uint>(_mesh_handles.size()),
                                     request, std::move(modifications), build_accel);
}

#ifndef LC_DISABLE_DSL

Var<Hit> Accel::trace_closest(Expr<Ray> ray) const noexcept {
    return Expr<Accel>{*this}.trace_closest(ray);
}

Var<bool> Accel::trace_any(Expr<Ray> ray) const noexcept {
    return Expr<Accel>{*this}.trace_any(ray);
}
RayQuery Accel::trace_all(Expr<Ray> ray) const noexcept {
    return Expr<Accel>{*this}.trace_all(ray);
}

Var<float4x4> Accel::instance_transform(Expr<int> instance_id) const noexcept {
    return Expr<Accel>{*this}.instance_transform(instance_id);
}

Var<float4x4> Accel::instance_transform(Expr<uint> instance_id) const noexcept {
    return Expr<Accel>{*this}.instance_transform(instance_id);
}

void Accel::set_instance_transform(Expr<int> instance_id, Expr<float4x4> mat) const noexcept {
    Expr<Accel>{*this}.set_instance_transform(instance_id, mat);
}

void Accel::set_instance_transform(Expr<uint> instance_id, Expr<float4x4> mat) const noexcept {
    Expr<Accel>{*this}.set_instance_transform(instance_id, mat);
}

void Accel::set_instance_visibility(Expr<int> instance_id, Expr<bool> vis) const noexcept {
    Expr<Accel>{*this}.set_instance_visibility(instance_id, vis);
}

void Accel::set_instance_visibility(Expr<uint> instance_id, Expr<bool> vis) const noexcept {
    Expr<Accel>{*this}.set_instance_visibility(instance_id, vis);
}
void Accel::set_instance_opaque(Expr<int> instance_id, Expr<bool> opaque) const noexcept {
    Expr<Accel>{*this}.set_instance_opaque(instance_id, opaque);
}

void Accel::set_instance_opaque(Expr<uint> instance_id, Expr<bool> opaque) const noexcept {
    Expr<Accel>{*this}.set_instance_opaque(instance_id, opaque);
}

#endif

void Accel::emplace_back_handle(uint64_t mesh, float4x4 const &transform, bool visible, bool opaque) noexcept {
    auto index = static_cast<uint>(_mesh_handles.size());
    Modification modification{index};
    modification.set_mesh(mesh);
    modification.set_transform(transform);
    modification.set_visibility(visible);
    modification.set_opaque(opaque);
    _modifications[index] = modification;
    _mesh_handles.emplace_back(mesh);
}

void Accel::pop_back() noexcept {
    if (auto n = _mesh_handles.size()) {
        _mesh_handles.pop_back();
        _modifications.erase(n - 1u);
    } else {
        LUISA_WARNING_WITH_LOCATION(
            "Ignoring pop-back operation on empty accel.");
    }
}
void Accel::set_handle(size_t index, uint64_t mesh, float4x4 const &transform, bool visible, bool opaque) noexcept {
    if (index >= size()) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Invalid index {} in accel #{}.",
            index, handle());
    } else {
        Modification modification{static_cast<uint>(index)};
        modification.set_transform(transform);
        modification.set_visibility(visible);
        modification.set_opaque(opaque);
        if (mesh != _mesh_handles[index]) [[likely]] {
            modification.set_mesh(mesh);
            _mesh_handles[index] = mesh;
        }
        _modifications[index] = modification;
    }
}

void Accel::set_transform_on_update(size_t index, float4x4 transform) noexcept {
    if (index >= size()) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Invalid index {} in accel #{}.",
            index, handle());
    } else {
        auto [iter, _] = _modifications.try_emplace(
            index, Modification{static_cast<uint>(index)});
        iter->second.set_transform(transform);
    }
}

void Accel::set_opaque_on_update(size_t index, bool opaque) noexcept {
    if (index >= size()) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Invalid index {} in accel #{}.",
            index, handle());
    } else {
        auto [iter, _] = _modifications.try_emplace(
            index, Modification{static_cast<uint>(index)});
        iter->second.set_opaque(opaque);
    }
}

void Accel::set_visibility_on_update(size_t index, bool visible) noexcept {
    if (index >= size()) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Invalid index {} in accel #{}.",
            index, handle());
    } else {
        auto [iter, _] = _modifications.try_emplace(
            index, Modification{static_cast<uint>(index)});
        iter->second.set_visibility(visible);
    }
}

#ifndef LC_DISABLE_DSL

Expr<Accel>::Expr(const RefExpr *expr) noexcept
    : _expression{expr} {}
Expr<Accel>::Expr(const Accel &accel) noexcept
    : _expression{detail::FunctionBuilder::current()->accel_binding(
          accel.handle())} {}
Var<Hit> Expr<Accel>::trace_closest(Expr<Ray> ray) const noexcept {
    return def<Hit>(
        detail::FunctionBuilder::current()->call(
            Type::of<Hit>(), CallOp::RAY_TRACING_TRACE_CLOSEST,
            {_expression, ray.expression()}));
}
Var<bool> Expr<Accel>::trace_any(Expr<Ray> ray) const noexcept {
    return def<bool>(
        detail::FunctionBuilder::current()->call(
            Type::of<bool>(), CallOp::RAY_TRACING_TRACE_ANY,
            {_expression, ray.expression()}));
}
RayQuery Expr<Accel>::trace_all(Expr<Ray> ray) const noexcept {
    return RayQuery(
        detail::FunctionBuilder::current()->call(
            Type::of<RayQuery>(), CallOp::RAY_TRACING_TRACE_ALL,
            {_expression, ray.expression()}));
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
void Expr<Accel>::set_instance_visibility(Expr<int> instance_id, Expr<bool> vis) const noexcept {
    detail::FunctionBuilder::current()->call(
        CallOp::RAY_TRACING_SET_INSTANCE_VISIBILITY,
        {_expression, instance_id.expression(), vis.expression()});
}
void Expr<Accel>::set_instance_transform(Expr<uint> instance_id, Expr<float4x4> mat) const noexcept {
    detail::FunctionBuilder::current()->call(
        CallOp::RAY_TRACING_SET_INSTANCE_TRANSFORM,
        {_expression, instance_id.expression(), mat.expression()});
}
void Expr<Accel>::set_instance_visibility(Expr<uint> instance_id, Expr<bool> vis) const noexcept {
    detail::FunctionBuilder::current()->call(
        CallOp::RAY_TRACING_SET_INSTANCE_VISIBILITY,
        {_expression, instance_id.expression(), vis.expression()});
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

#endif

}// namespace luisa::compute
