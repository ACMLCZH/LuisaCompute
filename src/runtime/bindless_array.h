//
// Created by Mike Smith on 2021/4/7.
//

#pragma once

#include <core/stl/unordered_map.h>
#include <runtime/rhi/sampler.h>
#include <runtime/mipmap.h>
#include <runtime/rhi/resource.h>
#include <runtime/buffer.h>
#include <runtime/image.h>
#include <runtime/volume.h>

namespace luisa::compute {

class BindlessTexture2D;
class BindlessTexture3D;
class Command;
class Device;
class ManagedBindless;

namespace detail {
class BindlessArrayExprProxy;
}

template<typename T>
class BindlessBuffer;

template<typename T>
struct Expr;

// BindlessArray is a heap that contain references to buffer, 2d-image and 3d-image
// every element can contain one buffer, one 2d-image and one 3d-image's reference
// see test_bindless.cpp as example
class LC_RUNTIME_API BindlessArray final : public Resource {

public:
    using Modification = BindlessArrayUpdateCommand::Modification;

    struct ModSlotHash {
        using is_avalanching = void;
        [[nodiscard]] auto operator()(Modification m, uint64_t seed = hash64_default_seed) const noexcept {
            return hash_value(static_cast<size_t>(m.slot), seed);
        }
    };

    struct ModSlotEqual {
        [[nodiscard]] auto operator()(Modification lhs, Modification rhs) const noexcept {
            return lhs.slot == rhs.slot;
        }
    };

private:
    size_t _size{0u};
    // "emplace" and "remove" operations will be cached under _updates and commit in update() command
    luisa::unordered_set<Modification, ModSlotHash, ModSlotEqual> _updates;

private:
    friend class Device;
    friend class ManagedBindless;
    BindlessArray(DeviceInterface *device, size_t size) noexcept;

public:
    BindlessArray() noexcept = default;
    ~BindlessArray() noexcept override;
    using Resource::operator bool;
    BindlessArray(BindlessArray &&) noexcept = default;
    BindlessArray(BindlessArray const &) noexcept = delete;
    BindlessArray &operator=(BindlessArray &&rhs) noexcept {
        _move_from(std::move(rhs));
        return *this;
    }
    BindlessArray &operator=(BindlessArray const &) noexcept = delete;
    // properties
    [[nodiscard]] auto size() const noexcept { return _size; }
    // whether there are any stashed updates
    [[nodiscard]] auto dirty() const noexcept { return !_updates.empty(); }
    // on-update functions' operations will be committed by update()
    void emplace_buffer_on_update(size_t index, uint64_t handle, size_t offset_bytes) noexcept;
    void emplace_tex2d_on_update(size_t index, uint64_t handle, Sampler sampler) noexcept;
    void emplace_tex3d_on_update(size_t index, uint64_t handle, Sampler sampler) noexcept;
    BindlessArray &remove_buffer_on_update(size_t index) noexcept;
    BindlessArray &remove_tex2d_on_update(size_t index) noexcept;
    BindlessArray &remove_tex3d_on_update(size_t index) noexcept;

    template<typename T>
        requires is_buffer_or_view_v<std::remove_cvref_t<T>>
    auto &emplace_on_update(size_t index, T &&buffer) noexcept {
        BufferView view{buffer};
        emplace_buffer_on_update(index, view.handle(), view.offset_bytes());
        return *this;
    }

    auto &emplace_on_update(size_t index, const Image<float> &image, Sampler sampler) noexcept {
        emplace_tex2d_on_update(index, image.handle(), sampler);
        return *this;
    }

    auto &emplace_on_update(size_t index, const Volume<float> &volume, Sampler sampler) noexcept {
        emplace_tex3d_on_update(index, volume.handle(), sampler);
        return *this;
    }

    [[nodiscard]] luisa::unique_ptr<Command> update() noexcept;

    // DSL interface
    [[nodiscard]] auto operator->() const noexcept {
        return reinterpret_cast<const detail::BindlessArrayExprProxy *>(this);
    }
};

}// namespace luisa::compute
