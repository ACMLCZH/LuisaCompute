#pragma once

#include <runtime/image.h>
#include <runtime/sparse_texture.h>
namespace luisa::compute {
template<typename T>
class SparseImage final : public SparseTexture {
private:
    uint2 _size{};
    uint32_t _mip_levels{};
    PixelStorage _storage{};

public:
    using Resource::operator bool;
    SparseImage(SparseImage &&) noexcept = default;
    SparseImage(const SparseImage &) noexcept = delete;
    SparseImage &operator=(SparseImage &&rhs) noexcept {
        _move_from(std::move(rhs));
        return *this;
    }
    SparseImage &operator=(const SparseImage &) noexcept = delete;
    [[nodiscard]] auto view(uint32_t level) const noexcept {
        if (level >= _mip_levels) [[unlikely]] {
            detail::error_image_invalid_mip_levels(level, _mip_levels);
        }
        auto mip_size = luisa::max(_size >> level, 1u);
        return ImageView<T>{handle(), _storage, level, mip_size};
    }
    // properties
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto mip_levels() const noexcept { return _mip_levels; }
    [[nodiscard]] auto storage() const noexcept { return _storage; }
    [[nodiscard]] auto format() const noexcept { return pixel_storage_to_format<T>(_storage); }
    [[nodiscard]] auto view() const noexcept { return view(0u); }
};
}// namespace luisa::compute