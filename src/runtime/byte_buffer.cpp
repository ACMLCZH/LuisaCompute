#include <luisa/runtime/byte_buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/core/logging.h>

namespace luisa::compute {
namespace detail {
LC_RUNTIME_API void error_buffer_size_not_aligned(size_t align) noexcept {
    LUISA_ERROR_WITH_LOCATION("Buffer size must be aligned to {}.", align);
}
}// namespace detail
ByteBuffer::ByteBuffer(DeviceInterface *device, const BufferCreationInfo &info) noexcept
    : Resource{device, Tag::BUFFER, info},
      _size_bytes{info.total_size_bytes} {}

ByteBuffer::ByteBuffer(DeviceInterface *device, size_t size_bytes) noexcept
    : ByteBuffer{
          device,
          [&] {
              if (size_bytes == 0) [[unlikely]] {
                  detail::error_buffer_size_is_zero();
              }
              if ((size_bytes & 3) != 0) [[unlikely]] {
                  detail::error_buffer_size_not_aligned(4);
              }
              return device->create_buffer(Type::of<uint32_t>(), size_bytes / 4u);
          }()} {}
ByteBuffer::~ByteBuffer() noexcept {
    if (*this) { device()->destroy_buffer(handle()); }
}
ByteBuffer Device::create_byte_buffer(size_t byte_size) noexcept {
    return ByteBuffer{impl(), byte_size};
}
}// namespace luisa::compute