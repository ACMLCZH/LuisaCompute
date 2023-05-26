#pragma once

#include <backends/ext/dstorage_ext_interface.h>
#include <vstl/common.h>

namespace lc::validation {

using namespace luisa::compute;

class DStorageExtImpl : public DStorageExt, public vstd::IOperatorNewBase {
    DStorageExt *_impl;

public:
    DeviceInterface *device() const noexcept override;
    FileCreationInfo open_file_handle(luisa::string_view path) noexcept override;
    void close_file_handle(uint64_t handle) noexcept override;
    PinnedMemoryInfo pin_host_memory(void *ptr, size_t size_bytes) noexcept override;
    void unpin_host_memory(uint64_t handle) noexcept override;
    ResourceCreationInfo create_stream_handle() noexcept override;
    void set_config(bool hdd) noexcept override {
        _impl->set_config(hdd);
    }
    DStorageExtImpl(DStorageExt *ext);
    void compress(const void *data, size_t size_bytes,
                  Compression algorithm,
                  CompressionQuality quality,
                  vector<std::byte> &result) noexcept override;
};

}// namespace lc::validation
