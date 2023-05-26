#pragma once

#include <runtime/rhi/command.h>
#include <backends/ext/registry.h>

namespace luisa::compute {

class DStorageReadCommand : public CustomCommand {

public:
    struct FileSource {
        uint64_t handle;
        size_t offset_bytes;
        size_t size_bytes;
    };

    struct MemorySource {
        uint64_t handle;
        size_t offset_bytes;
        size_t size_bytes;
    };

    using Source = luisa::variant<
        FileSource,
        MemorySource>;

    struct BufferRequest {
        uint64_t handle;
        size_t offset_bytes;
        size_t size_bytes;
    };

    struct TextureRequest {
        uint64_t handle;
        uint32_t level;
        uint32_t size[3u];
    };

    struct MemoryRequest {
        void *data;
        size_t size_bytes;
    };

    using Request = luisa::variant<
        BufferRequest,
        TextureRequest,
        MemoryRequest>;

    using Compression = DStorageCompression;

private:
    Source _source;
    Request _request;
    Compression _compression;

public:
    DStorageReadCommand(Source source,
                        Request request,
                        Compression compression) noexcept
        : _source{source},
          _request{request},
          _compression{compression} {}
    [[nodiscard]] const auto &source() const noexcept { return _source; }
    [[nodiscard]] const auto &request() const noexcept { return _request; }
    [[nodiscard]] auto compression() const noexcept { return _compression; }
    [[nodiscard]] uint64_t uuid() const noexcept override { return to_underlying(CustomCommandUUID::DSTORAGE_READ); }
    LUISA_MAKE_COMMAND_COMMON(StreamTag::CUSTOM)
};

}// namespace luisa::compute
