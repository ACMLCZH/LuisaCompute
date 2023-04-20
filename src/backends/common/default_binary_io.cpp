#include <core/stl/filesystem.h>
#include <backends/common/default_binary_io.h>
#include <runtime/context.h>
#include <runtime/context_paths.h>
#include <core/logging.h>
namespace luisa::compute {

LockedBinaryFileStream::LockedBinaryFileStream(DefaultBinaryIO const *binary_io, ::FILE *file, size_t length, const luisa::string &path, DefaultBinaryIO::MapIndex &&idx) noexcept
    : _stream{file, length},
      _binary_io{binary_io},
      _idx{std::move(idx)} {}

LockedBinaryFileStream::~LockedBinaryFileStream() noexcept {
    _binary_io->_unlock(_idx, false);
}

luisa::unique_ptr<BinaryStream> DefaultBinaryIO::_read(luisa::string const &file_path) const noexcept {
    auto idx = _lock(file_path, false);
    auto file = std::fopen(file_path.c_str(), "rb");
    if (file) {
        auto length = BinaryFileStream::seek_len(file);
        if (length == 0) [[unlikely]] {
            _unlock(idx, false);
            return nullptr;
        }
        return luisa::make_unique<LockedBinaryFileStream>(this, file, length, file_path, std::move(idx));
    } else {
        _unlock(idx, false);
        LUISA_INFO("Read file {} failed.", file_path);
        return nullptr;
    }
}

DefaultBinaryIO::MapIndex DefaultBinaryIO::_lock(luisa::string const &name, bool is_write) const noexcept {
    MapIndex iter;
    FileMutex *ptr;
    {
        std::lock_guard lck{_global_mtx};
        iter = _mutex_map.emplace(name);
        ptr = &iter.value();
    }
    if (is_write) {
        ptr->mtx.lock();
    } else {
        ptr->mtx.lock_shared();
    }
    return iter;
}

void DefaultBinaryIO::_unlock(MapIndex const &idx, bool is_write) const noexcept {
    auto &v = idx.value();
    if (is_write) {
        v.mtx.unlock();
    } else {
        v.mtx.unlock_shared();
    }
    if ((--v.ref_count) == 0) {
        std::lock_guard lck{_global_mtx};
        _mutex_map.remove(idx);
    }
}

void DefaultBinaryIO::_write(luisa::string const &file_path, luisa::span<std::byte const> data) const noexcept {
    auto idx = _lock(file_path, true);
    auto disposer = vstd::scope_exit([&]() { _unlock(idx, true); });
    auto f = fopen(file_path.c_str(), "wb");
    if (f) [[likely]] {
#ifdef _WIN32
#define LUISA_FWRITE _fwrite_nolock
#define LUISA_FCLOSE _fclose_nolock
#else
#define LUISA_FWRITE fwrite
#define LUISA_FCLOSE fclose
#endif
        LUISA_FWRITE(data.data(), data.size(), 1, f);
        LUISA_FCLOSE(f);
#undef LUISA_FWRITE
#undef LUISA_FCLOSE
    } else {
        LUISA_WARNING("Write file {} failed.", file_path);
    }
}

DefaultBinaryIO::DefaultBinaryIO(const Context &ctx) noexcept : _ctx(ctx) {
    if (!std::filesystem::exists(_ctx.paths().cache_directory())) {
        LUISA_INFO("Created cache directory.");
        std::filesystem::create_directories(_ctx.paths().cache_directory());
    }
    if (!std::filesystem::exists(_ctx.paths().data_directory())) {
        LUISA_INFO("Created data directory.");
        std::filesystem::create_directories(_ctx.paths().data_directory());
    }
}

luisa::unique_ptr<BinaryStream> DefaultBinaryIO::read_shader_bytecode(luisa::string_view name) const noexcept {
    std::filesystem::path local_path{name};
    if (local_path.is_absolute()) {
        return _read(luisa::to_string(name));
    }
    auto file_path = luisa::to_string(_ctx.paths().runtime_directory() / name);
    return _read(file_path);
}

luisa::unique_ptr<BinaryStream> DefaultBinaryIO::read_shader_cache(luisa::string_view name) const noexcept {
    auto file_path = luisa::to_string(_ctx.paths().cache_directory() / name);
    return _read(file_path);
}

luisa::unique_ptr<BinaryStream> DefaultBinaryIO::read_internal_shader(luisa::string_view name) const noexcept {
    auto file_path = luisa::to_string(_ctx.paths().data_directory() / name);
    return _read(file_path);
}

void DefaultBinaryIO::write_shader_bytecode(luisa::string_view name, luisa::span<std::byte const> data) const noexcept {
    std::filesystem::path local_path{name};
    if (local_path.is_absolute()) {
        _write(luisa::to_string(name), data);
        return;
    }
    auto file_path = luisa::to_string(_ctx.paths().runtime_directory() / name);
    _write(file_path, data);
}

void DefaultBinaryIO::write_shader_cache(luisa::string_view name, luisa::span<std::byte const> data) const noexcept {
    auto file_path = luisa::to_string(_ctx.paths().cache_directory() / name);
    _write(file_path, data);
}

void DefaultBinaryIO::write_internal_shader(luisa::string_view name, luisa::span<std::byte const> data) const noexcept {
    auto file_path = luisa::to_string(_ctx.paths().data_directory() / name);
    _write(file_path, data);
}

}// namespace luisa::compute