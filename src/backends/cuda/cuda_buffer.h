//
// Created by Mike on 3/14/2023.
//

#pragma once

#include <cuda.h>

namespace luisa::compute::cuda {

class CUDABuffer {

public:
    struct Binding {
        CUdeviceptr handle;
        size_t size;
    };

private:
    CUdeviceptr _handle;
    size_t _size;

public:
    explicit CUDABuffer(size_t size_bytes) noexcept;
    ~CUDABuffer() noexcept;
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] Binding binding(size_t offset, size_t size) const noexcept;
};

}// namespace luisa::compute::cuda