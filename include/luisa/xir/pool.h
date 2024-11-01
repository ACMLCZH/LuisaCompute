#pragma once

#include <luisa/core/dll_export.h>

#include <luisa/core/concepts.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::xir {

class Pool;

class LC_XIR_API PooledObject {

protected:
    PooledObject() noexcept = default;

public:
    virtual ~PooledObject() noexcept = default;

    // make the object pinned to its memory location
    PooledObject(PooledObject &&) noexcept = delete;
    PooledObject(const PooledObject &) noexcept = delete;
    PooledObject &operator=(PooledObject &&) noexcept = delete;
    PooledObject &operator=(const PooledObject &) noexcept = delete;
};

class LC_XIR_API Pool : public concepts::Noncopyable {

private:
    luisa::vector<PooledObject *> _objects;

public:
    explicit Pool(size_t init_cap = 0u) noexcept;
    ~Pool() noexcept;

public:
    static void push_current(Pool *pool) noexcept;
    static Pool *pop_current() noexcept;
    [[nodiscard]] static Pool *current() noexcept;

    template<typename F>
    decltype(auto) with_current(F &&f);

public:
    template<typename T, typename... Args>
        requires std::derived_from<T, PooledObject>
    [[nodiscard]] T *create(Args &&...args) {
        auto object = luisa::new_with_allocator<T>(std::forward<Args>(args)...);
        _objects.emplace_back(object);
        return object;
    }
};

class LC_XIR_API PoolGuard {

private:
    Pool *_pool;

public:
    explicit PoolGuard(Pool *pool) noexcept;
    ~PoolGuard() noexcept;

    PoolGuard(PoolGuard &&) noexcept = delete;
    PoolGuard(const PoolGuard &) noexcept = delete;
    PoolGuard &operator=(PoolGuard &&) noexcept = delete;
    PoolGuard &operator=(const PoolGuard &) noexcept = delete;
};

template<typename F>
decltype(auto) Pool::with_current(F &&f) {
    PoolGuard guard{this};
    return std::forward<F>(f)();
}

}// namespace luisa::compute::xir
