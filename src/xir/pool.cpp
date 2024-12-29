#include <luisa/core/logging.h>
#include <luisa/xir/pool.h>

namespace luisa::compute::xir {

Pool::Pool(size_t init_cap) noexcept {
    if (init_cap != 0u) {
        _objects.reserve(init_cap);
    }
}

Pool::~Pool() noexcept {
    for (auto object : _objects) {
        luisa::delete_with_allocator(object);
    }
}

namespace detail {
[[nodiscard]] static auto thread_local_pool_stack() noexcept {
    static thread_local luisa::vector<Pool *> pools;
    return &pools;
}
}// namespace detail

void Pool::push_current(Pool *pool) noexcept {
    detail::thread_local_pool_stack()->push_back(pool);
}

Pool *Pool::pop_current() noexcept {
    LUISA_ASSERT(!detail::thread_local_pool_stack()->empty(), "No pool to pop.");
    auto pool = detail::thread_local_pool_stack()->back();
    detail::thread_local_pool_stack()->pop_back();
    return pool;
}

Pool *Pool::current() noexcept {
    LUISA_DEBUG_ASSERT(!detail::thread_local_pool_stack()->empty(), "No current pool.");
    return detail::thread_local_pool_stack()->back();
}

PoolGuard::PoolGuard(Pool *pool) noexcept
    : _pool{pool} { Pool::push_current(pool); }

PoolGuard::~PoolGuard() noexcept {
    auto popped = Pool::pop_current();
    LUISA_ASSERT(popped == _pool, "Popped pool does not match.");
}

}// namespace luisa::compute::xir
