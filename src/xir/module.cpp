#include <luisa/core/logging.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/function.h>
#include <luisa/xir/module.h>

namespace luisa::compute::xir {

Module::Module() noexcept = default;

Function *Module::create_kernel() noexcept {
    auto f = Pool::current()->create<Function>(FunctionTag::KERNEL, nullptr);
    add_function(f);
    return f;
}

Function *Module::create_callable(const Type *ret_type) noexcept {
    auto f = Pool::current()->create<Function>(FunctionTag::CALLABLE, ret_type);
    add_function(f);
    return f;
}

void Module::add_constant(Constant *constant) noexcept {
    LUISA_DEBUG_ASSERT(!constant->is_linked(), "Constant is already linked.");
    _constants.insert_front(constant);
}

Constant *Module::create_constant(const Type *type, const void *data) noexcept {
    auto c = Pool::current()->create<Constant>(type, data);
    add_constant(c);
    return c;
}

void Module::add_function(Function *function) noexcept {
    LUISA_DEBUG_ASSERT(!function->is_linked(), "Function is already linked.");
    _functions.insert_front(function);
}

}// namespace luisa::compute::xir
