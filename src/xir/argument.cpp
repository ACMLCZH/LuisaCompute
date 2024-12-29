#include <luisa/ast/type_registry.h>
#include <luisa/xir/argument.h>

namespace luisa::compute::xir {

Argument::Argument(const Type *type, Function *parent_function) noexcept
    : DerivedValue{type}, _parent_function{parent_function} {}

void Argument::_set_parent_function(Function *func) noexcept {
    _parent_function = func;
}

}// namespace luisa::compute::xir
