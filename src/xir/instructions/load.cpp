#include <luisa/xir/instructions/load.h>

namespace luisa::compute::xir {

LoadInst::LoadInst(Pool *pool, const Type *type, Value *variable) noexcept
    : DerivedInstruction{pool, type} {
    auto oprands = std::array{variable};
    set_operands(oprands);
}

Value *LoadInst::variable() noexcept {
    return operand(0);
}

const Value *LoadInst::variable() const noexcept {
    return operand(0);
}

void LoadInst::set_variable(Value *variable) noexcept {
    return set_operand(0, variable);
}

}// namespace luisa::compute::xir
