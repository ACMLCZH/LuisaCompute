#include <luisa/core/logging.h>
#include <luisa/xir/instructions/cast.h>

namespace luisa::compute::xir {

CastInst::CastInst(Pool *pool, const Type *target_type,
                   CastOp op, Value *value) noexcept
    : Instruction{pool, target_type}, _op{op} {
    auto operands = std::array{value};
    set_operands(operands);
}

Value *CastInst::value() noexcept {
    return operand(0);
}

const Value *CastInst::value() const noexcept {
    return operand(0);
}

void CastInst::set_op(CastOp op) noexcept {
    _op = op;
}

void CastInst::set_value(Value *value) noexcept {
    set_operand(0, value);
}

}// namespace luisa::compute::xir
