#include <luisa/xir/user.h>

namespace luisa::compute::xir {

void User::remove_operand_uses() noexcept {
    for (auto o : _operands) {
        o->remove_self();
    }
}

void User::set_operands(luisa::vector<Use *> operands) noexcept {
    remove_operand_uses();
    _operands = std::move(operands);
}

void User::set_operands(Pool &pool, luisa::span<Value *const> operands) noexcept {
    luisa::vector<Use *> operand_uses;
    operand_uses.reserve(operands.size());
    for (auto o : operands) {
        operand_uses.emplace_back(pool.create<Use>(o, this));
    }
    set_operands(std::move(operand_uses));
}

}// namespace luisa::compute::xir
