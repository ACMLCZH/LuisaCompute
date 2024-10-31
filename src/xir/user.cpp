#include <luisa/core/logging.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/user.h>

namespace luisa::compute::xir {

void User::set_operand(size_t index, Value *value) noexcept {
    LUISA_DEBUG_ASSERT(index < _operands.size(), "Index out of range.");
    LUISA_DEBUG_ASSERT(_operands[index] != nullptr && _operands[index]->user() == this, "Invalid operand.");
    _operands[index]->set_value(value, _should_add_self_to_operand_use_lists());
}

Use *User::operand_use(size_t index) noexcept {
    LUISA_DEBUG_ASSERT(index < _operands.size(), "Index out of range.");
    return _operands[index];
}

const Use *User::operand_use(size_t index) const noexcept {
    LUISA_DEBUG_ASSERT(index < _operands.size(), "Index out of range.");
    return _operands[index];
}

Value *User::operand(size_t index) noexcept {
    return operand_use(index)->value();
}

const Value *User::operand(size_t index) const noexcept {
    return operand_use(index)->value();
}

void User::set_operand_count(size_t n) noexcept {
    if (n < _operands.size()) {// remove redundant operands
        for (auto i = n; i < _operands.size(); i++) {
            operand_use(i)->reset_value();
        }
        _operands.resize(n);
    } else {// create new operands
        _operands.reserve(n);
        for (auto i = _operands.size(); i < n; i++) {
            auto use = pool()->create<Use>(this);
            _operands.emplace_back(use);
        }
    }
}

void User::set_operands(luisa::span<Value *const> operands) noexcept {
    set_operand_count(operands.size());
    for (auto i = 0u; i < operands.size(); i++) {
        set_operand(i, operands[i]);
    }
}

void User::add_operand(Value *value) noexcept {
    auto use = pool()->create<Use>(this);
    use->set_value(value, _should_add_self_to_operand_use_lists());
    _operands.emplace_back(use);
}

void User::insert_operand(size_t index, Value *value) noexcept {
    LUISA_DEBUG_ASSERT(index <= _operands.size(), "Index out of range.");
    auto use = pool()->create<Use>(this);
    use->set_value(value, _should_add_self_to_operand_use_lists());
    _operands.insert(_operands.cbegin() + index, use);
}

void User::remove_operand(size_t index) noexcept {
    if (index < _operands.size()) {
        _operands[index]->reset_value();
        _operands.erase(_operands.cbegin() + index);
    }
}

luisa::span<Use *> User::operand_uses() noexcept {
    return _operands;
}

luisa::span<const Use *const> User::operand_uses() const noexcept {
    return _operands;
}

size_t User::operand_count() const noexcept {
    return _operands.size();
}

}// namespace luisa::compute::xir
