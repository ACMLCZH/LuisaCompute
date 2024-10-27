#include <luisa/core/logging.h>
#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

Instruction::Instruction(Pool *pool, const Type *type) noexcept
    : Super{pool, type} {}

void Instruction::_set_parent_block(BasicBlock *block) noexcept {
    _parent_block = block;
}

bool Instruction::_should_add_self_to_operand_use_lists() const noexcept {
    return is_linked();
}

void Instruction::_remove_self_from_operand_use_lists() noexcept {
    for (auto use : operand_uses()) {
        if (use->value() != nullptr) {
            use->remove_self();
        }
    }
}

void Instruction::_add_self_to_operand_use_lists() noexcept {
    for (auto use : operand_uses()) {
        LUISA_DEBUG_ASSERT(!use->is_linked(), "Use already linked.");
        if (auto value = use->value()) {
            value->use_list().insert_front(use);
        }
    }
}

void Instruction::remove_self() noexcept {
    Super::remove_self();
    _remove_self_from_operand_use_lists();
    _set_parent_block(nullptr);
}

void Instruction::insert_before_self(Instruction *node) noexcept {
    Super::insert_before_self(node);
    node->_add_self_to_operand_use_lists();
    node->_set_parent_block(_parent_block);
}

void Instruction::insert_after_self(Instruction *node) noexcept {
    Super::insert_after_self(node);
    node->_add_self_to_operand_use_lists();
    node->_set_parent_block(_parent_block);
}

void Instruction::replace_self_with(Instruction *node) noexcept {
    replace_all_uses_with(node);
    insert_before_self(node);
    remove_self();
}

}// namespace luisa::compute::xir
