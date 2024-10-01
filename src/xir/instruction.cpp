#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

Instruction::Instruction(Pool *pool,
                         const Type *type,
                         BasicBlock *parent_block,
                         const Name *name) noexcept
    : Super{pool, type, name} {
    set_parent_block(parent_block);
}

void Instruction::remove_self() noexcept {
    Super::remove_self();
    set_parent_block(nullptr);
    remove_operand_uses();
}

void Instruction::insert_before_self(Instruction *node) noexcept {
    Super::insert_before_self(node);
    node->set_parent_block(parent_block());
    node->add_operand_uses();
}

void Instruction::insert_after_self(Instruction *node) noexcept {
    Super::insert_after_self(node);
    node->set_parent_block(parent_block());
    node->add_operand_uses();
}

}// namespace luisa::compute::xir
