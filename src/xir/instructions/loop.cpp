#include <luisa/core/logging.h>
#include <luisa/ast/type_registry.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/instructions/loop.h>

namespace luisa::compute::xir {

LoopInst::LoopInst(Pool *pool) noexcept
    : DerivedTerminatorInstruction{pool} {
    set_operands(std::array{static_cast<Value *>(nullptr)});
}

void LoopInst::set_prepare_block(BasicBlock *block) noexcept {
    set_operand(operand_index_prepare_block, block);
}

void LoopInst::set_body_block(BasicBlock *block) noexcept {
    _body_block = block;
}

void LoopInst::set_update_block(BasicBlock *block) noexcept {
    _update_block = block;
}

BasicBlock *LoopInst::create_prepare_block(bool overwrite_existing) noexcept {
    LUISA_ASSERT(prepare_block() == nullptr || overwrite_existing,
                 "Prepare block already exists.");
    auto new_block = pool()->create<BasicBlock>();
    set_prepare_block(new_block);
    return new_block;
}

BasicBlock *LoopInst::create_body_block(bool overwrite_existing) noexcept {
    LUISA_ASSERT(body_block() == nullptr || overwrite_existing,
                 "Body block already exists.");
    auto new_block = pool()->create<BasicBlock>();
    set_body_block(new_block);
    return new_block;
}

BasicBlock *LoopInst::create_update_block(bool overwrite_existing) noexcept {
    LUISA_ASSERT(update_block() == nullptr || overwrite_existing,
                 "Update block already exists.");
    auto new_block = pool()->create<BasicBlock>();
    set_update_block(new_block);
    return new_block;
}

BasicBlock *LoopInst::prepare_block() noexcept {
    return static_cast<BasicBlock *>(operand(operand_index_prepare_block));
}

const BasicBlock *LoopInst::prepare_block() const noexcept {
    return const_cast<LoopInst *>(this)->prepare_block();
}

BasicBlock *LoopInst::body_block() noexcept {
    return _body_block;
}

const BasicBlock *LoopInst::body_block() const noexcept {
    return _body_block;
}

BasicBlock *LoopInst::update_block() noexcept {
    return _update_block;
}

const BasicBlock *LoopInst::update_block() const noexcept {
    return _update_block;
}

}// namespace luisa::compute::xir
