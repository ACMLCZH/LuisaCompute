#include <luisa/core/logging.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/instructions/ray_query.h>

namespace luisa::compute::xir {

RayQueryInst::RayQueryInst(Value *query_object) noexcept {
    auto on_surface_candidate_block = static_cast<Value *>(nullptr);
    auto on_procedural_candidate_block = static_cast<Value *>(nullptr);
    auto operands = std::array{query_object, on_surface_candidate_block, on_procedural_candidate_block};
    LUISA_DEBUG_ASSERT(operands[operand_index_query_object] == query_object, "Invalid query object operand.");
    set_operands(operands);
}

void RayQueryInst::set_query_object(Value *query_object) noexcept {
    set_operand(operand_index_query_object, query_object);
}

void RayQueryInst::set_on_surface_candidate_block(BasicBlock *block) noexcept {
    set_operand(operand_index_on_surface_candidate_block, block);
}

void RayQueryInst::set_on_procedural_candidate_block(BasicBlock *block) noexcept {
    set_operand(operand_index_on_procedural_candidate_block, block);
}

BasicBlock *RayQueryInst::create_on_surface_candidate_block() noexcept {
    auto block = Pool::current()->create<BasicBlock>();
    set_on_surface_candidate_block(block);
    return block;
}

BasicBlock *RayQueryInst::create_on_procedural_candidate_block() noexcept {
    auto block = Pool::current()->create<BasicBlock>();
    set_on_procedural_candidate_block(block);
    return block;
}

Value *RayQueryInst::query_object() noexcept {
    return operand(operand_index_query_object);
}

BasicBlock *RayQueryInst::on_surface_candidate_block() noexcept {
    return static_cast<BasicBlock *>(operand(operand_index_on_surface_candidate_block));
}

BasicBlock *RayQueryInst::on_procedural_candidate_block() noexcept {
    return static_cast<BasicBlock *>(operand(operand_index_on_procedural_candidate_block));
}

const Value *RayQueryInst::query_object() const noexcept {
    return const_cast<RayQueryInst *>(this)->query_object();
}

const BasicBlock *RayQueryInst::on_surface_candidate_block() const noexcept {
    return const_cast<RayQueryInst *>(this)->on_surface_candidate_block();
}

const BasicBlock *RayQueryInst::on_procedural_candidate_block() const noexcept {
    return const_cast<RayQueryInst *>(this)->on_procedural_candidate_block();
}

}// namespace luisa::compute::xir
