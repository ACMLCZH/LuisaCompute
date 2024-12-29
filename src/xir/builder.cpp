#include <luisa/core/logging.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>

namespace luisa::compute::xir {

void Builder::_check_valid_insertion_point() const noexcept {
    LUISA_ASSERT(_insertion_point != nullptr, "Invalid insertion point.");
}

Builder::Builder() noexcept = default;

IfInst *Builder::if_(Value *cond) noexcept {
    return _create_and_append_instruction<IfInst>(cond);
}

SwitchInst *Builder::switch_(Value *value) noexcept {
    return _create_and_append_instruction<SwitchInst>(value);
}

LoopInst *Builder::loop() noexcept {
    return _create_and_append_instruction<LoopInst>();
}

SimpleLoopInst *Builder::simple_loop() noexcept {
    return _create_and_append_instruction<SimpleLoopInst>();
}

BranchInst *Builder::br(BasicBlock *target) noexcept {
    auto inst = _create_and_append_instruction<BranchInst>();
    inst->set_target_block(target);
    return inst;
}

ConditionalBranchInst *Builder::cond_br(Value *cond, BasicBlock *true_target, BasicBlock *false_target) noexcept {
    auto inst = _create_and_append_instruction<ConditionalBranchInst>(cond);
    inst->set_true_target(true_target);
    inst->set_false_target(false_target);
    return inst;
}

BreakInst *Builder::break_(BasicBlock *target_block) noexcept {
    auto inst = _create_and_append_instruction<BreakInst>();
    inst->set_target_block(target_block);
    return inst;
}

ContinueInst *Builder::continue_(BasicBlock *target_block) noexcept {
    auto inst = _create_and_append_instruction<ContinueInst>();
    inst->set_target_block(target_block);
    return inst;
}

UnreachableInst *Builder::unreachable_(luisa::string_view message) noexcept {
    return _create_and_append_instruction<UnreachableInst>(luisa::string{message});
}

AssertInst *Builder::assert_(Value *condition, luisa::string_view message) noexcept {
    return _create_and_append_instruction<AssertInst>(condition, luisa::string{message});
}

ReturnInst *Builder::return_(Value *value) noexcept {
    return _create_and_append_instruction<ReturnInst>(value);
}

ReturnInst *Builder::return_void() noexcept {
    return _create_and_append_instruction<ReturnInst>();
}

CallInst *Builder::call(const Type *type, Value *callee, luisa::span<Value *const> arguments) noexcept {
    return _create_and_append_instruction<CallInst>(type, callee, arguments);
}

CallInst *Builder::call(const Type *type, Value *callee, std::initializer_list<Value *> arguments) noexcept {
    return _create_and_append_instruction<CallInst>(type, callee, luisa::span{arguments.begin(), arguments.end()});
}

IntrinsicInst *Builder::call(const Type *type, IntrinsicOp op, std::initializer_list<Value *> arguments) noexcept {
    return _create_and_append_instruction<IntrinsicInst>(type, op, luisa::span{arguments.begin(), arguments.end()});
}

PhiInst *Builder::phi(const Type *type, std::initializer_list<PhiIncoming> incomings) noexcept {
    auto inst = _create_and_append_instruction<PhiInst>(type);
    for (auto incoming : incomings) { inst->add_incoming(incoming.value, incoming.block); }
    return inst;
}

PrintInst *Builder::print(luisa::string format, std::initializer_list<Value *> values) noexcept {
    return _create_and_append_instruction<PrintInst>(std::move(format), luisa::span{values.begin(), values.end()});
}

AllocaInst *Builder::alloca_(const Type *type, AllocSpace space) noexcept {
    return _create_and_append_instruction<AllocaInst>(type, space);
}

AllocaInst *Builder::alloca_local(const Type *type) noexcept {
    return alloca_(type, AllocSpace::LOCAL);
}

AllocaInst *Builder::alloca_shared(const Type *type) noexcept {
    return alloca_(type, AllocSpace::SHARED);
}

GEPInst *Builder::gep(const Type *type, Value *base, std::initializer_list<Value *> indices) noexcept {
    return _create_and_append_instruction<GEPInst>(type, base, luisa::span{indices.begin(), indices.end()});
}

IntrinsicInst *Builder::call(const Type *type, IntrinsicOp op, luisa::span<Value *const> arguments) noexcept {
    return _create_and_append_instruction<IntrinsicInst>(type, op, arguments);
}

CastInst *Builder::static_cast_(const Type *type, Value *value) noexcept {
    return _create_and_append_instruction<CastInst>(type, CastOp::STATIC_CAST, value);
}

CastInst *Builder::bit_cast_(const Type *type, Value *value) noexcept {
    return _create_and_append_instruction<CastInst>(type, CastOp::BITWISE_CAST, value);
}

Value *Builder::static_cast_if_necessary(const Type *type, Value *value) noexcept {
    return value->type() == type ? value : static_cast_(type, value);
}

Value *Builder::bit_cast_if_necessary(const Type *type, Value *value) noexcept {
    return value->type() == type ? value : bit_cast_(type, value);
}

PhiInst *Builder::phi(const Type *type, luisa::span<const PhiIncoming> incomings) noexcept {
    auto inst = _create_and_append_instruction<PhiInst>(type);
    for (auto incoming : incomings) { inst->add_incoming(incoming.value, incoming.block); }
    return inst;
}

PrintInst *Builder::print(luisa::string format, luisa::span<Value *const> values) noexcept {
    return _create_and_append_instruction<PrintInst>(std::move(format), values);
}

GEPInst *Builder::gep(const Type *type, Value *base, luisa::span<Value *const> indices) noexcept {
    return _create_and_append_instruction<GEPInst>(type, base, indices);
}

LoadInst *Builder::load(const Type *type, Value *variable) noexcept {
    return _create_and_append_instruction<LoadInst>(type, variable);
}

StoreInst *Builder::store(Value *variable, Value *value) noexcept {
    return _create_and_append_instruction<StoreInst>(variable, value);
}

OutlineInst *Builder::outline() noexcept {
    return _create_and_append_instruction<OutlineInst>();
}

RayQueryInst *Builder::ray_query(Value *query_object) noexcept {
    return _create_and_append_instruction<RayQueryInst>(query_object);
}

void Builder::set_insertion_point(Instruction *insertion_point) noexcept {
    _insertion_point = insertion_point;
}

void Builder::set_insertion_point(BasicBlock *block) noexcept {
    auto instruction = block ? block->instructions().tail_sentinel()->prev() : nullptr;
    set_insertion_point(instruction);
}

}// namespace luisa::compute::xir
