#pragma once

#include <luisa/ast/type_registry.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/break.h>
#include <luisa/xir/instructions/call.h>
#include <luisa/xir/instructions/cast.h>
#include <luisa/xir/instructions/comment.h>
#include <luisa/xir/instructions/continue.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/intrinsic.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/print.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/instructions/unreachable.h>

namespace luisa::compute::xir {

class LC_XIR_API Builder {

private:
    Instruction *_insertion_point = nullptr;

private:
    void _check_valid_insertion_point() const noexcept;
    [[nodiscard]] Pool *_pool_from_insertion_point() const noexcept;

    template<typename T, typename... Args>
    [[nodiscard]] auto _create_and_append_instruction(Args &&...args) noexcept {
        auto pool = _pool_from_insertion_point();
        auto inst = pool->create<T>(std::forward<Args>(args)...);
        _insertion_point->insert_after_self(inst);
        set_insertion_point(inst);
        return inst;
    }

public:
    Builder() noexcept;
    void set_insertion_point(Instruction *insertion_point) noexcept;
    void set_insertion_point(BasicBlock *block) noexcept;
    [[nodiscard]] auto insertion_point() noexcept -> Instruction * { return _insertion_point; }
    [[nodiscard]] auto insertion_point() const noexcept -> const Instruction * { return _insertion_point; }

public:
    BranchInst *if_(Value *cond) noexcept;
    SwitchInst *switch_(Value *value) noexcept;
    LoopInst *loop() noexcept;

    BreakInst *break_() noexcept;
    ContinueInst *continue_() noexcept;
    UnreachableInst *unreachable_() noexcept;
    ReturnInst *return_(Value *value) noexcept;
    ReturnInst *return_void() noexcept;

    CallInst *call(const Type *type, Value *callee, luisa::span<Value *const> arguments) noexcept;
    CallInst *call(const Type *type, Value *callee, std::initializer_list<Value *> arguments) noexcept;

    IntrinsicInst *call(const Type *type, IntrinsicOp op, luisa::span<Value *const> arguments) noexcept;
    IntrinsicInst *call(const Type *type, IntrinsicOp op, std::initializer_list<Value *> arguments) noexcept;

    CastInst *static_cast_(const Type *type, Value *value) noexcept;
    CastInst *bit_cast_(const Type *type, Value *value) noexcept;

    PhiInst *phi(const Type *type, luisa::span<const PhiIncoming> incomings = {}) noexcept;
    PhiInst *phi(const Type *type, std::initializer_list<PhiIncoming> incomings) noexcept;

    PrintInst *print(luisa::string format, luisa::span<Value *const> values) noexcept;
    PrintInst *print(luisa::string format, std::initializer_list<Value *> values) noexcept;

    AllocaInst *alloca_(const Type *type, AllocSpace space) noexcept;
    AllocaInst *alloca_local(const Type *type) noexcept;
    AllocaInst *alloca_shared(const Type *type) noexcept;

    GEPInst *gep(const Type *type, Value *base, luisa::span<Value *const> indices) noexcept;
    GEPInst *gep(const Type *type, Value *base, std::initializer_list<Value *> indices) noexcept;

    LoadInst *load(const Type *type, Value *variable) noexcept;
    StoreInst *store(Value *variable, Value *value) noexcept;

    CommentInst *comment(luisa::string text) noexcept;

    [[nodiscard]] Constant *const_(const Type *type, const void *data) const noexcept;

    template<typename T>
    [[nodiscard]] Constant *const_(const T &value) noexcept {
        return const_(Type::of<T>(), &value);
    }
};

}// namespace luisa::compute::xir
