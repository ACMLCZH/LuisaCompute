#pragma once

#include <luisa/xir/user.h>

namespace luisa::compute::xir {

class BasicBlock;

enum struct DerivedInstructionTag {

    /* utility instructions */
    SENTINEL,// sentinels in instruction list
    COMMENT, // comments

    /* control flow instructions */
    UNREACHABLE,// basic block terminator: unreachable
    BRANCH,     // basic block terminator: conditional branches
    SWITCH,     // basic block terminator: switch branches
    LOOP,       // basic block terminator: loops
    BREAK,      // basic block terminator: break (removed after control flow normalization)
    CONTINUE,   // basic block terminator: continue (removed after control flow normalization)
    RETURN,     // basic block terminator: return (early returns are removed after control flow normalization)
    PHI,        // basic block beginning: phi nodes

    // variable instructions
    ALLOCA,
    LOAD,
    STORE,
    GEP,

    /* other instructions */
    CALL,     // user or external function calls
    INTRINSIC,// intrinsic function calls
    CAST,     // type casts
    PRINT,    // kernel print
    AUTO_DIFF,// automatic differentiation
    RAY_QUERY,// ray queries
};

class LC_XIR_API Instruction : public IntrusiveNode<Instruction, User> {

private:
    friend BasicBlock;
    BasicBlock *_parent_block = nullptr;

protected:
    void _set_parent_block(BasicBlock *block) noexcept;
    void _remove_self_from_operand_use_lists() noexcept;
    void _add_self_to_operand_use_lists() noexcept;
    [[nodiscard]] bool _should_add_self_to_operand_use_lists() const noexcept override;

public:
    explicit Instruction(Pool *pool, const Type *type = nullptr,
                         const Name *name = nullptr) noexcept;
    [[nodiscard]] virtual DerivedInstructionTag derived_instruction_tag() const noexcept {
        return DerivedInstructionTag::SENTINEL;
    }
    [[nodiscard]] DerivedValueTag derived_value_tag() const noexcept final {
        return DerivedValueTag::INSTRUCTION;
    }

    void remove_self() noexcept override;
    void insert_before_self(Instruction *node) noexcept override;
    void insert_after_self(Instruction *node) noexcept override;
    void replace_self_with(Instruction *node) noexcept;

    [[nodiscard]] BasicBlock *parent_block() noexcept { return _parent_block; }
    [[nodiscard]] const BasicBlock *parent_block() const noexcept { return _parent_block; }
};

using InstructionList = InlineIntrusiveList<Instruction>;

}// namespace luisa::compute::xir
