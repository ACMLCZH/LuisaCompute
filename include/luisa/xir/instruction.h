#pragma once

#include <luisa/xir/user.h>

namespace luisa::compute::xir {

class BasicBlock;

enum struct DerivedInstructionTag {

    /* utility instructions */
    SENTINEL,// sentinels in instruction list

    /* control flow instructions */
    IF,                // basic block terminator: conditional branches
    SWITCH,            // basic block terminator: switch branches
    LOOP,              // basic block terminator: loops
    SIMPLE_LOOP,       // basic block terminator: simple (do-while) loops
    BRANCH,            // basic block terminator: unconditional branches
    CONDITIONAL_BRANCH,// basic block terminator: conditional branches
    UNREACHABLE,       // basic block terminator: unreachable
    BREAK,             // basic block terminator: break (removed after control flow normalization)
    CONTINUE,          // basic block terminator: continue (removed after control flow normalization)
    RETURN,            // basic block terminator: return (early returns are removed after control flow normalization)
    PHI,               // basic block beginning: phi nodes

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
    ASSERT,   // assertion

    OUTLINE,  // mark that the body might be outlined (e.g., for faster compilation)
    AUTO_DIFF,// automatic differentiation
    RAY_QUERY,// ray queries
};

class LC_XIR_API Instruction : public IntrusiveNode<Instruction, DerivedValue<DerivedValueTag::INSTRUCTION, User>> {

private:
    friend BasicBlock;
    BasicBlock *_parent_block = nullptr;

protected:
    void _set_parent_block(BasicBlock *block) noexcept;
    void _remove_self_from_operand_use_lists() noexcept;
    void _add_self_to_operand_use_lists() noexcept;
    [[nodiscard]] bool _should_add_self_to_operand_use_lists() const noexcept override;

public:
    explicit Instruction(const Type *type = nullptr) noexcept;
    [[nodiscard]] virtual DerivedInstructionTag derived_instruction_tag() const noexcept {
        return DerivedInstructionTag::SENTINEL;
    }

    void remove_self() noexcept override;
    void insert_before_self(Instruction *node) noexcept override;
    void insert_after_self(Instruction *node) noexcept override;
    void replace_self_with(Instruction *node) noexcept;

    [[nodiscard]] virtual bool is_terminator() noexcept { return false; }
    [[nodiscard]] BasicBlock *parent_block() noexcept { return _parent_block; }
    [[nodiscard]] const BasicBlock *parent_block() const noexcept { return _parent_block; }
};

using InstructionList = InlineIntrusiveList<Instruction>;

class LC_XIR_API TerminatorInstruction : public Instruction {
public:
    TerminatorInstruction() noexcept;
    [[nodiscard]] bool is_terminator() noexcept final { return true; }
};

// unconditional branch
class LC_XIR_API BranchTerminatorInstruction : public TerminatorInstruction {

public:
    static constexpr size_t operand_index_target = 0u;
    static constexpr size_t derived_operand_index_offset = 1u;

public:
    BranchTerminatorInstruction() noexcept;

    void set_target_block(BasicBlock *target) noexcept;
    BasicBlock *create_target_block(bool overwrite_existing = false) noexcept;

    [[nodiscard]] BasicBlock *target_block() noexcept;
    [[nodiscard]] const BasicBlock *target_block() const noexcept;
};

// conditional branch
class LC_XIR_API ConditionalBranchTerminatorInstruction : public TerminatorInstruction {

public:
    static constexpr size_t operand_index_condition = 0u;
    static constexpr size_t operand_index_true_target = 1u;
    static constexpr size_t operand_index_false_target = 2u;
    static constexpr size_t derived_operand_index_offset = 3u;

public:
    explicit ConditionalBranchTerminatorInstruction(Value *condition = nullptr) noexcept;

    void set_condition(Value *condition) noexcept;
    void set_true_target(BasicBlock *target) noexcept;
    void set_false_target(BasicBlock *target) noexcept;

    BasicBlock *create_true_block(bool overwrite_existing = false) noexcept;
    BasicBlock *create_false_block(bool overwrite_existing = false) noexcept;

    [[nodiscard]] Value *condition() noexcept;
    [[nodiscard]] const Value *condition() const noexcept;

    [[nodiscard]] BasicBlock *true_block() noexcept;
    [[nodiscard]] const BasicBlock *true_block() const noexcept;

    [[nodiscard]] BasicBlock *false_block() noexcept;
    [[nodiscard]] const BasicBlock *false_block() const noexcept;
};

template<DerivedInstructionTag tag, typename Base = Instruction>
class DerivedInstruction : public Base {
public:
    using Base::Base;

    [[nodiscard]] static constexpr DerivedInstructionTag
    static_derived_instruction_tag() noexcept { return tag; }

    [[nodiscard]] DerivedInstructionTag
    derived_instruction_tag() const noexcept final {
        return static_derived_instruction_tag();
    }
};

template<DerivedInstructionTag tag>
class DerivedTerminatorInstruction : public DerivedInstruction<tag, TerminatorInstruction> {
public:
    using DerivedInstruction<tag, TerminatorInstruction>::DerivedInstruction;
};

template<DerivedInstructionTag tag>
class DerivedBranchInstruction : public DerivedInstruction<tag, BranchTerminatorInstruction> {
public:
    using DerivedInstruction<tag, BranchTerminatorInstruction>::DerivedInstruction;
};

template<DerivedInstructionTag tag>
class DerivedConditionalBranchInstruction : public DerivedInstruction<tag, ConditionalBranchTerminatorInstruction> {
public:
    using DerivedInstruction<tag, ConditionalBranchTerminatorInstruction>::DerivedInstruction;
};

class LC_XIR_API InstructionMergeMixin {

private:
    BasicBlock *_merge_block{nullptr};

protected:
    InstructionMergeMixin() noexcept = default;
    ~InstructionMergeMixin() noexcept = default;

public:
    void set_merge_block(BasicBlock *block) noexcept { _merge_block = block; }
    [[nodiscard]] BasicBlock *merge_block() noexcept { return _merge_block; }
    [[nodiscard]] const BasicBlock *merge_block() const noexcept { return _merge_block; }
    BasicBlock *create_merge_block(bool overwrite_existing = false) noexcept;
};

}// namespace luisa::compute::xir
