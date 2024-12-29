#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

// Note: this instruction must be the terminator of a basic block.
class LC_XIR_API ReturnInst final : public DerivedTerminatorInstruction<DerivedInstructionTag::RETURN> {

public:
    static constexpr size_t operand_index_return_value = 0u;

public:
    explicit ReturnInst(Value *value = nullptr) noexcept;

    // nullptr for void return
    void set_return_value(Value *value) noexcept;
    [[nodiscard]] Value *return_value() noexcept;
    [[nodiscard]] const Value *return_value() const noexcept;
    [[nodiscard]] const Type *return_type() const noexcept;
};

}// namespace luisa::compute::xir
