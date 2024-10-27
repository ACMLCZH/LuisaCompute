#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

class LC_XIR_API LoadInst final : public Instruction {
public:
    explicit LoadInst(Pool *pool, const Type *type = nullptr,
                      Value *variable = nullptr) noexcept;
    [[nodiscard]] DerivedInstructionTag derived_instruction_tag() const noexcept override {
        return DerivedInstructionTag::LOAD;
    }
    [[nodiscard]] Value *variable() noexcept;
    [[nodiscard]] const Value *variable() const noexcept;

    void set_variable(Value *variable) noexcept;
};

}// namespace luisa::compute::xir
