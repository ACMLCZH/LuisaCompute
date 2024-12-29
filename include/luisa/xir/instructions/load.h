#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

class LC_XIR_API LoadInst final : public DerivedInstruction<DerivedInstructionTag::LOAD> {
public:
    explicit LoadInst(const Type *type = nullptr,
                      Value *variable = nullptr) noexcept;

    [[nodiscard]] Value *variable() noexcept;
    [[nodiscard]] const Value *variable() const noexcept;

    void set_variable(Value *variable) noexcept;
};

}// namespace luisa::compute::xir
