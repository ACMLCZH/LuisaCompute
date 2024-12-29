#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

// Note: this instruction must be the terminator of a basic block.
class UnreachableInst final : public DerivedTerminatorInstruction<DerivedInstructionTag::UNREACHABLE> {

private:
    luisa::string _message;

public:
    explicit UnreachableInst(luisa::string message = {}) noexcept;
    void set_message(luisa::string_view message) noexcept;
    [[nodiscard]] luisa::string_view message() const noexcept;
};

}// namespace luisa::compute::xir
