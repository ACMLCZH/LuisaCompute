#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

class Function;
class Instruction;
class User;

class LC_XIR_API BasicBlock : public DerivedValue<DerivedValueTag::BASIC_BLOCK> {

private:
    InstructionList _instructions;

public:
    BasicBlock() noexcept;
    [[nodiscard]] auto &instructions() noexcept { return _instructions; }
    [[nodiscard]] auto &instructions() const noexcept { return _instructions; }
};

}// namespace luisa::compute::xir
