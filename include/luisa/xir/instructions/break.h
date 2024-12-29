#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

// Note: this instruction must be the terminator of a basic block.
class BreakInst final : public DerivedBranchInstruction<DerivedInstructionTag::BREAK> {
public:
    using DerivedBranchInstruction::DerivedBranchInstruction;
};

}// namespace luisa::compute::xir
