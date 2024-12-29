#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

class BasicBlock;

class OutlineInst final : public DerivedBranchInstruction<DerivedInstructionTag::OUTLINE>,
                          public InstructionMergeMixin {
public:
    using DerivedBranchInstruction::DerivedBranchInstruction;
};

}// namespace luisa::compute::xir
