#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

class BasicBlock;

// Branch instruction:
//
// if (cond) {
//   true_block
// } else {
//   false_block
// }
// { merge_block }
//
// Note: this instruction must be the terminator of a basic block.
class IfInst final : public DerivedConditionalBranchInstruction<DerivedInstructionTag::IF>,
                     public InstructionMergeMixin {
public:
    using DerivedConditionalBranchInstruction::DerivedConditionalBranchInstruction;
};

}// namespace luisa::compute::xir
