#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

enum struct AllocSpace {
    LOCAL,
    SHARED,
};

class LC_XIR_API AllocaInst final : public DerivedInstruction<DerivedInstructionTag::ALLOCA> {

private:
    AllocSpace _space;

public:
    explicit AllocaInst(const Type *type = nullptr,
                        AllocSpace space = AllocSpace::LOCAL) noexcept;
    [[nodiscard]] bool is_lvalue() const noexcept override { return true; }
    void set_space(AllocSpace space) noexcept;
    [[nodiscard]] auto space() const noexcept { return _space; }
};

}// namespace luisa::compute::xir
