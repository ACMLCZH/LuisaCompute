#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

class LC_XIR_API CommentInst : public Instruction {

private:
    luisa::string _comment;

public:
    explicit CommentInst(Pool *pool,
                         luisa::string comment = {},
                         const Name *name = nullptr) noexcept;

    void set_comment(luisa::string_view comment) noexcept;
    [[nodiscard]] auto comment() const noexcept { return luisa::string_view{_comment}; }
};

}// namespace luisa::compute::xir
