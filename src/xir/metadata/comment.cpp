#include <luisa/xir/metadata/comment.h>

namespace luisa::compute::xir {

CommentMD::CommentMD(luisa::string comment) noexcept
    : _comment{std::move(comment)} {}

void CommentMD::set_comment(luisa::string_view comment) noexcept {
    _comment = comment;
}

}// namespace luisa::compute::xir
