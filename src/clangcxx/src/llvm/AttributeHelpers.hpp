#pragma once
#include "clang/AST/Attr.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Expr.h"

namespace luisa::clangcxx {
inline static bool isLuisaAttribute(clang::AnnotateAttr* Anno)
{
    return Anno->getAnnotation() == "luisa-shader";
}

inline static bool isIgnore(clang::AnnotateAttr* Anno)
{
    if (!isLuisaAttribute(Anno))
        return false;
    if (Anno->args_size() == 1)
    {
        auto arg = Anno->args_begin();
        if (auto TypeLiterial = llvm::dyn_cast<clang::StringLiteral>((*arg)->IgnoreParenCasts()))
        {
            return (TypeLiterial->getString() == "ignore");
        }
    }
    return false;
}
}