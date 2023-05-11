//
// Created by Mike Smith on 2022/3/23.
//

#pragma once

#include <core/stl.h>
#include <dsl/expr.h>
#include <dsl/operators.h>

namespace luisa::compute {

template<typename T>
class Local {

private:
    const RefExpr *_expression;
    size_t _size;

private:
    static auto _local_type(size_t n) noexcept {
        auto elem = Type::of<T>();
        if (elem->is_scalar() && n > 1u && n <= 4u) {
            return Type::from(luisa::format(
                "vector<{},{}>", elem->description(), n));
        }
        return Type::from(luisa::format(
            "array<{},{}>", elem->description(), n));
    }

public:
    explicit Local(size_t n) noexcept
        : _expression{detail::FunctionBuilder::current()->local(_local_type(n))},
          _size{n} {}

    Local(Local &&) noexcept = default;
    Local(const Local &another) noexcept
        : _size{another._size} {
        auto fb = detail::FunctionBuilder::current();
        _expression = fb->local(_local_type(_size));
        fb->assign(_expression, another._expression);
    }
    Local &operator=(const Local &rhs) noexcept {
        if (&rhs != this) [[likely]] {
            LUISA_ASSERT(
                _size == rhs._size,
                "Incompatible sizes ({} and {}).",
                _size, rhs._size);
            detail::FunctionBuilder::current()->assign(
                _expression, rhs._expression);
        }
        return *this;
    }
    Local &operator=(Local &&rhs) noexcept {
        *this = static_cast<const Local &>(rhs);
        return *this;
    }

    template<typename U>
        requires is_array_expr_v<U>
    Local &operator=(U &&rhs) noexcept {
        constexpr auto n = array_expr_dimension_v<U>;
        LUISA_ASSERT(_size == n, "Incompatible sizes ({} and {}).", _size, n);
        detail::FunctionBuilder::current()->assign(
            _expression, rhs._expression);
        return *this;
    }

    [[nodiscard]] auto expression() const noexcept { return _expression; }
    [[nodiscard]] auto size() const noexcept { return _size; }

    template<typename U>
        requires is_integral_expr_v<U>
    [[nodiscard]] Var<T> &operator[](U &&index) const noexcept {
        auto i = def(std::forward<U>(index));
        auto f = detail::FunctionBuilder::current();
        auto expr = f->access(
            Type::of<T>(), _expression, i.expression());
        return *f->create_temporary<Var<T>>(expr);
    }

    template<typename I>
    [[nodiscard]] auto read(I &&index) const noexcept { return (*this)[std::forward<I>(index)]; }

    template<typename I, typename U>
    void write(I &&i, U &&u) const noexcept { (*this)[std::forward<I>(i)] = std::forward<U>(u); }
};

}// namespace luisa::compute

// disable address-of operators
template<typename T>
[[nodiscard]] inline ::luisa::compute::Local<T> *operator&(::luisa::compute::Local<T> &) noexcept {
    static_assert(::luisa::always_false_v<T>,
                  LUISA_DISABLE_DSL_ADDRESS_OF_MESSAGE);
    std::abort();
}

template<typename T>
[[nodiscard]] inline const ::luisa::compute::Local<T> *operator&(const ::luisa::compute::Local<T> &) noexcept {
    static_assert(::luisa::always_false_v<T>,
                  LUISA_DISABLE_DSL_ADDRESS_OF_MESSAGE);
    std::abort();
}
