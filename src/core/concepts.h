//
// Created by Mike Smith on 2021/2/3.
//

#pragma once

#include <span>
#include <atomic>
#include <concepts>
#include <type_traits>

#include <core/macro.h>
#include <core/basic_types.h>
#include <core/atomic.h>

namespace luisa::concepts {

struct Noncopyable {
    Noncopyable() = default;
    Noncopyable(const Noncopyable &) = delete;
    Noncopyable &operator=(const Noncopyable &) = delete;
    Noncopyable(Noncopyable &&) noexcept = default;
    Noncopyable &operator=(Noncopyable &&) noexcept = default;
};

template<typename T>
concept iterable = requires(T v) {
    v.begin();
    v.end();
};

template<typename T>
concept span_convertible = requires(T v) {
    std::span{std::forward<T>(v)};
};

template<typename T, typename... Args>
concept constructible = requires(Args... args) {
    T{args...};
};

template<typename T>
concept trivially_default_constructible = std::is_trivially_constructible_v<T>;

template<typename Src, typename Dest>
concept static_convertible = requires(Src s) {
    static_cast<Dest>(s);
};

template<typename Src, typename Dest>
concept bitwise_convertible = sizeof(Src) >= sizeof(Dest);

template<typename Src, typename Dest>
concept reinterpret_convertible = requires(Src s) {
    reinterpret_cast<Dest *>(&s);
};

template<typename F, typename... Args>
concept invocable = std::is_invocable_v<F, Args...>;

template<typename Ret, typename F, typename... Args>
concept invocable_with_return = std::is_invocable_r_v<Ret, F, Args...>;

template<typename T>
concept pointer = std::is_pointer_v<T>;

template<typename T>
concept non_pointer = std::negation_v<std::is_pointer<T>>;

template<typename T>
concept container = requires(T a) {
    a.begin();
    a.size();
};

template<typename T>
concept integral = is_integral_v<T>;

template<typename T>
concept scalar = is_scalar_v<T>;

template<typename T>
concept vector = is_vector_v<T>;

template<typename T>
concept vector2 = is_vector2_v<T>;

template<typename T>
concept vector3 = is_vector3_v<T>;

template<typename T>
concept vector4 = is_vector4_v<T>;

template<typename T>
concept bool_vector = is_bool_vector_v<T>;

template<typename T>
concept float_vector = is_float_vector_v<T>;

template<typename T>
concept int_vector = is_int_vector_v<T>;

template<typename T>
concept uint_vector = is_uint_vector_v<T>;

template<typename T>
concept matrix = is_matrix_v<T>;

template<typename T>
concept matrix2 = is_matrix2_v<T>;

template<typename T>
concept matrix3 = is_matrix3_v<T>;

template<typename T>
concept matrix4 = is_matrix4_v<T>;

template<typename T>
concept basic = is_basic_v<T>;

template<typename T>
concept atomic = is_atomic_v<T>;

// operator traits
#define LUISA_MAKE_UNARY_OP_CONCEPT(op, op_name) \
    template<typename Operand>                   \
    concept op_name = requires(Operand operand) { op operand; };
LUISA_MAKE_UNARY_OP_CONCEPT(+, operator_plus)
LUISA_MAKE_UNARY_OP_CONCEPT(-, operator_minus)
LUISA_MAKE_UNARY_OP_CONCEPT(!, operator_not)
LUISA_MAKE_UNARY_OP_CONCEPT(~, operator_bit_not)
#undef LUISA_MAKE_UNARY_OP_CONCEPT

#define LUISA_MAKE_BINARY_OP_CONCEPT(op, op_name) \
    template<typename Lhs, typename Rhs>          \
    concept op_name = requires(Lhs lhs, Rhs rhs) { lhs op rhs; };
LUISA_MAKE_BINARY_OP_CONCEPT(+, operator_add)
LUISA_MAKE_BINARY_OP_CONCEPT(-, operator_sub)
LUISA_MAKE_BINARY_OP_CONCEPT(*, operator_mul)
LUISA_MAKE_BINARY_OP_CONCEPT(/, operator_div)
LUISA_MAKE_BINARY_OP_CONCEPT(%, operator_mod)
LUISA_MAKE_BINARY_OP_CONCEPT(&, operator_bit_and)
LUISA_MAKE_BINARY_OP_CONCEPT(|, operator_bit_or)
LUISA_MAKE_BINARY_OP_CONCEPT(^, operator_bit_Xor)
LUISA_MAKE_BINARY_OP_CONCEPT(<<, operator_shift_left)
LUISA_MAKE_BINARY_OP_CONCEPT(>>, operator_shift_right)
LUISA_MAKE_BINARY_OP_CONCEPT(&&, operator_and)
LUISA_MAKE_BINARY_OP_CONCEPT(||, operator_or)
LUISA_MAKE_BINARY_OP_CONCEPT(==, operator_equal)
LUISA_MAKE_BINARY_OP_CONCEPT(!=, operator_not_equal)
LUISA_MAKE_BINARY_OP_CONCEPT(<, operator_less)
LUISA_MAKE_BINARY_OP_CONCEPT(<=, operator_less_equal)
LUISA_MAKE_BINARY_OP_CONCEPT(>, operator_greater)
LUISA_MAKE_BINARY_OP_CONCEPT(>=, operator_greater_equal)

LUISA_MAKE_BINARY_OP_CONCEPT(=, assignable)
LUISA_MAKE_BINARY_OP_CONCEPT(+=, add_assignable)
LUISA_MAKE_BINARY_OP_CONCEPT(-=, sub_assignable)
LUISA_MAKE_BINARY_OP_CONCEPT(*=, mul_assignable)
LUISA_MAKE_BINARY_OP_CONCEPT(/=, div_assignable)
LUISA_MAKE_BINARY_OP_CONCEPT(%=, mod_assignable)
LUISA_MAKE_BINARY_OP_CONCEPT(&=, bit_and_assignable)
LUISA_MAKE_BINARY_OP_CONCEPT(|=, bit_or_assignable)
LUISA_MAKE_BINARY_OP_CONCEPT(^=, bit_xor_assignable)
LUISA_MAKE_BINARY_OP_CONCEPT(<<=, shift_left_assignable)
LUISA_MAKE_BINARY_OP_CONCEPT(>>=, shift_right_assignable)
#undef LUISA_MAKE_BINARY_OP_CONCEPT

template<typename Lhs, typename Rhs>
concept operator_access = requires(Lhs lhs, Rhs rhs) { lhs[rhs]; };

template<typename T>
concept function = std::is_function_v<T>;

namespace detail {

    template<typename... T>
    struct all_same_impl : std::false_type {};

    template<>
    struct all_same_impl<> : std::true_type {};

    template<typename T>
    struct all_same_impl<T> : std::true_type {};

    template<typename First, typename... Other>
    struct all_same_impl<First, Other...> : std::conjunction<std::is_same<First, Other>...> {};

}// namespace detail

template<typename... T>
using is_same = detail::all_same_impl<T...>;

template<typename... T>
constexpr auto is_same_v = is_same<T...>::value;

template<typename... T>
concept same = is_same_v<T...>;

template<typename... T>
concept vector_same_dimension = is_vector_same_dimension_v<T...>;

template<typename T>
struct array_dimension {
    static constexpr size_t value = 0u;
};

template<typename T, size_t N>
struct array_dimension<T[N]> {
    static constexpr auto value = N;
};

template<typename T, size_t N>
struct array_dimension<std::array<T, N>> {
    static constexpr auto value = N;
};

template<typename T>
constexpr auto array_dimension_v = array_dimension<T>::value;

template<typename T>
struct array_element {
    using type = T;
};

template<typename T, size_t N>
struct array_element<T[N]> {
    using type = T;
};

template<typename T, size_t N>
struct array_element<std::array<T, N>> {
    using type = T;
};

template<typename T>
using array_element_t = typename array_element<T>::type;

template<typename T>
struct is_array : std::false_type {};

template<typename T, size_t N>
struct is_array<T[N]> : std::true_type {};

template<typename T, size_t N>
struct is_array<std::array<T, N>> : std::true_type {};

template<typename T>
constexpr auto is_array_v = is_array<T>::value;

template<typename T>
concept array = is_array_v<T>;

template<typename T>
concept basic_or_array = basic<T> || array<T>;

template<typename T>
concept not_basic_or_array = !basic_or_array<T>;

template<typename T>
struct is_tuple : std::false_type {};

template<typename... T>
struct is_tuple<std::tuple<T...>> : std::true_type {};

template<typename T>
constexpr auto is_tuple_v = is_tuple<T>::value;

namespace detail {

template<typename T>
struct dimension_impl {
    static constexpr size_t value = 1u;
};

template<typename T, size_t N>
struct dimension_impl<T[N]> {
    static constexpr auto value = N;
};

template<typename T, size_t N>
struct dimension_impl<std::array<T, N>> {
    static constexpr auto value = N;
};

template<typename T, size_t N>
struct dimension_impl<Vector<T, N>> {
    static constexpr auto value = N;
};

template<size_t N>
struct dimension_impl<Matrix<N>> {
    static constexpr auto value = N;
};

}// namespace detail

template<typename T>
using dimension = detail::dimension_impl<std::remove_cvref_t<T>>;

template<typename T>
constexpr auto dimension_v = dimension<T>::value;

}// namespace luisa::concepts
