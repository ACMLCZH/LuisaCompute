#pragma once

#include <luisa/xir/value.h>

namespace luisa::compute::xir {

class Function;

enum struct DerivedArgumentTag {
    VALUE,
    REFERENCE,
    RESOURCE,
};

class LC_XIR_API Argument : public DerivedValue<DerivedValueTag::ARGUMENT> {

private:
    Function *_parent_function = nullptr;

private:
    friend class Function;
    void _set_parent_function(Function *func) noexcept;

public:
    explicit Argument(const Type *type = nullptr, Function *parent_function = nullptr) noexcept;
    [[nodiscard]] virtual DerivedArgumentTag derived_argument_tag() const noexcept = 0;

    [[nodiscard]] bool is_lvalue() const noexcept final {
        return derived_argument_tag() == DerivedArgumentTag::VALUE ||
               derived_argument_tag() == DerivedArgumentTag::REFERENCE;
    }

    [[nodiscard]] auto is_value() const noexcept { return derived_argument_tag() == DerivedArgumentTag::VALUE; }
    [[nodiscard]] auto is_reference() const noexcept { return derived_argument_tag() == DerivedArgumentTag::REFERENCE; }
    [[nodiscard]] auto is_resource() const noexcept { return derived_argument_tag() == DerivedArgumentTag::RESOURCE; }

    [[nodiscard]] Function *parent_function() noexcept { return _parent_function; }
    [[nodiscard]] const Function *parent_function() const noexcept { return _parent_function; }
};

template<DerivedArgumentTag tag>
class DerivedArgument : public Argument {
public:
    using Argument::Argument;

    [[nodiscard]] static constexpr auto
    static_derived_argument_tag() noexcept { return tag; }

    [[nodiscard]] DerivedArgumentTag
    derived_argument_tag() const noexcept final { return static_derived_argument_tag(); }
};

class ValueArgument final : public DerivedArgument<DerivedArgumentTag::VALUE> {
public:
    using DerivedArgument::DerivedArgument;
};

class ReferenceArgument final : public DerivedArgument<DerivedArgumentTag::REFERENCE> {
public:
    using DerivedArgument::DerivedArgument;
};

class ResourceArgument final : public DerivedArgument<DerivedArgumentTag::RESOURCE> {
public:
    using DerivedArgument::DerivedArgument;
};

using ArgumentList = InlineIntrusiveList<Argument>;

}// namespace luisa::compute::xir
