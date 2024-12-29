#pragma once

#include <luisa/xir/argument.h>
#include <luisa/xir/basic_block.h>

namespace luisa::compute::xir {

enum struct DerivedFunctionTag {
    KERNEL,
    CALLABLE,
    EXTERNAL,
};

class LC_XIR_API Function : public IntrusiveForwardNode<Function, DerivedValue<DerivedValueTag::FUNCTION>> {

private:
    luisa::vector<Argument *> _arguments;

public:
    explicit Function(const Type *type = nullptr) noexcept;
    [[nodiscard]] virtual DerivedFunctionTag derived_function_tag() const noexcept = 0;

    void add_argument(Argument *argument) noexcept;
    void insert_argument(size_t index, Argument *argument) noexcept;
    void remove_argument(Argument *argument) noexcept;
    void remove_argument(size_t index) noexcept;
    void replace_argument(Argument *old_argument, Argument *new_argument) noexcept;
    void replace_argument(size_t index, Argument *argument) noexcept;

    Argument *create_argument(const Type *type, bool by_ref) noexcept;
    ValueArgument *create_value_argument(const Type *type) noexcept;
    ReferenceArgument *create_reference_argument(const Type *type) noexcept;
    ResourceArgument *create_resource_argument(const Type *type) noexcept;

    [[nodiscard]] auto &arguments() noexcept { return _arguments; }
    [[nodiscard]] auto &arguments() const noexcept { return _arguments; }
};

using FunctionList = IntrusiveForwardList<Function>;

template<DerivedFunctionTag tag, typename Base = Function>
class DerivedFunction : public Base {
public:
    using Base::Base;
    [[nodiscard]] static constexpr DerivedFunctionTag static_derived_function_tag() noexcept { return tag; }
    [[nodiscard]] DerivedFunctionTag derived_function_tag() const noexcept final { return static_derived_function_tag(); }
};

class LC_XIR_API FunctionDefinition : public Function {

private:
    BasicBlock *_body_block{nullptr};

public:
    using Function::Function;

    void set_body_block(BasicBlock *block) noexcept;
    BasicBlock *create_body_block(bool overwrite_existing = false) noexcept;

    [[nodiscard]] BasicBlock *body_block() noexcept { return _body_block; }
    [[nodiscard]] const BasicBlock *body_block() const noexcept { return _body_block; }
};

class LC_XIR_API CallableFunction final : public DerivedFunction<DerivedFunctionTag::CALLABLE, FunctionDefinition> {
public:
    using DerivedFunction::DerivedFunction;
};

class LC_XIR_API KernelFunction final : public DerivedFunction<DerivedFunctionTag::KERNEL, FunctionDefinition> {

public:
    static constexpr auto default_block_size = luisa::make_uint3(64u, 1u, 1u);

private:
    luisa::uint3 _block_size{default_block_size};

public:
    explicit KernelFunction(luisa::uint3 block_size = default_block_size) noexcept;
    void set_block_size(luisa::uint3 size) noexcept;
    [[nodiscard]] luisa::uint3 block_size() const noexcept { return _block_size; }
};

class LC_XIR_API ExternalFunction final : public DerivedFunction<DerivedFunctionTag::EXTERNAL> {
public:
    using DerivedFunction::DerivedFunction;
};

}// namespace luisa::compute::xir
