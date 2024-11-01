#pragma once

#include <luisa/xir/use.h>
#include <luisa/xir/metadata.h>

namespace luisa::compute {
class Type;
}// namespace luisa::compute

namespace luisa::compute::xir {

enum struct DerivedValueTag {
    FUNCTION,
    EXTERNAL_FUNCTION,
    BASIC_BLOCK,
    INSTRUCTION,
    CONSTANT,
    VALUE_ARGUMENT,
    REFERENCE_ARGUMENT,
};

class LC_XIR_API Value : public PooledObject,
                         public MetadataMixin {

private:
    const Type *_type = nullptr;
    UseList _use_list;

public:
    explicit Value(const Type *type = nullptr) noexcept;
    [[nodiscard]] virtual DerivedValueTag derived_value_tag() const noexcept = 0;
    [[nodiscard]] virtual bool is_user() const noexcept { return false; }

    void replace_all_uses_with(Value *value) noexcept;
    virtual void set_type(const Type *type) noexcept;

    [[nodiscard]] auto type() const noexcept { return _type; }
    [[nodiscard]] auto &use_list() noexcept { return _use_list; }
    [[nodiscard]] auto &use_list() const noexcept { return _use_list; }
};

template<DerivedValueTag tag, typename Base = Value>
class DerivedValue : public Base {
public:
    using Base::Base;

    [[nodiscard]] static constexpr DerivedValueTag
    static_derived_value_tag() noexcept { return tag; }

    [[nodiscard]] DerivedValueTag
    derived_value_tag() const noexcept final {
        return static_derived_value_tag();
    }
};

}// namespace luisa::compute::xir
