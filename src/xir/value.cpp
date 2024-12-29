#include <luisa/core/logging.h>
#include <luisa/xir/use.h>
#include <luisa/xir/value.h>

namespace luisa::compute::xir {

Value::Value(const Type *type) noexcept : _type{type} {}

void Value::replace_all_uses_with(Value *value) noexcept {
    while (!_use_list.empty()) {
        auto use = &_use_list.front();
        LUISA_DEBUG_ASSERT(use->value() == this, "Invalid use.");
        use->remove_self();
        use->set_value(value);
        if (value) { use->add_to_list(value->use_list()); }
    }
}

void Value::set_type(const Type *type) noexcept { _type = type; }

}// namespace luisa::compute::xir
