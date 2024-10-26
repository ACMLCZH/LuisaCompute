#include <luisa/xir/function.h>

namespace luisa::compute::xir {

Function::Function(Pool *pool, FunctionTag tag, const Type *type, const Name *name) noexcept
    : Super{pool, type, name}, _body{pool->create<BasicBlock>()},
      _function_tag{tag}, _arguments{pool}, _shared_variables{pool}, _local_variables{pool} {
    _body->_set_parent_value(this);
    _arguments.head_sentinel()->set_parent_function(this);
    _arguments.tail_sentinel()->set_parent_function(this);
    _shared_variables.head_sentinel()->set_parent_function(this);
    _shared_variables.tail_sentinel()->set_parent_function(this);
    _local_variables.head_sentinel()->set_parent_function(this);
    _local_variables.tail_sentinel()->set_parent_function(this);
}

void Function::add_argument(Argument *argument) noexcept {
    _arguments.insert_back(argument);
}

void Function::add_shared_variable(SharedVariable *shared) noexcept {
    _shared_variables.insert_back(shared);
}

void Function::add_local_variable(LocalVariable *local) noexcept {
    _local_variables.insert_back(local);
}

Argument *Function::create_argument(const Type *type, bool by_ref, const Name *name) noexcept {
    auto argument = pool()->create<Argument>(by_ref, this, type, name);
    add_argument(argument);
    return argument;
}

Argument *Function::create_value_argument(const Type *type, const Name *name) noexcept {
    return create_argument(type, false, name);
}

Argument *Function::create_reference_argument(const Type *type, const Name *name) noexcept {
    return create_argument(type, true, name);
}

SharedVariable *Function::create_shared_variable(const Type *type, const Name *name) noexcept {
    auto shared = pool()->create<SharedVariable>(this, type, name);
    add_shared_variable(shared);
    return shared;
}

LocalVariable *Function::create_local_variable(const Type *type, const Name *name) noexcept {
    auto local = pool()->create<LocalVariable>(this, type, name);
    add_local_variable(local);
    return local;
}

}// namespace luisa::compute::xir
