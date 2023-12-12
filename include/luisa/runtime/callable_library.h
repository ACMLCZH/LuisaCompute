#pragma once

#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/ast/external_function.h>
#include <luisa/ast/function.h>
#include <luisa/ast/function_builder.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/shader.h>

namespace luisa::compute {

template<typename T>
class Callable;

class LC_RUNTIME_API CallableLibrary {

private:
    struct DeserPackage {
        detail::FunctionBuilder *builder;
        luisa::unordered_map<uint64_t, luisa::shared_ptr<detail::FunctionBuilder>> callable_map;
    };
    using CallableMap = luisa::unordered_map<luisa::string, luisa::shared_ptr<const detail::FunctionBuilder>>;
    CallableMap _callables;
    static void serialize_func_builder(detail::FunctionBuilder const &builder, luisa::vector<std::byte> &vec) noexcept;
    static void deserialize_func_builder(detail::FunctionBuilder &builder, std::byte const *&ptr, DeserPackage &pack) noexcept;
    template<typename T>
    static void ser_value(T const &t, luisa::vector<std::byte> &vec) noexcept;
    template<typename T>
    static T deser_value(std::byte const *&ptr, DeserPackage &pack) noexcept;
    template<typename T>
    static void deser_ptr(T obj, std::byte const *&ptr, DeserPackage &pack) noexcept;

public:
    template<typename T>
    Callable<T> get_callable(luisa::string_view name) const noexcept;
    [[nodiscard]] Function get_function(luisa::string_view name) const noexcept;
    template<size_t dim, typename... T>
    Shader<dim, T...> compile_kernel(Device &device, luisa::string_view name, const ShaderOption &option) const noexcept {
        return {device.impl(), get_function(name), option};
    }
    [[nodiscard]] luisa::shared_ptr<const detail::FunctionBuilder> get_function_builder(luisa::string_view name) const noexcept;
    [[nodiscard]] luisa::vector<luisa::string_view> names() const noexcept;
    CallableLibrary() noexcept;
    void add_callable(luisa::string_view name, luisa::shared_ptr<const detail::FunctionBuilder> callable) noexcept;
    void load(luisa::span<const std::byte> binary) noexcept;
    [[nodiscard]] luisa::vector<std::byte> serialize() const noexcept;
    CallableLibrary(CallableLibrary const &) = delete;
    CallableLibrary(CallableLibrary &&) noexcept;
    ~CallableLibrary() noexcept;
};

}// namespace luisa::compute
