#pragma once
#include <luisa/core/dll_export.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/context.h>
#include <luisa/vstl/ranges.h>
#include <filesystem>

namespace luisa::clangcxx {

struct LC_CLANGCXX_API Compiler {
    Compiler(const compute::ShaderOption &option);
    compute::ShaderCreationInfo create_shader(
        compute::Context const &context,
        compute::Device &device,
        vstd::IRange<luisa::string_view>& defines,
        const std::filesystem::path &shader_path,
        const std::filesystem::path &include_path) LUISA_NOEXCEPT;
    static void lsp_compile_commands(
        compute::Context const &context,
        vstd::IRange<luisa::string_view>& defines,
        const std::filesystem::path &shader_dir,
        const std::filesystem::path &shader_relative_dir,
        const std::filesystem::path &include_path,
        luisa::vector<char> &result);
private:
    compute::ShaderOption option;
    static luisa::vector<luisa::string> compile_args(
        compute::Context const &context,
        vstd::IRange<luisa::string_view>& defines,
        const std::filesystem::path &shader_path,
        const std::filesystem::path &include_path,
        bool is_lsp);
};

}// namespace luisa::clangcxx
