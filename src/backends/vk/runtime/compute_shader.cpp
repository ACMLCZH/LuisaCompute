#include "compute_shader.h"
#include "device.h"
#include "../log.h"
namespace lc::vk {
ComputeShader::ComputeShader(
    Device *device,
    vstd::span<hlsl::Property const> binds,
    vstd::span<uint const> spv_code,
    vstd::span<std::byte const> cache_code)
    : Shader{device, ShaderTag::ComputeShader, binds} {
    VkPipelineCacheCreateInfo pso_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    if (cache_code.size() > sizeof(VkPipelineCacheHeaderVersionOne) &&
        device->is_pso_same(*reinterpret_cast<const VkPipelineCacheHeaderVersionOne *>(cache_code.data()))) {
        pso_ci.initialDataSize = cache_code.size();
        pso_ci.pInitialData = cache_code.data();
    }
    VK_CHECK_RESULT(vkCreatePipelineCache(device->logic_device(), &pso_ci, nullptr, &_pipe_cache));
    VkShaderModuleCreateInfo module_create_info{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spv_code.size_bytes(),
        .pCode = spv_code.data()};
    VkShaderModule shader_module;
    VK_CHECK_RESULT(vkCreateShaderModule(device->logic_device(), &module_create_info, nullptr, &shader_module));
    auto dispose_module = vstd::scope_exit([&] {
        vkDestroyShaderModule(device->logic_device(), shader_module, nullptr);
    });
    VkComputePipelineCreateInfo pipe_ci{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .flags = 0,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .flags = 0,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader_module,
            .pName = "main"},
        .layout = _pipeline_layout};

    VK_CHECK_RESULT(vkCreateComputePipelines(device->logic_device(), _pipe_cache, 1, &pipe_ci, nullptr, &_pipeline));
}
void ComputeShader::serialize_pso(vstd::vector<std::byte> &result) {
    auto last_size = result.size();
    size_t pso_size;
    VK_CHECK_RESULT(vkGetPipelineCacheData(device()->logic_device(), _pipe_cache, &pso_size, nullptr));
    result.resize_uninitialized(last_size + pso_size);
    VK_CHECK_RESULT(vkGetPipelineCacheData(device()->logic_device(), _pipe_cache, &pso_size, result.data() + last_size));
    result.resize_uninitialized(last_size + pso_size);
}
ComputeShader::~ComputeShader() {
    vkDestroyPipeline(device()->logic_device(), _pipeline, nullptr);
    vkDestroyPipelineCache(device()->logic_device(), _pipe_cache, nullptr);
}
}// namespace lc::vk