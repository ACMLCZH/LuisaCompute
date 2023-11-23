#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>

struct GLFWwindow;
struct ImGuiContext;

namespace luisa::compute {

class Swapchain;

template<typename T>
class Image;

class LC_GUI_API ImGuiWindow {

public:
    struct Config {
        luisa::string name{"Display"};
        uint2 size{800, 600};
        bool resizable{true};
        bool vsync{true};
        bool hdr{false};
        uint back_buffers{2};
    };

public:
    class Impl;

private:
    luisa::unique_ptr<Impl> _impl;

public:
    ImGuiWindow(Device &device, Stream &stream, const Config &config = {}) noexcept;
    ~ImGuiWindow() noexcept;
    ImGuiWindow(ImGuiWindow &&) noexcept = default;
    ImGuiWindow(const ImGuiWindow &) noexcept = delete;
    ImGuiWindow &operator=(ImGuiWindow &&) noexcept = default;
    ImGuiWindow &operator=(const ImGuiWindow &) noexcept = delete;

public:
    [[nodiscard]] ImGuiContext *context() const noexcept;
    [[nodiscard]] GLFWwindow *handle() const noexcept;
    [[nodiscard]] Swapchain &swapchain() const noexcept;
    [[nodiscard]] Image<float> &framebuffer() const noexcept;
    [[nodiscard]] bool should_close() const noexcept;
    void set_should_close(bool b = true) noexcept;
    void prepare_frame() noexcept;// calls glfwPollEvents, ImGui::NewFrame, and other stuff; also makes the context current
    void render_frame() noexcept; // calls ImGui::Render, glfwSwapBuffers, and other stuff; also restores the current context as before prepare_frame

    template<typename F>
    void wtih_frame(F &&f) noexcept {
        prepare_frame();
        luisa::invoke(std::forward<F>(f));
        render_frame();
    }
};

}// namespace luisa::compute
