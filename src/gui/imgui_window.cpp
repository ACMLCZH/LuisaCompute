#include <mutex>
#include <random>

#if defined(LUISA_PLATFORM_WINDOWS)
#define GLFW_EXPOSE_NATIVE_WIN32
#elif defined(LUISA_PLATFORM_APPLE)
#define GLFW_EXPOSE_NATIVE_COCOA
#else
#define GLFW_EXPOSE_NATIVE_X11// TODO: other window compositors
#endif

#ifndef GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_NONE
#endif

#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>

#include <luisa/core/logging.h>
#include <luisa/core/stl/queue.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/bindless_array.h>
#include <luisa/runtime/swapchain.h>
#include <luisa/runtime/rtx/accel.h>
#include <luisa/dsl/sugar.h>
#include <luisa/gui/imgui_window.h>

namespace luisa::compute::detail {

[[nodiscard]] inline auto glfw_window_native_handle(GLFWwindow *window) noexcept {
#if defined(LUISA_PLATFORM_WINDOWS)
    return reinterpret_cast<uint64_t>(glfwGetWin32Window(window));
#elif defined(LUISA_PLATFORM_APPLE)
    return reinterpret_cast<uint64_t>(glfwGetCocoaWindow(window));
#else
    return reinterpret_cast<uint64_t>(glfwGetX11Window(window));
#endif
}

struct alignas(16u) GUIVertex {
    float px;
    float py;
    float pz;
    uint clip_idx;
    float2 uv;
    uint packed_color;
    uint tex_id;
};

}// namespace luisa::compute::detail

LUISA_STRUCT(luisa::compute::detail::GUIVertex, px, py, pz, clip_idx, uv, packed_color, tex_id) {

    [[nodiscard]] auto p() const noexcept {
        return make_float3(px, py, pz);
    }

    [[nodiscard]] auto color() const noexcept {
        auto r = (packed_color & 0xffu) / 255.f;
        auto g = ((packed_color >> 8u) & 0xffu) / 255.f;
        auto b = ((packed_color >> 16u) & 0xffu) / 255.f;
        auto a = ((packed_color >> 24u) & 0xffu) / 255.f;
        return make_float4(r, g, b, a);
    }
};

namespace luisa::compute {

class ImGuiWindow::Impl {
private:
    class CtxGuard {

    private:
        ImGuiContext *_curr_ctx;
        ImGuiContext *_old_ctx;

    public:
        explicit CtxGuard(ImGuiContext *curr) noexcept
            : _curr_ctx{curr} {
            _old_ctx = ImGui::GetCurrentContext();
            ImGui::SetCurrentContext(_curr_ctx);
        }

        ~CtxGuard() noexcept {
            auto curr_ctx = ImGui::GetCurrentContext();
            LUISA_ASSERT(curr_ctx == nullptr /* destroyed */ || curr_ctx == _curr_ctx,
                         "ImGui context mismatch.");
            ImGui::SetCurrentContext(_old_ctx);
        }

    public:
        CtxGuard(CtxGuard &&) noexcept = delete;
        CtxGuard(const CtxGuard &) noexcept = delete;
        CtxGuard &operator=(CtxGuard &&) noexcept = delete;
        CtxGuard &operator=(const CtxGuard &) noexcept = delete;
    };

    using Vertex = detail::GUIVertex;

private:
    Device &_device;
    Stream &_stream;
    Config _config;
    ImGuiContext *_context;
    GLFWwindow *_main_window;
    Swapchain _main_swapchain;
    Image<float> _main_framebuffer;
    Image<float> _font_texture;
    BindlessArray _texture_array;
    uint _texture_array_offset{0u};
    luisa::unordered_map<ImGuiID, Swapchain> _platform_swapchains;
    luisa::unordered_map<ImGuiID, Image<float>> _platform_framebuffers;

    // for rendering
    Shader2D<Image<float>, float3> _clear_shader;
    Shader2D<Image<float> /* framebuffer */,
             uint2 /* clip base */,
             Accel /* accel */,
             Buffer<Triangle> /* triangles */,
             Buffer<Vertex> /* vertices */,
             BindlessArray /* textures */,
             Buffer<float4> /* clip rectangles */>
        _render_shader;
    Accel _accel;
    uint64_t _mesh_handle{~0ull};
    Buffer<Vertex> _vertex_buffer;
    Buffer<Triangle> _triangle_buffer;
    Buffer<float4> _clip_buffer;

private:
    template<typename F>
    decltype(auto) _with_context(F &&f) noexcept {
        CtxGuard guard{_context};
        return luisa::invoke(std::forward<F>(f));
    }

private:
    void _on_imgui_create_window(ImGuiViewport *vp) noexcept {
        auto glfw_window = static_cast<GLFWwindow *>(vp->PlatformHandle);
        LUISA_ASSERT(glfw_window != nullptr && glfw_window != _main_window,
                     "Invalid GLFW window.");
        auto frame_width = 0, frame_height = 0;
        glfwGetFramebufferSize(glfw_window, &frame_width, &frame_height);
        auto native_handle = detail::glfw_window_native_handle(glfw_window);
        auto sc = _device.create_swapchain(native_handle, _stream,
                                           make_uint2(frame_width, frame_height),
                                           _config.hdr, _config.vsync,
                                           _config.back_buffers);
        auto fb = _device.create_image<float>(sc.backend_storage(), frame_width, frame_height);
        _platform_swapchains[vp->ID] = std::move(sc);
        _platform_framebuffers[vp->ID] = std::move(fb);
    }
    void _on_imgui_destroy_window(ImGuiViewport *vp) noexcept {
        _stream.synchronize();
        if (auto glfw_window = static_cast<GLFWwindow *>(vp->PlatformHandle);
            glfw_window != _main_window) {
            _platform_swapchains.erase(vp->ID);
        }
    }
    void _on_imgui_set_window_size(ImGuiViewport *vp, ImVec2) noexcept {
        _stream.synchronize();
        auto frame_width = 0, frame_height = 0;
        auto glfw_window = static_cast<GLFWwindow *>(vp->PlatformHandle);
        LUISA_ASSERT(glfw_window != nullptr, "Invalid GLFW window.");
        glfwGetFramebufferSize(glfw_window, &frame_width, &frame_height);
        auto native_handle = detail::glfw_window_native_handle(glfw_window);
        auto &sc = glfw_window == _main_window ? _main_swapchain : _platform_swapchains.at(vp->ID);
        auto &fb = glfw_window == _main_window ? _main_framebuffer : _platform_framebuffers.at(vp->ID);
        sc = {};
        fb = {};
        sc = _device.create_swapchain(native_handle, _stream,
                                      make_uint2(frame_width, frame_height),
                                      _config.hdr, _config.vsync,
                                      _config.back_buffers);
        fb = _device.create_image<float>(sc.backend_storage(), frame_width, frame_height);
    }
    void _on_imgui_render_window(ImGuiViewport *vp, void *) noexcept {
        auto &sc = _platform_swapchains.at(vp->ID);
        auto &fb = _platform_framebuffers.at(vp->ID);
        _draw(sc, fb, vp->DrawData);
    }

public:
    Impl(Device &device, Stream &stream, luisa::string name, const Config &config) noexcept
        : _device{device},
          _stream{stream},
          _config{config},
          _context{[] {
              IMGUI_CHECKVERSION();
              return ImGui::CreateContext();
          }()},
          _main_window{nullptr} {

        // initialize GLFW
        static std::once_flag once_flag;
        std::call_once(once_flag, [] {
            glfwSetErrorCallback([](int error, const char *description) noexcept {
                if (error != GLFW_NO_ERROR) [[likely]] {
                    LUISA_ERROR("GLFW Error (code = 0x{:08x}): {}.", error, description);
                }
            });
            if (!glfwInit()) [[unlikely]] {
                LUISA_ERROR_WITH_LOCATION("Failed to initialize GLFW.");
            }
        });

        // create main window
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, config.resizable);
        _main_window = glfwCreateWindow(static_cast<int>(config.size.x),
                                        static_cast<int>(config.size.y),
                                        name.c_str(),
                                        nullptr, nullptr);
        LUISA_ASSERT(_main_window != nullptr, "Failed to create GLFW window.");
        glfwSetWindowUserPointer(_main_window, this);

        // create main swapchain
        auto frame_width = 0, frame_height = 0;
        glfwGetFramebufferSize(_main_window, &frame_width, &frame_height);
        auto native_handle = detail::glfw_window_native_handle(_main_window);
        _main_swapchain = _device.create_swapchain(
            native_handle, stream,
            make_uint2(frame_width, frame_height),
            config.hdr, config.vsync, config.back_buffers);
        _main_framebuffer = _device.create_image<float>(
            _main_swapchain.backend_storage(), frame_width, frame_height);

        // TODO: install callbacks

        // imgui config
        _with_context([this] {
            auto &io = ImGui::GetIO();
            io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;// Enable Keyboard Controls
            io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad; // Enable Gamepad Controls
            io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;    // Enable Docking
            io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;  // Enable Multi-Viewport / Platform Windows

            // styles
            ImGui::StyleColorsDark();
            if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) [[likely]] {
                auto &style = ImGui::GetStyle();
                style.WindowRounding = 0.0f;
                style.Colors[ImGuiCol_WindowBg].w = 1.0f;
            }

            // register glfw window
            ImGui_ImplGlfw_InitForOther(_main_window, true);

            // register renderer (this)
            io.BackendRendererUserData = this;
            io.BackendRendererName = "imgui_impl_luisa";
            io.BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset;
            io.BackendFlags |= ImGuiBackendFlags_RendererHasViewports;
            if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) [[likely]] {
                auto &platform_io = ImGui::GetPlatformIO();
                static constexpr auto imgui_get_this = [] {
                    return ImGui::GetCurrentContext() ?
                               static_cast<Impl *>(ImGui::GetIO().BackendRendererUserData) :
                               nullptr;
                };
                platform_io.Renderer_CreateWindow = [](ImGuiViewport *vp) noexcept {
                    if (auto self = imgui_get_this()) {
                        self->_on_imgui_create_window(vp);
                    }
                };
                platform_io.Renderer_DestroyWindow = [](ImGuiViewport *vp) noexcept {
                    if (auto self = imgui_get_this()) {
                        self->_on_imgui_destroy_window(vp);
                    }
                };
                platform_io.Renderer_SetWindowSize = [](ImGuiViewport *vp, ImVec2 size) noexcept {
                    if (auto self = imgui_get_this()) {
                        self->_on_imgui_set_window_size(vp, size);
                    }
                };
                platform_io.Renderer_RenderWindow = [](ImGuiViewport *vp, void *user_data) noexcept {
                    if (auto self = imgui_get_this()) {
                        self->_on_imgui_render_window(vp, user_data);
                    }
                };
            }
        });

        // create texture array
        _texture_array = _device.create_bindless_array();

        // create shaders
        _clear_shader = _device.compile<2>([](ImageFloat fb, Float3 color) noexcept {
            auto tid = dispatch_id().xy();
            fb.write(tid, make_float4(color, 1.f));
        });
        _render_shader = _device.compile<2>([ssaa = config.ssaa](ImageFloat fb, UInt2 offset, AccelVar accel,
                                                                 BufferVar<Triangle> triangles, BufferVar<Vertex> vertices,
                                                                 BindlessVar texture_array, BufferFloat4 clip_rects) noexcept {
            auto tid = offset + dispatch_id().xy();
            $if (all(tid < dispatch_size().xy())) {
                auto offsets = ssaa ?
                                   luisa::vector{
                                       make_float2(1.f / 3.f, 1.f / 3.f),
                                       make_float2(2.f / 3.f, 1.f / 3.f),
                                       make_float2(1.f / 3.f, 2.f / 3.f),
                                       make_float2(2.f / 3.f, 2.f / 3.f),
                                   } :
                                   luisa::vector{make_float2(.5f)};
                auto k = static_cast<float>(1. / static_cast<double>(offsets.size()));
                auto sum = def(make_float3(0.f));
                auto old = fb.read(tid).xyz();
                for (auto offset : offsets) {
                    auto o = make_float3(make_float2(tid) + offset, -1.f);
                    auto d = make_float3(0.f, 0.f, 1.f);
                    auto ray = make_ray(o, d);
                    auto beta = def(1.f);
                    auto depth = def(0u);
                    $while (beta > 1e-3f & depth < 64u) {
                        depth += 1u;
                        auto hit = accel.intersect(ray, {});
                        $if (!hit->is_triangle()) { $break; };
                        auto triangle = triangles->read(hit.prim);
                        auto v0 = vertices->read(triangle.i0);
                        auto v1 = vertices->read(triangle.i1);
                        auto v2 = vertices->read(triangle.i2);
                        auto p = hit->triangle_interpolate(v0->p(), v1->p(), v2->p());
                        auto clip = clip_rects->read(v0.clip_idx);
                        $if (all(p.xy() >= clip.xy() && p.xy() <= clip.zw())) {
                            auto uv = hit->triangle_interpolate(v0.uv, v1.uv, v2.uv);
                            auto c = hit->triangle_interpolate(v0->color(), v1->color(), v2->color());
                            auto tex_id = v0.tex_id;
                            $if (tex_id != 0u) {
                                c *= texture_array->tex2d(v0.tex_id).sample(uv);
                            };
                            sum += k * c.xyz() * beta * c.w;
                            beta *= 1.f - c.w;
                        };
                        // step through the layer
                        auto pp = p + make_float3(0.f, 0.f, depth_peeling_step * .5);
                        ray = make_ray(pp, d);
                    };
                    // accumulate the background
                    sum += k * beta * old;
                }
                fb.write(tid, make_float4(sum, 1.f));
            };
        });
    }

    ~Impl() noexcept {
        _stream.synchronize();
        _with_context([] {
            ImGui_ImplGlfw_Shutdown();
            ImGui::DestroyContext();
        });
        LUISA_ASSERT(_platform_swapchains.empty(),
                     "Some ImGui windows are not destroyed.");
        _main_swapchain = {};
        glfwDestroyWindow(_main_window);
        if (_accel) {
            _accel = {};
            _device.impl()->destroy_mesh(_mesh_handle);
        }
    }

public:
    [[nodiscard]] auto handle() const noexcept { return _main_window; }
    [[nodiscard]] auto context() const noexcept { return _context; }
    [[nodiscard]] auto &swapchain() const noexcept { return const_cast<Swapchain &>(_main_swapchain); }
    [[nodiscard]] auto &framebuffer() const noexcept { return const_cast<Image<float> &>(_main_framebuffer); }
    [[nodiscard]] auto should_close() const noexcept {
        return static_cast<bool>(glfwWindowShouldClose(_main_window));
    }
    [[nodiscard]] auto set_should_close(bool b) noexcept {
        glfwSetWindowShouldClose(_main_window, b);
    }
    [[nodiscard]] auto register_texture(const Image<float> &image, Sampler sampler) noexcept {
        auto tex_id = static_cast<uint64_t>(++_texture_array_offset);
        _texture_array.emplace_on_update(tex_id, image, sampler);
        _stream << _texture_array.update();
        return tex_id;
    }

private:
    void _create_font_texture() noexcept {
        auto &io = ImGui::GetIO();
        auto pixels = static_cast<unsigned char *>(nullptr);
        auto width = 0, height = 0;
        io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
        // TODO: mipmaps?
        _font_texture = _device.create_image<float>(PixelStorage::BYTE4, width, height, 1);
        _stream << _font_texture.copy_from(pixels);
        auto tex_id = register_texture(_font_texture, Sampler::linear_point_edge());
        io.Fonts->SetTexID(reinterpret_cast<ImTextureID>(tex_id));
    }

private:
    luisa::vector<Vertex> _vertices;
    luisa::vector<Triangle> _triangles;
    luisa::vector<float4> _clip_rects;

    static constexpr auto depth_peeling_step = 1. / 64.;

    void _build_accel() noexcept {

        // create resources if not created
        AccelOption o{
            .hint = AccelOption::UsageHint::FAST_BUILD,
            .allow_compaction = false,
            .allow_update = false};
        if (!_accel) {
            _accel = _device.create_accel(o);
            _mesh_handle = _device.impl()->create_mesh(o).handle;
            _accel.emplace_back(_mesh_handle);
        }
        if (!_vertex_buffer) { _vertex_buffer = _device.create_buffer<Vertex>(std::max(next_pow2(_vertices.size()), 64_k)); }
        if (!_triangle_buffer) { _triangle_buffer = _device.create_buffer<Triangle>(std::max(next_pow2(_triangles.size()), 64_k)); }
        if (!_clip_buffer) { _clip_buffer = _device.create_buffer<float4>(std::max(next_pow2(_clip_rects.size()), static_cast<size_t>(64u))); }

        // resize buffers if insufficient
        if (_vertex_buffer.size() < _vertices.size() ||
            _triangle_buffer.size() < _triangles.size() ||
            _clip_buffer.size() < _clip_rects.size()) {
            _stream.synchronize();
            if (_vertex_buffer.size() < _vertices.size()) {
                _vertex_buffer = {};
                _vertex_buffer = _device.create_buffer<Vertex>(std::max(next_pow2(_vertices.size()), 64_k));
            }
            if (_triangle_buffer.size() < _triangles.size()) {
                _triangle_buffer = {};
                _triangle_buffer = _device.create_buffer<Triangle>(std::max(next_pow2(_triangles.size()), 64_k));
            }
            if (_clip_buffer.size() < _clip_rects.size()) {
                _clip_buffer = {};
                _clip_buffer = _device.create_buffer<float4>(std::max(next_pow2(_clip_rects.size()), static_cast<size_t>(64u)));
            }
        }
        // update the buffers and build the accel
        _stream << _vertex_buffer.view(0u, _vertices.size()).copy_from(_vertices.data())
                << _triangle_buffer.view(0u, _triangles.size()).copy_from(_triangles.data())
                << _clip_buffer.view(0u, _clip_rects.size()).copy_from(_clip_rects.data())
                << luisa::make_unique<MeshBuildCommand>(
                       _mesh_handle, AccelBuildRequest::FORCE_BUILD,
                       _vertex_buffer.handle(), 0u,
                       _vertices.size() * sizeof(Vertex), sizeof(Vertex),
                       _triangle_buffer.handle(), 0u,
                       _triangles.size() * sizeof(Triangle))
                << _accel.build(AccelBuildRequest::FORCE_BUILD);
    }

    void _draw(Swapchain &sc, Image<float> &fb, ImDrawData *draw_data) noexcept {
        auto vp = draw_data->OwnerViewport;
        // clear framebuffer if needed
        // if (!(vp->Flags & ImGuiViewportFlags_NoRendererClear)) {
        //     _stream << _clear_shader(fb, make_float3(0.f)).dispatch(fb.size());
        // }
        // render imgui draw data to framebuffer
        auto clip_offset = make_float2(draw_data->DisplayPos.x, draw_data->DisplayPos.y);
        auto clip_scale = make_float2(draw_data->FramebufferScale.x, draw_data->FramebufferScale.y);
        auto clip_size = make_float2(draw_data->DisplaySize.x, draw_data->DisplaySize.y) * clip_scale;
        auto transform = [clip_offset, clip_scale](ImVec2 p) noexcept {
            return (make_float2(p.x, p.y) - clip_offset) * clip_scale;
        };
        if (all(clip_size > 0.f)) {
            _vertices.clear();
            _triangles.clear();
            _clip_rects.clear();
            _vertices.reserve(64_k);
            _triangles.reserve(64_k);
            _clip_rects.reserve(64u);
            auto accum_clip_min = make_float2(std::numeric_limits<float>::max());
            auto accum_clip_max = make_float2(-std::numeric_limits<float>::max());
            for (auto i = 0u; i < draw_data->CmdListsCount; i++) {
                auto cmd_list = draw_data->CmdLists[i];
                for (auto j = 0u; j < cmd_list->CmdBuffer.Size; j++) {
                    auto cmd = &cmd_list->CmdBuffer[j];
                    // user callback
                    if (auto callback = cmd->UserCallback) {
                        // we ignore ImDrawCallback_ResetRenderState
                        // since we don't have any state to reset
                        if (callback != ImDrawCallback_ResetRenderState) {
                            callback(cmd_list, cmd);
                        }
                        continue;
                    }
                    // render command
                    auto clip_min = max((make_float2(cmd->ClipRect.x, cmd->ClipRect.y) - clip_offset) * clip_scale, 0.f);
                    auto clip_max = min((make_float2(cmd->ClipRect.z, cmd->ClipRect.w) - clip_offset) * clip_scale, clip_size);
                    if (any(clip_max <= clip_min) || cmd->ElemCount == 0) { continue; }
                    // process the command
                    auto clip_idx = static_cast<uint>(_clip_rects.size());
                    _clip_rects.emplace_back(make_float4(clip_min, clip_max));
                    auto tex_id = static_cast<uint>(reinterpret_cast<uint64_t>(cmd->TextureId));
                    accum_clip_min = min(accum_clip_min, clip_min);
                    accum_clip_max = max(accum_clip_max, clip_max);
                    // triangles
                    for (auto t = 0u; t < cmd->ElemCount; t += 3u) {
                        auto o = static_cast<uint>(_triangles.size());
                        auto make_vertex = [&](ImDrawVert v) noexcept {
                            auto p = transform(v.pos);
                            // from back to front
                            auto z = (draw_data->TotalIdxCount / 3u - 1u - o) * depth_peeling_step;
                            return Vertex{.px = p.x,
                                          .py = p.y,
                                          .pz = static_cast<float>(z),
                                          .clip_idx = clip_idx,
                                          .uv = make_float2(v.uv.x, v.uv.y),
                                          .packed_color = v.col,
                                          .tex_id = tex_id};
                        };
                        auto i0 = cmd_list->IdxBuffer[cmd->IdxOffset + t + 0u] + cmd->VtxOffset;
                        auto i1 = cmd_list->IdxBuffer[cmd->IdxOffset + t + 1u] + cmd->VtxOffset;
                        auto i2 = cmd_list->IdxBuffer[cmd->IdxOffset + t + 2u] + cmd->VtxOffset;
                        auto v0 = cmd_list->VtxBuffer[i0];
                        auto v1 = cmd_list->VtxBuffer[i1];
                        auto v2 = cmd_list->VtxBuffer[i2];
                        _triangles.emplace_back(Triangle{o * 3u + 0u, o * 3u + 1u, o * 3u + 2u});
                        _vertices.emplace_back(make_vertex(v0));
                        _vertices.emplace_back(make_vertex(v1));
                        _vertices.emplace_back(make_vertex(v2));
                    }
                }
            }
            if (!_triangles.empty() && all(accum_clip_max > accum_clip_min)) {
                _build_accel();
                auto clip_min_floor = make_uint2(floor(accum_clip_min));
                auto clip_max_ceil = make_uint2(ceil(accum_clip_max));
                _stream << _render_shader(fb, clip_min_floor, _accel,
                                          _triangle_buffer, _vertex_buffer,
                                          _texture_array, _clip_buffer)
                               .dispatch(clip_max_ceil - clip_min_floor);
            }
        }
        _stream << sc.present(fb);
    }

    void _render() noexcept {
        auto &io = ImGui::GetIO();
        if (auto draw_data = ImGui::GetDrawData()) {
            _draw(_main_swapchain, _main_framebuffer, draw_data);
        }
        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
        }
    }

private:
    bool _inside_frame{false};
    ImGuiContext *_old_ctx{nullptr};

public:
    void prepare_frame() noexcept {
        LUISA_ASSERT(!_inside_frame,
                     "Already inside an ImGui frame. "
                     "Did you forget to call ImGuiWindow::render_frame()?");
        _inside_frame = true;
        _old_ctx = ImGui::GetCurrentContext();
        glfwPollEvents();
        ImGui::SetCurrentContext(_context);
        // ImGui checks if the font texture is created in
        // ImGui::NewFrame() so we have to create it here
        if (!_font_texture) { _create_font_texture(); }
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
    }
    void render_frame() noexcept {
        LUISA_ASSERT(_inside_frame,
                     "Not inside an ImGui frame. "
                     "Did you forget to call ImGuiWindow::prepare_frame()?");
        LUISA_ASSERT(ImGui::GetCurrentContext() == _context,
                     "Invalid ImGui context.");
        ImGui::Render();
        _render();
        ImGui::SetCurrentContext(_old_ctx);
        _old_ctx = nullptr;
        _inside_frame = false;
    }
};

ImGuiWindow::ImGuiWindow(Device &device, Stream &stream, luisa::string name, const Config &config) noexcept
    : _impl{luisa::make_unique<Impl>(device, stream, std::move(name), config)} {}

ImGuiWindow::~ImGuiWindow() noexcept = default;

GLFWwindow *ImGuiWindow::handle() const noexcept { return _impl->handle(); }
Swapchain &ImGuiWindow::swapchain() const noexcept { return _impl->swapchain(); }
Image<float> &ImGuiWindow::framebuffer() const noexcept { return _impl->framebuffer(); }
ImGuiContext *ImGuiWindow::context() const noexcept { return _impl->context(); }
bool ImGuiWindow::should_close() const noexcept { return _impl->should_close(); }
void ImGuiWindow::set_should_close(bool b) noexcept { _impl->set_should_close(b); }
void ImGuiWindow::prepare_frame() noexcept { _impl->prepare_frame(); }
void ImGuiWindow::render_frame() noexcept { _impl->render_frame(); }

uint64_t ImGuiWindow::register_texture(const Image<float> &image, const Sampler &sampler) noexcept {
    return _impl->register_texture(image, sampler);
}

}// namespace luisa::compute
