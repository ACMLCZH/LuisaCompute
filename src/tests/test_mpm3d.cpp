//
// Created by Mike Smith on 2022/5/11.
//

#include <random>
#include <fstream>
#include <chrono>
#include <dsl/syntax.h>
#include <dsl/sugar.h>
#include <runtime/buffer.h>
#include <runtime/device.h>
#include <runtime/stream.h>
#include <core/logging.h>
#include <gui/window.h>
#include <gui/framerate.h>

int main(int argc, char *argv[]) {

    using namespace luisa;
    using namespace luisa::compute;

    auto sqr = [](auto x) noexcept { return x * x; };

    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, ispc, metal", argv[0]);
        exit(1);
    }
    auto device = context.create_device(argv[1]);

    static constexpr auto n_grid = 64;
    static constexpr auto n_steps = 25u;

    static constexpr auto n_particles = n_grid * n_grid * n_grid / 4u;
    static constexpr auto dx = 1.f / n_grid;
    static constexpr auto dt = 8e-5f;
    static constexpr auto p_rho = 1.f;
    static constexpr auto p_vol = (dx * .5f) * (dx * .5f) * (dx * .5f);
    static constexpr auto p_mass = p_rho * p_vol;
    static constexpr auto gravity = 9.8f;
    static constexpr auto bound = 3;
    static constexpr auto E = 400.f;

    static constexpr auto resolution = 1024u;

    auto x = device.create_buffer<float3>(n_particles);
    auto v = device.create_buffer<float3>(n_particles);
    auto C = device.create_buffer<float3x3>(n_particles);
    auto J = device.create_buffer<float>(n_particles);
    auto grid = device.create_buffer<float4>(n_grid * n_grid * n_grid);
    Window window{"MPM3D", resolution, resolution};
    // luisa::vector<std::array<uint8_t, 4u>> display_buffer(resolution * resolution);
    // std::fstream file("luisa_cpp_speed.csv", std::ios_base::out);
    // file << "Frame, Time(ms)\n";
    auto stream = device.create_stream(StreamTag::GRAPHICS);
    auto swap_chain{device.create_swapchain(
        window.native_handle(),
        stream,
        make_uint2(resolution),
        false, false, 3)};
    auto display = device.create_image<float>(swap_chain.backend_storage(), make_uint2(resolution));

    auto index = [](auto xyz) noexcept {
        using T = vector_expr_element_t<decltype(xyz)>;
        auto p = clamp(xyz, static_cast<T>(0), static_cast<T>(n_grid - 1));
        return p.x + p.y * n_grid + p.z * n_grid * n_grid;
    };
    auto outer_product = [](auto a, auto b) noexcept {
        return make_float3x3(
            make_float3(a[0] * b[0], a[1] * b[0], a[2] * b[0]),
            make_float3(a[0] * b[1], a[1] * b[1], a[2] * b[1]),
            make_float3(a[0] * b[2], a[1] * b[2], a[2] * b[2]));
    };
    auto trace = [](auto m) noexcept { return m[0][0] + m[1][1] + m[2][2]; };

    auto clear_grid = device.compile<3>([&] {
        set_block_size(8, 8, 1);
        auto idx = index(dispatch_id().xyz());
        grid->write(idx, make_float4());
    });

    auto point_to_grid = device.compile<1>([&] {
        set_block_size(64, 1, 1);
        auto p = dispatch_id().x;
        auto Xp = x->read(p) / dx;
        auto base = make_int3(Xp - 0.5f);
        auto fx = Xp - make_float3(base);
        std::array w{0.5f * sqr(1.5f - fx),
                     0.75f - sqr(fx - 1.0f),
                     0.5f * sqr(fx - 0.5f)};
        auto stress = -4.f * dt * E * p_vol * (J->read(p) - 1.f) / sqr(dx);
        auto affine = make_float3x3(
            stress, 0.f, 0.f,
            0.f, stress, 0.f,
            0.f, 0.f, stress) + p_mass * C->read(p);
        auto vp = v->read(p);
        for (auto ii = 0; ii < 27; ii++) {
            auto offset = make_int3(ii % 3, ii / 3 % 3, ii / 3 / 3);
            auto i = offset.x;
            auto j = offset.y;
            auto k = offset.z;
            auto dpos = (make_float3(offset) - fx) * dx;
            auto weight = w[i].x * w[j].y * w[k].z;
            auto vadd = weight * (p_mass * vp + affine * dpos);
            auto idx = index(base + offset);
            grid->atomic(idx).x.fetch_add(vadd.x);
            grid->atomic(idx).y.fetch_add(vadd.y);
            grid->atomic(idx).z.fetch_add(vadd.z);
            grid->atomic(idx).w.fetch_add(weight * p_mass);
        }
    });

    auto simulate_grid = device.compile<3>([&] {
        set_block_size(8, 8, 1);
        auto coord = make_int3(dispatch_id().xyz());
        auto i = index(coord);
        auto v_and_m = grid->read(i);
        auto v = v_and_m.xyz();
        auto m = v_and_m.w;
        v = ite(m > 0.f, v / m, v);
        v.y -= dt * gravity;
        v = ite((coord < bound && v < 0.f) || (coord > n_grid - bound && v > 0.f), 0.f, v);
        grid->write(i, make_float4(v, m));
    });

    auto grid_to_point = device.compile<1>([&] {
        set_block_size(64, 1, 1);
        auto p = dispatch_id().x;
        auto Xp = x->read(p) / dx;
        auto base = make_int3(Xp - 0.5f);
        auto fx = Xp - make_float3(base);
        std::array w{0.5f * sqr(1.5f - fx),
                     0.75f - sqr(fx - 1.0f),
                     0.5f * sqr(fx - 0.5f)};
        auto new_v = def(make_float3(0.f));
        auto new_C = def(make_float3x3(0.f));
        for (auto ii = 0; ii < 27; ii++) {
            auto offset = make_int3(ii % 3, ii / 3 % 3, ii / 3 / 3);
            auto i = offset.x;
            auto j = offset.y;
            auto k = offset.z;
            auto dpos = (make_float3(offset) - fx) * dx;
            auto weight = w[i].x * w[j].y * w[k].z;
            auto idx = index(base + offset);
            auto g_v = grid->read(idx).xyz();
            new_v += weight * g_v;
            new_C = new_C + 4.f * weight * outer_product(g_v, dpos) / sqr(dx);
        }
        v->write(p, new_v);
        x->write(p, x->read(p) + new_v * dt);
        J->write(p, J->read(p) * (1.f + dt * trace(new_C)));
        C->write(p, new_C);
    });
    auto substep = [&](CommandList &cmd_list) noexcept {
        cmd_list << clear_grid().dispatch(n_grid, n_grid, n_grid)
                 << point_to_grid().dispatch(n_particles)
                 << simulate_grid().dispatch(n_grid, n_grid, n_grid)
                 << grid_to_point().dispatch(n_particles);
    };

    auto init = [&](Stream &stream) noexcept {
        luisa::vector<float3> x_init(n_particles);
        std::default_random_engine random{std::random_device{}()};
        std::uniform_real_distribution<float> uniform;
        for (auto i = 0; i < n_particles; i++) {
            auto rx = uniform(random);
            auto ry = uniform(random);
            auto rz = uniform(random);
            x_init[i] = make_float3(rx * .4f + .2f, ry * .4f + .2f, rz * .4f + .2f);
        }
        luisa::vector<float3> v_init(n_particles, make_float3(0.f));
        luisa::vector<float> J_init(n_particles, 1.f);
        luisa::vector<float3x3> C_init(n_particles, make_float3x3(0.f));
        stream << x.copy_from(x_init.data())
               << v.copy_from(v_init.data())
               << J.copy_from(J_init.data())
               << C.copy_from(C_init.data())
               << synchronize();
    };

    auto clear_display = device.compile<2>([&] {
        display->write(dispatch_id().xy(),
                       make_float4(.1f, .2f, .3f, 1.f));
    });

    static constexpr auto phi = radians(28);
    static constexpr auto theta = radians(32);

    auto T = [&](Float3 a0) noexcept {
        auto a = a0 - 0.5f;
        auto c = cos(phi);
        auto s = sin(phi);
        auto C = cos(theta);
        auto S = sin(theta);
        a.x = a.x * c + a.z * s;
        a.z = a.z * c - a.x * s;
        return make_float2(a.x, a.y * C + a.z * S) + 0.5f;
    };

    auto draw_particles = device.compile<1>([&] {
        auto p = dispatch_id().x;
        auto basepos = T(x->read(p));
        for (auto i = -1; i <= 1; i++) {
            for (auto j = -1; j <= 1; j++) {
                auto pos = make_int2(basepos * static_cast<float>(resolution)) + make_int2(i, j);
                $if(pos.x >= 0 & pos.x < resolution & pos.y >= 0 & pos.y < resolution) {
                    display->write(make_uint2(cast<uint>(pos.x), resolution - 1u - pos.y),
                                   make_float4(.4f, .6f, .6f, 1.f));
                };
            }
        }
    });

    init(stream);
    Framerate fps;
    while (!window.should_close()) {
        fps.record(1u);
        LUISA_INFO("FPS: {}", fps.report());
        CommandList cmd_list;
        for (auto i = 0u; i < n_steps; i++) { substep(cmd_list); }
        cmd_list << clear_display().dispatch(resolution, resolution)
                 << draw_particles().dispatch(n_particles);
        stream << cmd_list.commit() << swap_chain.present(display);
        window.pool_event();
    }
    stream << synchronize();
}
