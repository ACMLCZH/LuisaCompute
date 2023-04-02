#include <luisa-compute.h>
#include <dsl/sugar.h>
#include <stb/stb_image_write.h>
#include <iostream>

using namespace luisa;
using namespace luisa::compute;

float lcg(uint &state) noexcept {
    constexpr auto lcg_a = 1664525u;
    constexpr auto lcg_c = 1013904223u;
    state = lcg_a * state + lcg_c;
    return cast<float>(state & 0x00ffffffu) *
           (1.0f / static_cast<float>(0x01000000u));
};

int main(int argc, char *argv[]) {
    constexpr uint32_t width = 1280;
    constexpr uint32_t height = 720;
    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, ispc, metal", argv[0]);
        exit(1);
    }
    auto device = context.create_device(argv[1]);
    auto stream = device.create_stream();
    auto device_image1 = device.create_image<float>(PixelStorage::BYTE4, width, height, 0u);
    int count = 1024;
    float radius = .2f;
    // aabb
    luisa::vector<AABB> aabbs{size_t(count)};
    uint state = 0;
    for (int i = 0; i < count; i++) {
        auto pos = make_float3(lcg(state) * 2.f - 1.f, lcg(state) * 2.f - 1.f, lcg(state) * 2.f - 1.f) * 10.f;
        auto max = pos + radius;
        auto min = pos - radius;
        aabbs[i].packed_max = {max.x, max.y, max.z};
        aabbs[i].packed_min = {min.x, min.y, min.z};
    }
    auto accel = device.create_accel();
    auto aabb_buffer = device.create_buffer<AABB>(count);
    auto procedural_primitives = device.create_procedural_primitive(aabb_buffer.view());
    accel.emplace_back(procedural_primitives);
    // triangle
    std::array vertices{
        float3(-0.5f, -0.5f, 0.0f),
        float3(0.5f, -0.5f, 0.0f),
        float3(0.0f, 0.5f, 0.0f)};
    std::array indices{0u, 1u, 2u};
    auto vertex_buffer = device.create_buffer<float3>(3u);
    auto triangle_buffer = device.create_buffer<Triangle>(1u);
    auto mesh = device.create_mesh(vertex_buffer, triangle_buffer);
    accel.emplace_back(mesh, scaling(5.0f));
    stream << aabb_buffer.copy_from(aabbs.data())
           << procedural_primitives.build()
           << vertex_buffer.copy_from(vertices.data())
           << triangle_buffer.copy_from(indices.data())
           << mesh.build()
           << accel.build()
           << synchronize();

    Kernel2D kernel = [&](Float3 pos) {
        Var coord = dispatch_id().xy();
        Var size = dispatch_size().xy();
        auto aspect = size.x.cast<float>() / size.y.cast<float>();
        auto p = (make_float2(coord)) / make_float2(size) * 2.f - 1.f;
        static constexpr auto fov = radians(45.8f);
        auto origin = pos;
        auto direction = normalize(make_float3(p * tan(0.5f * fov) * make_float2(aspect, 1.0f), -1.0f));
        auto ray = make_ray(origin, direction);
        device_image1->write(coord, make_float4(make_float3(p, 0.5f), 1.0f));

        // traversal aceeleration structure with ray-query
        Var sphere_dist = 1e30f;
        Var<float3> sphere_color;
        auto hit = accel->query_all(ray)
                       .on_triangle_candidate([](auto &candidate) noexcept {
                           candidate.commit();
                       })
                       .on_procedural_candidate([&](auto &candidate) noexcept {
                           auto h = candidate.hit();
                           auto aabb = aabb_buffer->read(h.prim);
                           //ray-sphere intersection
                           auto origin = (aabb->min() + aabb->max()) * .5f;
                           auto ray_origin = ray->origin();
                           auto L = origin - ray_origin;
                           auto dir = ray->direction();
                           auto cos_theta = dot(dir, normalize(L));
                           $if(cos_theta > 0.f) {
                               auto d_oc = length(L);
                               auto tc = d_oc * cos_theta;
                               auto d = sqrt(d_oc * d_oc - tc * tc);
                               $if(d <= radius) {
                                   auto t1c = sqrt(radius * radius - d * d);
                                   auto dist = tc - t1c;
                                   // save normal as color
                                   $if(dist <= sphere_dist) {
                                       sphere_dist = dist;
                                       auto normal = normalize(ray_origin + dir * dist - origin);
                                       sphere_color = normal * 0.5f + 0.5f;
                                   };
                                   candidate.commit(dist);
                               };
                           };
                       })
                       .trace();
        $if(hit->is_procedural()) {
            // write depth as color
            // device_image1->write(coord, make_float4(make_float3(1.f / log(hit->committed_ray_t)), 1.f));
            // write normal
            device_image1->write(coord, make_float4(sphere_color, 1.f));
        }
        $elif(hit->is_triangle()) {
            // write bary-centric
            device_image1->write(coord, make_float4(hit.bary, 0.f, 1.f));
        }
        $else {
            device_image1->write(coord, make_float4(0.f, 0.f, 0.f, 1.f));
        };
    };
    luisa::vector<std::array<uint8_t, 4u>> pixels{width * height};
    auto s = device.compile(kernel);
    const float3 pos = make_float3(0.f, 0.f, 18.0f);
    stream
        << s(pos).dispatch(width, height)
        << device_image1.copy_to(pixels.data())
        << synchronize();
    stbi_write_png("test_procedural.png", width, height, 4, pixels.data(), 0);
}