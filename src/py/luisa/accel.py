import lcapi
from . import globalvars
from .globalvars import get_global_device
from .structtype import StructType
from .arraytype import ArrayType
from .mathtypes import *
from .func import func
from .types import ref, uint
from .builtin import _builtin_call, _bitwise_cast

# Ray
Ray = StructType(16, _origin=ArrayType(float,3), t_min=float, _dir=ArrayType(float,3), t_max=float)

@func
def make_ray(origin: float3, direction: float3, t_min: float, t_max:float):
    r = Ray()
    r._origin[0] = origin[0]
    r._origin[1] = origin[1]
    r._origin[2] = origin[2]
    r._dir[0] = direction[0]
    r._dir[1] = direction[1]
    r._dir[2] = direction[2]
    r.t_min = t_min
    r.t_max = t_max
    return r

@func
def offset_ray_origin(p: float3, n: float3):
    origin = 1 / 32
    float_scale = 1.0 / 65536.0
    int_scale = 256.0
    of_i = int3(int_scale * n)
    int_p = int3()
    int_p.x = _bitwise_cast(int, p.x)
    int_p.y = _bitwise_cast(int, p.y)
    int_p.z = _bitwise_cast(int, p.z)
    p_i_tmp = int_p + select(of_i, -of_i, p < 0.0)
    p_i = float3()
    p_i.x = _bitwise_cast(float, p_i_tmp.x)
    p_i.y = _bitwise_cast(float, p_i_tmp.y)
    p_i.z = _bitwise_cast(float, p_i_tmp.z)
    return select(p_i, p + float_scale * n, abs(p) < origin)

@func
def get_origin(self):
    return float3(self._origin[0], self._origin[1], self._origin[2])
Ray.add_method('get_origin', get_origin)

@func
def get_dir(self):
    return float3(self._dir[0], self._dir[1], self._dir[2])
Ray.add_method('get_dir', get_dir)

@func
def set_origin(self, val: float3):
    self._origin[0] = val.x
    self._origin[1] = val.y
    self._origin[2] = val.z
Ray.add_method('set_origin', set_origin)

@func
def set_dir(self, val: float3):
    self._dir[0] = val.x
    self._dir[1] = val.y
    self._dir[2] = val.z
Ray.add_method('set_dir', set_dir)



# Hit
Hit = StructType(16, inst=int, prim=int, bary=float2)
UHit = StructType(16, inst=uint, prim=uint, bary=float2)

@func
def miss(self):
    return self.inst == -1
Hit.add_method('miss', miss)

@func
def interpolate(self, a, b, c):
    return (1.0 - self.bary.x - self.bary.y) * a + self.bary.x * b + self.bary.y * c
Hit.add_method('interpolate', interpolate)

# Var<float> interpolate(Expr<Hit> hit, Expr<float> a, Expr<float> b, Expr<float> c) noexcept {
#     return (1.0f - hit.bary.x - hit.bary.y) * a + hit.bary.x * b + hit.bary.y * c;
# }

# Var<float2> interpolate(Expr<Hit> hit, Expr<float2> a, Expr<float2> b, Expr<float2> c) noexcept {
#     return (1.0f - hit.bary.x - hit.bary.y) * a + hit.bary.x * b + hit.bary.y * c;
# }

# Var<float3> interpolate(Expr<Hit> hit, Expr<float3> a, Expr<float3> b, Expr<float3> c) noexcept {
#     return (1.0f - hit.bary.x - hit.bary.y) * a + hit.bary.x * b + hit.bary.y * c;
# }


class Accel:
    def __init__(self):
        self._accel = get_global_device().create_accel(lcapi.AccelUsageHint.FAST_TRACE)
        self.handle = self._accel.handle()

    def add(self, mesh, transform = float4x4.identity(), visible = True):
        self._accel.emplace_back(mesh.handle, transform, visible)

    def build(self):
        globalvars.stream.add(self._accel.build_command(lcapi.AccelBuildRequest.PREFER_UPDATE))

    @func
    def trace_closest(self, ray: Ray):
        uhit = _builtin_call(UHit, "TRACE_CLOSEST", self, ray)
        hit = Hit()
        hit.inst = _bitwise_cast(int, uhit.inst)
        hit.prim = _bitwise_cast(int, uhit.prim)
        hit.bary = uhit.bary
        return hit

    @func
    def trace_any(self, ray: Ray):
        return _builtin_call(bool, "TRACE_ANY", self, ray)

    @func
    def instance_transform(self, instance_id: int):
        return _builtin_call(float4x4, "INSTANCE_TO_WORLD_MATRIX", self, instance_id)

    @func
    def set_instance_transform(self, instance_id: int, mat: float4x4):
        _builtin_call("SET_INSTANCE_TRANSFORM", self, instance_id, mat)

    @func
    def set_instance_visibility(self, instance_id: int, vis: bool):
        _builtin_call("SET_INSTANCE_VISIBILITY", self, instance_id, vis)


class Mesh:
    def __init__(self, vertices, triangles):
        assert vertices.dtype == float3
        assert triangles.dtype == int and triangles.size%3==0
        # TODO: support buffer of structs or arrays
        self.handle = get_global_device().impl().create_mesh(
            vertices.handle, 0, 16, vertices.size,
            triangles.handle, 0, triangles.size//3,
            lcapi.AccelUsageHint.FAST_TRACE)
        globalvars.stream.add(lcapi.MeshBuildCommand.create(
            self.handle, lcapi.AccelBuildRequest.PREFER_UPDATE,
            vertices.handle, 0, vertices.size,
            triangles.handle, 0, triangles.size//3))
