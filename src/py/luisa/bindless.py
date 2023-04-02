import lcapi
from lcapi import uint2, uint3
from . import globalvars
from .globalvars import get_global_device as device
from .mathtypes import *
from . import Buffer, Image2D, Image3D
from .types import BuiltinFuncBuilder, to_lctype, uint, int16, uint16, int16_2, float16_2, uint16_2, int16_3, float16_3, uint16_3, int16_4, float16_4, uint16_4
from .builtin import check_exact_signature
from .func import func
from .builtin import _builtin_call

class BindlessArray:
    def __init__(self, n_slots = 65536):
        self.array = device().impl().create_bindless_array(n_slots)
        self.handle = lcapi.get_bindless_handle(self.array)
    def __del__(self):
        if(self.array != None):
            local_device = device()
            if local_device != None:
                local_device.impl().destroy_bindless_array(self.array)

    @staticmethod
    def bindless_array(dic):
        arr = BindlessArray.empty()
        for i in dic:
            arr.emplace(i, dic[i])
        arr.update()
        return arr

    @staticmethod
    def empty(n_slots = 65536):
        return BindlessArray(n_slots)

    def emplace(self, idx, res, filter = None, address = None, byte_offset = 0):
        if type(res) is Buffer:
            device().impl().emplace_buffer_in_bindless_array(self.array, idx, res.handle, byte_offset)
        elif type(res) is Image2D:
            if res.dtype != float:
                raise TypeError("Type of emplaced Image2D must be float")
            if filter == None:
                filter = lcapi.Filter.LINEAR_POINT
            if address == None:
                address = lcapi.Address.REPEAT
            sampler = lcapi.Sampler(filter, address)
            device().impl().emplace_tex2d_in_bindless_array(self.array, idx, res.handle, sampler)
        elif type(res) is Image3D:
            if res.dtype != float:
                raise TypeError("Type of emplaced Image3D must be float")
            if filter == None:
                filter = lcapi.Filter.LINEAR_POINT
            if address == None:
                address = lcapi.Address.REPEAT
            sampler = lcapi.Sampler(filter, address)
            device().impl().emplace_tex3d_in_bindless_array(self.array, idx, res.handle, sampler)
        else:
            raise TypeError(f"can't emplace {type(res)} in bindless array")

    def remove_buffer(self, idx):
        device().impl().remove_buffer_in_bindless_array(self.array, idx)
        
    def remove_texture2d(self, idx):
        device().impl().remove_tex2d_in_bindless_array(self.array, idx)
    def remove_texture3d(self, idx):
        device().impl().remove_tex3d_in_bindless_array(self.array, idx)
        
    def update(self, sync = False, stream = None):
        if stream is None:
            stream = globalvars.stream
        stream.update_bindless(self.array)
        if sync:
            stream.synchronize()

    # @func
    # def buffer_read(self: BindlessArray, dtype: type, buffer_index: int, element_index: int):
    #     return _builtin_call(dtype, "BINDLESS_BUFFER_READ", self, buffer_index, element_index)
    # might not be possible, because "type" is not a valid data type in LC

    @BuiltinFuncBuilder
    def buffer_read(*argnodes): # (dtype, buffer_index, element_index)
        check_exact_signature([type, int, uint], argnodes[1:], "buffer_read")
        dtype = argnodes[1].expr
        expr = lcapi.builder().call(to_lctype(dtype), lcapi.CallOp.BINDLESS_BUFFER_READ, [x.expr for x in [argnodes[0]] + list(argnodes[2:])])
        return dtype, expr

    @func
    def texture2d_read(self, texture2d_index, coord: uint2):
        return _builtin_call(float4, "BINDLESS_TEXTURE2D_READ", self, texture2d_index, coord)

    @func
    def texture2d_sample(self, texture2d_index, uv: float2):
        return _builtin_call(float4, "BINDLESS_TEXTURE2D_SAMPLE", self, texture2d_index, uv)
    @func
    def texture2d_sample_mip(self, texture2d_index, uv: float2, mip):
        return _builtin_call(float4, "BINDLESS_TEXTURE2D_SAMPLE_LEVEL", self, texture2d_index, uv, mip)

    @func
    def texture2d_sample_grad(self, texture2d_index, uv: float2, ddx: float2, ddy: float2):
        return _builtin_call(float4, "BINDLESS_TEXTURE2D_SAMPLE_GRAD", self, texture2d_index, uv, ddx, ddy)

    @func
    def texture2d_size(self, texture2d_index):
        return _builtin_call(uint2, "BINDLESS_TEXTURE2D_SIZE", self, texture2d_index)
    @func
    def texture3d_read(self, texture3d_index, coord: uint3):
        return _builtin_call(float4, "BINDLESS_TEXTURE3D_READ", self, texture3d_index, coord)

    @func
    def texture3d_sample(self, texture3d_index, uv: float3):
        return _builtin_call(float4, "BINDLESS_TEXTURE3D_SAMPLE", self, texture3d_index, uv)
    @func
    def texture3d_sample_mip(self, texture3d_index, uv: float3, mip):
        return _builtin_call(float4, "BINDLESS_TEXTURE3D_SAMPLE_LEVEL", self, texture3d_index, uv, mip)

    @func
    def texture3d_sample_grad(self, texture3d_index, uv: float3, ddx: float3, ddy: float3):
        return _builtin_call(float4, "BINDLESS_TEXTURE3D_SAMPLE_GRAD", self, texture3d_index, uv, ddx, ddy)

    @func
    def texture3d_size(self, texture3d_index):
        return _builtin_call(uint3, "BINDLESS_TEXTURE3D_SIZE", self, texture3d_index)

bindless_array = BindlessArray.bindless_array