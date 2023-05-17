from luisa import *
from luisa.builtin import *
from luisa.types import *
from luisa.util import *
import numpy as np
init()
dispatch_count = 16
kernel_dispatch_size = 64


@func
def dispatch(buffer):
    set_block_size(kernel_dispatch_size, 1, 1)
    old = buffer.atomic_fetch_add(kernel_id(), dispatch_size().x)


dispatch_buffer = IndirectDispatchBuffer(dispatch_count)


@func
def clear_indirect():
    set_block_size(1, 1, 1)
    dispatch_buffer.clear()


@func
def emplace_indirect():
    set_block_size(dispatch_count, 1, 1)
    k_id = dispatch_buffer.emplace(
        uint3(kernel_dispatch_size, 1, 1), uint3(dispatch_id().x, 1, 1))


buffer = Buffer(16, uint)
arr = np.zeros(16, dtype=np.uint32)
out_arr = np.zeros(16, dtype=np.uint32)
buffer.copy_from(arr)
clear_indirect(dispatch_size=(1, 1, 1))
emplace_indirect(dispatch_size=(dispatch_count, 1, 1))
dispatch(buffer, dispatch_size=dispatch_buffer)
buffer.copy_to(out_arr)
result = ""
for i in out_arr:
    result += str(i) + " "
print("Result should be: 0 1 4 9 16 25 36 49 64 81 100 121 144 169 196 225")
print(f"Result: {result}")
