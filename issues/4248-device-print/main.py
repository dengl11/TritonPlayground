import torch
import sys
import os
import triton
import triton.language as tl

@triton.jit
def ndscore_kernel(ptr):
    value = tl.load(ptr)
    # tl.device_print("pid", pid)
    tl.device_print("value in kernel from device_print", value)
    # tl.device_print("programs", tl.num_programs(0))
    # tl.store(ptr, value + 1)

# @triton.jit
# def ndscore_kernel(ptr):
#     value = tl.load(ptr)
#     print("value in kernel", value)
#     tl.store(ptr, value + 1)

# ptr = torch.tensor(-7, dtype=torch.int32).cuda()
# print("value before kernel", ptr.item())
# ndscore_kernel[(1,)](ptr)


ptr = torch.tensor(2200000000, dtype=torch.uint32).cuda()
print("value before kernel", ptr.item())
ndscore_kernel[(1,)](ptr)