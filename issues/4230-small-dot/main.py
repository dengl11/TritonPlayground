import torch
import sys
import os
import triton
import triton.language as tl

K = 4

@triton.jit
def matmul_kernel(p1, p2, rp, K: tl.constexpr):
    region = tl.arange(0, K)[:, None] * K + tl.arange(0, K)[None, :]
    m1 = tl.load(p1 + region)
    m2 = tl.load(p2+ region)
    r = tl.dot(m1, m2)
    tl.store(rp + region, r)


m1 = torch.rand([K, K], dtype=torch.float32).cuda()
m2 = torch.rand([K, K], dtype=torch.float32).cuda()
torch_result = m1 @ m2
print(f"Torch: {torch_result}")

triton_result = torch.empty([K, K], dtype=torch.float32).cuda()
matmul_kernel[(1,)](m1, m2, triton_result, K)
print(f"Triton: {triton_result}")
print(torch.allclose(torch_result, triton_result, atol=1e-2))