import torch
import shark_turbine.kernel as tk

import shark_turbine.kernel.lang as tkl

N = tkl.sym.N
M = tkl.sym.M
K = tkl.sym.K
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
BLOCK_K = tkl.sym.BLOCK_K

@tk.gen.thread(M // BLOCK_M, N // BLOCK_N)
def gemm(
    A: tkl.InputBuffer[M, K, tkl.f16],
    B: tkl.InputBuffer[N, K, tkl.f16],
    output: tkl.OutputBuffer[M, N, tkl.f16],
):
    grid_n = tkl.program_id(0)
    grid_m = tkl.program_id(1)

    acc = tkl.constant((BLOCK_M, BLOCK_N), tkl.f32, 0.0)

    @tkl.for_loop(0, K // BLOCK_K, init_args=[acc])
    def body(i, c):
        a = tkl.load(A, (grid_n, i * BLOCK_M), (BLOCK_M, BLOCK_K))
        b = tkl.load(B, (i * BLOCK_N, grid_m), (BLOCK_N, BLOCK_K))
        b = tkl.transpose(b, (1, 0))
        return (tkl.dot(a, b, c),)

    tkl.store(output, (grid_n, grid_m), body[0])


A = torch.randn(128, 2048)
B = torch.randn(1280, 2048)
output = torch.zeros(128, 1280)

with tk.gen.TestLaunchContext({BLOCK_M: 128, BLOCK_N: 256, BLOCK_K: 128}):
    gemm(A, B, output)
