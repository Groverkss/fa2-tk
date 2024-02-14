import torch
import shark_turbine.kernel as tk

import shark_turbine.kernel.lang as tkl

N = tkl.sym.N
M = tkl.sym.M
K = tkl.sym.K
BLOCK_SIZE = tkl.sym.BLOCK_SIZE


@tk.gen.thread(N // BLOCK_SIZE, M // BLOCK_SIZE)
def gemm(
    A: tkl.InputBuffer[N, K],
    B: tkl.InputBuffer[K, M],
    output: tkl.OutputBuffer[N, M],
):
    grid_n = tkl.program_id(0)
    grid_m = tkl.program_id(1)

    acc = tkl.constant((BLOCK_SIZE, BLOCK_SIZE), tkl.f32, 0.0)

    @tkl.for_loop(0, K // BLOCK_SIZE, init_args=[acc])
    def body(i, c):
        a = tkl.load(A, (grid_n, i * BLOCK_SIZE), (BLOCK_SIZE, BLOCK_SIZE))
        b = tkl.load(B, (i * BLOCK_SIZE, grid_m), (BLOCK_SIZE, BLOCK_SIZE))
        return (tkl.dot(a, b, c),)

    tkl.store(output, (grid_n, grid_m), body[0])


A = torch.randn(512, 1024)
B = torch.randn(1024, 2048)
output = torch.zeros(512, 2048)

with tk.gen.TestLaunchContext({BLOCK_SIZE: 64}):
    gemm(A, B, output)
