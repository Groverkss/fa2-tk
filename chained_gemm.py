import torch
import sympy
import shark_turbine.kernel as tk

from shark_turbine.kernel.compiler import (
    builder,
    kernel_codegen,
    vector_codegen,
)
from shark_turbine.kernel._support import (
    indexing,
)

import shark_turbine.kernel.lang as tkl

BATCH = tkl.sym.BATCH
N_HEADS = tkl.sym.N_HEADS
N_CTX = tkl.sym.N_CTX
D_HEAD = tkl.sym.D_HEAD

BLOCK_N = tkl.sym.BLOCK_N
BLOCK_M = tkl.sym.BLOCK_M


@tk.gen.thread(N_CTX // BLOCK_M, BATCH * N_HEADS)
def chain_gemm(
    Q: tkl.InputBuffer[BATCH, N_HEADS, N_CTX, D_HEAD],
    K: tkl.InputBuffer[BATCH, N_HEADS, N_CTX, D_HEAD],
    V: tkl.InputBuffer[BATCH, N_HEADS, N_CTX, D_HEAD],
    O: tkl.OutputBuffer[BATCH, N_HEADS, N_CTX, D_HEAD],
):
    grid_n = tkl.program_id(0)
    grid_m = tkl.program_id(1)

    batch = grid_m // N_HEADS
    head = grid_m % N_HEADS

    q = tkl.load(Q, (batch, head, grid_n * BLOCK_M, 0), (BLOCK_M, D_HEAD))
    acc = tkl.constant((BLOCK_M, D_HEAD), tkl.f32, 0.0)

    @tkl.for_loop(0, N_CTX, BLOCK_N, init_args=[acc])
    def body(i, acc):
        k = tkl.load(K, (batch, head, i, 0), (BLOCK_N, D_HEAD))
        kT = tkl.transpose(k, (1, 0))

        qkT = tkl.constant((BLOCK_M, BLOCK_N), tkl.f32, 0.0)
        qkT = tkl.dot(q, kT, qkT)

        v = tkl.load(V, (batch, head, i, 0), (BLOCK_N, D_HEAD))
        acc = tkl.dot(qkT, v, acc)

        return (acc,)

    tkl.store(O, (batch, head, grid_n * BLOCK_M, 0), body[0])

Q = torch.randn(4, 48, 4096, 64)
K = torch.randn(4, 48, 4096, 64)
V = torch.randn(4, 48, 4096, 64)
O = torch.randn(4, 48, 4096, 64)

with tk.gen.TestLaunchContext(
    {
        BLOCK_N: 128,
        BLOCK_M: 256,
    }
):
    chain_gemm(Q, K, V, O)
