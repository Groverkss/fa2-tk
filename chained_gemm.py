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
    Q: tkl.KernelBuffer[BATCH, N_HEADS, N_CTX, D_HEAD],
    K: tkl.KernelBuffer[BATCH, N_HEADS, N_CTX, D_HEAD],
    V: tkl.KernelBuffer[BATCH, N_HEADS, N_CTX, D_HEAD],
    O: tkl.KernelBuffer[BATCH, N_HEADS, N_CTX, D_HEAD],
):
    grid_n = tkl.program_id(0)
    grid_m = tkl.program_id(1)

    batch = grid_m // N_HEADS
    head = grid_m % N_HEADS

    q = tkl.load(Q, (batch, head, grid_n * BLOCK_M, 0), (BLOCK_M, D_HEAD))
    acc = tkl.constant((BLOCK_M, D_HEAD), torch.float32, 0.0)

    @tkl.for_loop(0, N_CTX, BLOCK_N, init_args=[acc])
    def body(i, acc):
        k = tkl.load(K, (batch, head, i, 0), (BLOCK_N, D_HEAD))
        kT = tkl.transpose(k, (1, 0))

        qkT = tkl.constant((BLOCK_M, BLOCK_N), torch.float32, 0.0)
        qkT = tkl.dot(q, kT, qkT)

        v = tkl.load(V, (batch, head, i, 0), (BLOCK_N, D_HEAD))
        acc = tkl.dot(qkT, v, acc)

        return (acc,)

    tkl.store(O, (batch, head, grid_n * BLOCK_M, 0), acc)


trace = chain_gemm._trace
print(trace.region_graph)
mb = builder.ModuleBuilder()
with indexing.IndexingContext() as idxc:
    idxc.bind_shaped(
        "Q", tkl.KernelBuffer[BATCH, N_HEADS, N_CTX, D_HEAD], (4, 48, 4096, 64)
    )
    idxc.bind_shaped(
        "K", tkl.KernelBuffer[BATCH, N_HEADS, N_CTX, D_HEAD], (4, 48, 4096, 64)
    )
    idxc.bind_shaped(
        "V", tkl.KernelBuffer[BATCH, N_HEADS, N_CTX, D_HEAD], (4, 48, 4096, 64)
    )
    idxc.bind_shaped(
        "O", tkl.KernelBuffer[BATCH, N_HEADS, N_CTX, D_HEAD], (4, 48, 4096, 64)
    )
    idxc.bind_constant(BLOCK_M, 128)
    idxc.bind_constant(BLOCK_N, 64)
    idxc.finalize()

    sig = kernel_codegen.KernelSignature()
    sig.add_from_graph_placeholders(trace.get_root_graph())
    sig.add_grid(chain_gemm.grid_type)
    print(sig)
    bound_sig, func_op = kernel_codegen.FunctionalKernelSignature.create(sig, mb)
    emitter = vector_codegen.ThreadEmitter(bound_sig, trace)
    emitter.emit()
    emitter.finish()
    print(mb.module_op.get_asm())
    mb.module_op.verify()
