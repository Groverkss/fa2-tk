import torch
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

N = tkl.sym.N
M = tkl.sym.M
K = tkl.sym.K

GRID_N = tkl.sym.GRID_N
GRID_M = tkl.sym.GRID_M


def inner_gemm(
    A: tkl.KernelBuffer[N, K],
    B: tkl.KernelBuffer[K, M],
    output: tkl.KernelBuffer[N, M],
    k: int,
    block_size: int,
):
    grid_n = tkl.program_id(0)
    grid_m = tkl.program_id(1)

    acc = tkl.constant((block_size, block_size), torch.float32, 0.0)

    @tkl.for_loop(0, k // block_size, init_args=[acc])
    def body(i, c):
        a = tkl.load(A, (grid_n, i * block_size), (block_size, block_size))
        b = tkl.load(B, (i * block_size, grid_m), (block_size, block_size))
        return (tkl.dot(a, b, c),)

    tkl.store(output, (grid_n, grid_m), body[0])


@tk.gen.thread(GRID_N, GRID_M)
def gemm(
    A: tkl.KernelBuffer[N, K],
    B: tkl.KernelBuffer[K, M],
    output: tkl.KernelBuffer[N, M],
):
    # TODO: We should find a way to parameterize these so we can autotune over them.
    # TODO: Ideally, we should be getting k from the symbol. The symbol value
    # is currently not available at tracing time which is a problem.
    k = 512
    block_size = 32
    inner_gemm(A, B, output, k, block_size)


trace = gemm._trace
print(trace.region_graph)
mb = builder.ModuleBuilder()
with indexing.IndexingContext() as idxc:
    BLOCK_SIZE = 32
    idxc.bind_constant(N, 512)
    idxc.bind_constant(M, 512)
    idxc.bind_constant(K, 512)
    idxc.bind_constant(GRID_N, 512 // BLOCK_SIZE)
    idxc.bind_constant(GRID_M, 512 // BLOCK_SIZE)

    sig = kernel_codegen.KernelSignature()
    sig.add_from_graph_placeholders(trace.get_root_graph())
    sig.add_grid(gemm.grid_type)
    print(sig)
    bound_sig, func_op = kernel_codegen.FunctionalKernelSignature.create(
        sig, mb
    )
    emitter = vector_codegen.ThreadEmitter(bound_sig, trace)
    emitter.emit()
    emitter.finish()
    print(mb.module_op.get_asm())
    mb.module_op.verify()
