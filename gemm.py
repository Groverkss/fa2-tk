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
BLOCK_SIZE = tkl.sym.BLOCK_SIZE

@tk.gen.thread(N // BLOCK_SIZE, M // BLOCK_SIZE)
def gemm(
    A: tkl.KernelBuffer[N, K],
    B: tkl.KernelBuffer[K, M],
    output: tkl.KernelBuffer[N, M],
):
    grid_n = tkl.program_id(0)
    grid_m = tkl.program_id(1)

    acc = tkl.constant((BLOCK_SIZE, BLOCK_SIZE), torch.float32, 0.0)

    @tkl.for_loop(0, K // BLOCK_SIZE, init_args=[acc])
    def body(i, c):
        a = tkl.load(A, (grid_n, i * BLOCK_SIZE), (BLOCK_SIZE, BLOCK_SIZE))
        b = tkl.load(B, (i * BLOCK_SIZE, grid_m), (BLOCK_SIZE, BLOCK_SIZE))
        return (tkl.dot(a, b, c),)

    tkl.store(output, (grid_n, grid_m), body[0])


trace = gemm._trace
print(trace.region_graph)
mb = builder.ModuleBuilder()
with indexing.IndexingContext() as idxc:
    idxc.bind_shaped("A", tkl.KernelBuffer[N, K], (512, 1024))
    idxc.bind_shaped("B", tkl.KernelBuffer[K, M], (1024, 2048))
    idxc.bind_shaped("output", tkl.KernelBuffer[N, M], (512, 2048))
    idxc.bind_constant(BLOCK_SIZE, 32)
    idxc.finalize()

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
