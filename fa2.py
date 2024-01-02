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

M = tkl.sym.M
K = tkl.sym.K

@tk.gen.thread(M)
def fa2(input: tkl.KernelBuffer[M, K], output: tkl.KernelBuffer[M, K]):
    pass

gm = fa2._trace.gm
print(gm.graph)
mb = builder.ModuleBuilder()
with indexing.IndexingContext() as idxc:
    idxc.bind_constant(M, 128)
    idxc.bind_constant(K, 64)

    sig = kernel_codegen.KernelSignature()
    sig.add_from_graph_placeholders(gm.graph)
    sig.add_grid(fa2.grid_type)
    print(sig)
    bound_sig, func_op = kernel_codegen.FunctionalKernelSignature.create(sig, mb)
    emitter = vector_codegen.ThreadEmitter(bound_sig)
    emitter.emit_graph(gm.graph)
    emitter.finish()
    print(mb.module_op.get_asm())
    mb.module_op.verify()
