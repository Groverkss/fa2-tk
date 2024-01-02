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
def control_flow(input: tkl.KernelBuffer[M, K], output: tkl.KernelBuffer[M, K]):
    row_idx = tkl.program_id(0)
    sum = input[row_idx, 0]
    prefetch = input[row_idx, 1]

    @tkl.for_loop(2, 5, init_args=[sum, prefetch])
    def prefetch_sum(i, iter_args):
        new_sum = iter_args[0] + iter_args[1]
        new_prefetch = input[row_idx, i]
        return new_sum, new_prefetch

    output[row_idx, 0] = prefetch_sum[0]

trace = control_flow._trace
print(trace.region_graph)
mb = builder.ModuleBuilder()
with indexing.IndexingContext() as idxc:
    idxc.bind_constant(M, 128)
    idxc.bind_constant(K, 64)

    sig = kernel_codegen.KernelSignature()
    sig.add_from_graph_placeholders(trace.get_root_graph())
    sig.add_grid(control_flow.grid_type)
    print(sig)
    bound_sig, func_op = kernel_codegen.FunctionalKernelSignature.create(sig, mb)
    emitter = vector_codegen.ThreadEmitter(bound_sig, trace)
    emitter.emit()
    emitter.finish()
    print(mb.module_op.get_asm())
    mb.module_op.verify()
