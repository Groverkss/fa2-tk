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
def softmax(input: tkl.KernelBuffer[M, K], output: tkl.KernelBuffer[M, K]):
    row_index = tkl.program_id(0)
    row = input[row_index, :]
    row_minus_max = row - torch.max(row)
    numerator = torch.exp(row_minus_max)
    denominator = torch.sum(numerator)
    softmax_output = numerator / denominator
    output[row_index, :] = softmax_output


trace = softmax._trace
print(trace.region_graph)
mb = builder.ModuleBuilder()
with indexing.IndexingContext() as idxc:
    idxc.bind_constant(M, 128)
    idxc.bind_constant(K, 64)

    sig = kernel_codegen.KernelSignature()
    sig.add_from_graph_placeholders(trace.get_root_graph())
    sig.add_grid(softmax.grid_type)
    print(sig)
    bound_sig, func_op = kernel_codegen.FunctionalKernelSignature.create(sig, mb)
    emitter = vector_codegen.ThreadEmitter(bound_sig, trace)
    emitter.emit()
    emitter.finish()
    print(mb.module_op.get_asm())
    mb.module_op.verify()
