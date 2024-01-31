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

BATCH = tkl.sym.BATCH
N_HEADS = tkl.sym.N_HEADS
N_CTX = tkl.sym.N_CTX
D_HEAD = tkl.sym.D_HEAD

BLOCK_N = tkl.sym.BLOCK_N
BLOCK_M = tkl.sym.BLOCK_M


@tk.gen.thread(N_CTX // BLOCK_M, BATCH * N_HEADS)
def fused_attention(
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
    acc_init = tkl.constant((BLOCK_M, D_HEAD), torch.float32, 0.0)
    max_stat_init = tkl.constant((BLOCK_M,), torch.float32, -1e9)
    sum_stat_init = tkl.constant((BLOCK_M,), torch.float32, 0.0)

    @tkl.for_loop(0, N_CTX, BLOCK_N, init_args=[max_stat_init, sum_stat_init, acc_init])
    def body(i, old_max, old_sum, old_acc):
        k = tkl.load(K, (batch, head, i, 0), (BLOCK_N, D_HEAD))
        kT = tkl.transpose(k, (1, 0))

        qkT = tkl.constant((BLOCK_M, BLOCK_N), torch.float32, 0.0)
        qkT = tkl.dot(q, kT, qkT)

        new_max = tkl.max(qkT, axis=1, acc=old_max)
        broadcasted_max = tkl.broadcast_in_dim(new_max, (BLOCK_M, BLOCK_N), (0,))
        partial_softmax = tkl.exp2(qkT - broadcasted_max)
        scale_factor = tkl.exp2(old_max - new_max)
        scaled_old_sum = scale_factor * old_sum
        new_sum = tkl.sum(partial_softmax, axis=1, acc=scaled_old_sum)
        broadcasted_scale_factor = tkl.broadcast_in_dim(
            scale_factor, (BLOCK_M, D_HEAD), (0,)
        )
        new_acc = old_acc * broadcasted_scale_factor

        v = tkl.load(V, (batch, head, i, 0), (BLOCK_N, D_HEAD))
        new_acc = tkl.dot(qkT, v, new_acc)

        return (new_max, new_sum, new_acc)

    sum_stat = body[1]
    result = body[2]
    one = tkl.constant((BLOCK_M,), torch.float32, 1.0)
    one_by_sum = (one / sum_stat)
    result = tkl.broadcast_in_dim(one_by_sum, (BLOCK_M, D_HEAD), (0,)) * result

    tkl.store(O, (batch, head, grid_n * BLOCK_M, 0), result)


trace = fused_attention._trace
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
    sig.add_grid(fused_attention.grid_type)
    print(sig)
    bound_sig, func_op = kernel_codegen.FunctionalKernelSignature.create(sig, mb)
    emitter = vector_codegen.ThreadEmitter(bound_sig, trace)
    emitter.emit()
    emitter.finish()
    print(mb.module_op.get_asm())
    mb.module_op.verify()
