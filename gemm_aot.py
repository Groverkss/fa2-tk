import logging

logging.basicConfig(level=logging.DEBUG)

import torch
from shark_turbine.aot import export
import shark_turbine.kernel as tk
import shark_turbine.kernel.lang as tkl


N = tkl.sym.N
M = tkl.sym.M
K = tkl.sym.K
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
BLOCK_K = tkl.sym.BLOCK_K


@tk.gen.kernel(
    M // BLOCK_M, N // BLOCK_N, constants={BLOCK_M: 16, BLOCK_N: 16, BLOCK_K: 16}
)
def matmul(
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


class NN(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return matmul(a, b)


model = NN()
a = torch.ones(64, 64, dtype=torch.float16)
b = torch.ones(64, 64, dtype=torch.float16)
exported = export(model, a, b)
exported.print_readable()

# See internal linalg and async IR.
exported.import_to("iree_internal")
exported.print_readable()

# Broken on IREE compilation failure.
# eager_results = model.forward(a, b)
# print(eager_results)
