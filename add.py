import torch
from shark_turbine.aot import export
import shark_turbine.kernel as tk

import shark_turbine.kernel.lang as tkl

N = tkl.sym.N
M = tkl.sym.M

@tk.gen.kernel(N)
def arith(
    A: tkl.InputBuffer[N, M, tkl.f16], B: tkl.InputBuffer[N, M, tkl.f16], output: tkl.OutputBuffer[N, M, tkl.f16]
):
    n = tkl.program_id(0)
    a = tkl.load(A, (n, 0), (1, M))
    b = tkl.load(B, (n, 0), (1, M))
    c = a + b
    tkl.store(output, (n, 0), c)

class NN(torch.nn.Module):
    def forward(self, a, b):
        return arith(a, b)


A = torch.rand(128, 128, dtype=torch.float16)
B = torch.rand(128, 128, dtype=torch.float16)

model = NN()
exported = export(model, A, B)

# See internal linalg and async IR.
exported.import_to("iree_internal")
exported.print_readable()
