import torch
from shark_turbine.aot import export
import shark_turbine.kernel as tk

import shark_turbine.kernel.lang as tkl

N = tkl.sym.N
M = tkl.sym.M

@tk.gen.kernel(N)
def arith(input: tkl.InputBuffer[N, M, tkl.f16], output: tkl.OutputBuffer[N, M, tkl.f16]):
    n = tkl.program_id(0)
    a_16 = tkl.constant((64, 32), dtype=tkl.f16, value=1.0)
    b_16 = tkl.constant((64, 32), dtype=tkl.f16, value=2.0)
    c_32 = tkl.constant((64, 32), dtype=tkl.f32, value=3.0)
    c = (a_16 * b_16) + (a_16 - b_16)
    tkl.store(output, (0, 0), c)

class NN(torch.nn.Module):
    def forward(self, input):
        return arith(input)

input = torch.randn(64, 32)

model = NN()
exported = export(model, input)

# See internal linalg and async IR.
exported.import_to("iree_internal")
exported.print_readable()
