import torch
import shark_turbine.kernel as tk

import shark_turbine.kernel.lang as tkl

N = tkl.sym.N
M = tkl.sym.M

@tk.gen.thread(N)
def arith(input: tkl.InputBuffer[N, M], output: tkl.OutputBuffer[N, M]):
    n = tkl.program_id(0)
    a_16 = tkl.constant((64, 32), dtype=tkl.f16, value=1.0)
    b_16 = tkl.constant((64, 32), dtype=tkl.f16, value=2.0)
    c_32 = tkl.constant((64, 32), dtype=tkl.f32, value=3.0)
    c = (a_16 * b_16) + (a_16 - b_16)

input = torch.randn(64, 32)
output = torch.empty(64, 32)
with tk.gen.TestLaunchContext():
    arith(input, output)
