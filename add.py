import torch
import shark_turbine.kernel as tk

import shark_turbine.kernel.lang as tkl

N = tkl.sym.N
M = tkl.sym.M

@tk.gen.thread(N)
def arith(
    A: tkl.InputBuffer[N, M], B: tkl.InputBuffer[N, M], output: tkl.OutputBuffer[N, M]
):
    n = tkl.program_id(0)
    a = tkl.load(A, (n, 0), (1, M))
    b = tkl.load(B, (n, 0), (1, M))
    c = a + b
    tkl.store(output, (n, 0), c)


A = torch.rand(128, 128)
B = torch.rand(128, 128)
output = torch.empty(128, 128)

with tk.gen.TestLaunchContext():
    arith(A, B, output)
