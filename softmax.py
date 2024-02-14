import torch
import shark_turbine.kernel as tk

import shark_turbine.kernel.lang as tkl

from shark_turbine.kernel._support.tracing import TestLaunchContext

M = tkl.sym.M
K = tkl.sym.K


@tk.gen.thread(M)
def softmax(input: tkl.InputBuffer[M, K], output: tkl.OutputBuffer[M, K]):
    row_index = tkl.program_id(0)
    row = tkl.load(input, (row_index, 0), (1, K))
    row_minus_max = row - tkl.max(row)
    numerator = tkl.exp2(row_minus_max)
    denominator = tkl.sum(numerator)
    softmax_output = numerator / denominator
    tkl.store(output, (row_index, 0), softmax_output)


input = tkl.InputBuffer[M, K](torch.randn(128, 256))
output = tkl.OutputBuffer[M, K](torch.zeros(128, 256))
with TestLaunchContext({M: 128}):
    softmax[128](input, output)
