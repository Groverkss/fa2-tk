import torch
from shark_turbine.aot import export
import shark_turbine.kernel as tk
import shark_turbine.kernel.lang as tkl


M = tkl.sym.M
N = tkl.sym.K

@tk.gen.kernel(M)
def softmax(
    input: tkl.InputBuffer[M, N, tkl.f32], output: tkl.OutputBuffer[M, N, tkl.f32]
):
    row_index = tkl.program_id(0)
    row = tkl.load(input, (row_index, 0), (1, N))
    row_minus_max = row - tkl.max(row)
    numerator = tkl.exp2(row_minus_max)
    denominator = tkl.sum(numerator)
    softmax_output = numerator / denominator
    tkl.store(output, (row_index, 0), softmax_output)


class NN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64, dtype=torch.float32)

    def forward(self, x):
        x = self.linear(x)
        x = softmax(x)
        return x


model = NN()
a = torch.ones(64, 64, dtype=torch.float32)
exported = export(model, a)

# See torch IR
exported.print_readable()

# See internal linalg and async IR.
exported.import_to("iree_internal")
exported.print_readable()

# Compile and Run
compiled = exported.compile(save_to="softmax_aot.vmfb", target_backends="rocm")

# Eager execution
eager_results = model.forward(a)
print(eager_results)
