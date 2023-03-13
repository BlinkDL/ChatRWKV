import math
from torch import nn
from torch.autograd import Function
import torch

import rwkv_cpp

torch.manual_seed(42)


class RWKVFunction(Function):
    @staticmethod
    def forward(ctx, input, weights, bias, old_h, old_cell):
        outputs = rwkv_cpp.forward(input, weights, bias, old_h, old_cell)
        new_h, new_cell = outputs[:2]
        variables = outputs[1:] + [weights]
        ctx.save_for_backward(*variables)

        return new_h, new_cell

class RWKV(nn.Module):
    def __init__(self, input_features, state_size):
        super(RWKV, self).__init__()
        self.input_features = input_features
        self.state_size = state_size
        self.weights = nn.Parameter(
            torch.Tensor(3 * state_size, input_features + state_size))
        self.bias = nn.Parameter(torch.Tensor(1, 3 * state_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, state):
        return RWKVFunction.apply(input, self.weights, self.bias, *state)
