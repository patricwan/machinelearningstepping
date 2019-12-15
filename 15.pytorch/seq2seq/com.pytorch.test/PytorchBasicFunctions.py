import numpy as np
import torch

import torch.nn as nn

def testSimpleNumpy():
    lst1 = [3.14, 2.17, 0, 1, 2]
    nd1 = np.array(lst1)
    print(nd1)
    # [3.14 2.17 0.   1.   2.  ]
    print(type(nd1))

def pyTorchTypes():
    x = torch.tensor([3,5,6])
    y = torch.rand(4,5)
    print(x)
    print(y)

class RNNSimple(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNSimple, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim = 1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
    return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

if __name__ == '__main__':
    testSimpleNumpy()
    pyTorchTypes()