import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        input_dim = 2;
        L1 = 10;
        L2 = 10;
        output_dim = 3;

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(input_dim, L1)
        self.fc2 = nn.Linear(L1, L2)
        self.fc3 = nn.Linear(L2, output_dim)

    def forward(self, x):
        x = x.view(1, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    net = MLP()
    print(net)
    input_dim = 2
    input = torch.randn(input_dim)
    print(input.size())
    output = net(input)
    print(output)
    print(input.size())
    print(output.size())
