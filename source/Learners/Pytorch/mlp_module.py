import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.set_default_tensor_type(torch.DoubleTensor)

class MLP(nn.Module):
    def __init__(self, input_dim, L1, L2, output_dim):
        super(MLP, self).__init__()
        # input_dim = 5;
        # L1 = 10;
        # L2 = 10;
        # output_dim = 1;

        self.input_dim = input_dim
        self.L1 = L1
        self.L2 = L2
        self.output_dim = output_dim

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(self.input_dim, self.L1)
        self.fc2 = nn.Linear(self.L1, self.L2)
        self.fc3 = nn.Linear(self.L2, self.output_dim)


    def forward(self, input_dict):
        print("PYTORCH: Input to forward:")
        print(input_dict)

        seqVec = input_dict["seqVec"]
        policyType = input_dict["policyType"]
        print("PYTORCH to python.")
        print("seqVec", seqVec) # Vector of Sequence*

        print("seqVec[0]", seqVec[0]) # Sequence* 

        print("seqVec[0].rewards[0]", seqVec[0].rewards[0]) # Real number 
        print("seqVec[0].rewards", seqVec[0].rewards) # Vector of double 

        print("seqVec[0].actions", seqVec[0].actions)

        print("seqVec[0].actions[0][0]", seqVec[0].actions[0][0]) # Vector



        # print("seqVec[0].actions=", seqVec[0].actions)
        # print("seqVec=", seqVec, seqVec[0].states)
        # print("seqVec=", seqVec, seqVec[0].states[0])
        # print("seqVec=", seqVec, seqVec[0].states[1])
        # print("seqVec=", seqVec, seqVec[0].states[2])

        print("policyType=", policyType)

        # print("modifying seqVec[0].states[0]")

        # seqVec[0].states[0] = 1000
        # temp = np.array(seqVec[0].states)
        # # print(np.sum(temp))
        # # print("seqVec=", seqVec, seqVec[0].states[0])

        # # input_ = np.array(seqVec[0].states)
        # input_ = seqVec[0].states
        # print("input: ")
        # print(input_)
        # output_ = self.forward3(input_)
        # print("output: ")
        # print(output_)



        # output_detached = output_.detach().numpy()
        # output_detached = list(output_detached)
        # print(output_detached)
        # print(type(output_detached))
        # # return 0

        # # seqVec[0].states[0] = output_detached[0]
        # # seqVec[0].states[1] = output_detached[1]
        # # seqVec[0].states[2] = output_detached[2]
        # # seqVec[0].states[3] = output_detached[3]
        # # seqVec[0].states[4] = output_detached[4]
        # print(seqVec[0].actions)

        # for i in range(len(list(seqVec[0].actions))):
        #     seqVec[0].actions[i] = output_detached[i]
            
        # # seqVec[0].actions[0] = output_detached[0]
        # # seqVec[0].actions[1] = output_detached[1]
        # # seqVec[0].actions[2] = output_detached[2]
        # print(seqVec[0].actions)
        # temp = seqVec[0].actions
        # temp = list(temp)
        # print(temp)
        # print(len(temp))

        # # seqVec[0].actions = output_detached
        return 0


    def forward2(self, x):
        # return x;
        # return list(x);
        a = torch.DoubleTensor(x)
        print("FORWARD2")
        b = a.view(1, -1)
        print("FORWARD3")
        c = F.relu(self.fc1(b))
        return c

    def forward3(self, x):
        # import pybind11
        # print(pybind11.__file__)
        # print(pybind11.__version__)

        # print(x)
        # print(x.dtype)
        # print(self.input_dim)
        # print(self.L1)
        # print(self.L2)
        # print(self.output_dim)
        # print(x)
        # print(x.dtype)
        # print(np.shape(x))
        print(x)

        a = torch.DoubleTensor(x)
        print("FORWARD2")
        b = a.view(1, -1)
        print("FORWARD3")
        c = F.relu(self.fc1(b))
        print("FORWARD4")
        d = F.relu(self.fc2(c))
        print("FORWARD5")
        e = self.fc3(d)
        print("FORWARD6")
        f = e[0]
        print(f.dtype)
        print("FORWARD7")
        print(f)

        # g = f.detach().numpy()
        # # g = np.array(f.data)
        # print("RETURNING8")
        # print(g)
        # print(type(g))
        # z = list(g)
        # print(type(z))
        # print(z)

        return f;
        # return f;
        # return 0



# if __name__ == "__main__":
#     input_dim = 5;
#     L1 = 10;
#     L2 = 10;
#     output_dim = 3;
#     net = MLP(input_dim, L1, L2, output_dim)
#     print(net)
#     # input = torch.randn(input_dim)
#     # print(input.to_numpy())
#     input = np.array([1, 2, 3, 4, 5])
#     print(input)

#     # print(input.size())
#     output = net(input)
#     print(dir(output))
#     print(output.__dict__)
#     print(output)
#     # print(input.size())
#     # print(output.size())
