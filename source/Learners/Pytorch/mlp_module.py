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

        vectorMiniBatch = input_dict["vectorMiniBatch"]
        # policyType = input_dict["policyType"]

        print("PYTORCH to python.")
        print("vectorMiniBatch = ", vectorMiniBatch) # Vector of Sequence*
        # print("policyType = ", policyType) # Vector of Sequence*

        # print("vectorMiniBatch[0]", vectorMiniBatch[0]) # Sequence* 

        # print("vectorMiniBatch[0].rewards[0]", vectorMiniBatch[0].rewards[0]) # Real number 
        # print("vectorMiniBatch[0].rewards", vectorMiniBatch[0].rewards) # Vector of double 

        # print("vectorMiniBatch[0].actions", vectorMiniBatch[0].actions)

        # print("vectorMiniBatch[0].actions[0][0]", vectorMiniBatch[0].actions[0][0]) # Vector



        # print("vectorMiniBatch[0].actions=", vectorMiniBatch[0].actions)
        # print("vectorMiniBatch=", vectorMiniBatch, vectorMiniBatch[0].states)
        # print("vectorMiniBatch=", vectorMiniBatch, vectorMiniBatch[0].states[0])
        # print("vectorMiniBatch=", vectorMiniBatch, vectorMiniBatch[0].states[1])
        # print("vectorMiniBatch=", vectorMiniBatch, vectorMiniBatch[0].states[2])

        # print("policyType=", policyType)

        # print("modifying vectorMiniBatch[0].states[0]")

        # vectorMiniBatch[0].states[0] = 1000
        # temp = np.array(vectorMiniBatch[0].states)
        # # print(np.sum(temp))
        # # print("vectorMiniBatch=", vectorMiniBatch, vectorMiniBatch[0].states[0])

        # # input_ = np.array(vectorMiniBatch[0].states)
        # input_ = vectorMiniBatch[0].states
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

        # # vectorMiniBatch[0].states[0] = output_detached[0]
        # # vectorMiniBatch[0].states[1] = output_detached[1]
        # # vectorMiniBatch[0].states[2] = output_detached[2]
        # # vectorMiniBatch[0].states[3] = output_detached[3]
        # # vectorMiniBatch[0].states[4] = output_detached[4]
        # print(vectorMiniBatch[0].actions)

        # for i in range(len(list(vectorMiniBatch[0].actions))):
        #     vectorMiniBatch[0].actions[i] = output_detached[i]
            
        # # vectorMiniBatch[0].actions[0] = output_detached[0]
        # # vectorMiniBatch[0].actions[1] = output_detached[1]
        # # vectorMiniBatch[0].actions[2] = output_detached[2]
        # print(vectorMiniBatch[0].actions)
        # temp = vectorMiniBatch[0].actions
        # temp = list(temp)
        # print(temp)
        # print(len(temp))

        # # vectorMiniBatch[0].actions = output_detached
        return 0

    def forwardVector(self, input_):
        # import pybind11
        # print(pybind11.__file__)
        # print(pybind11.__version__)
        print("PYTORCH: Input=")
        print(input_)
        # SHAPE: [K, D]
        input_tensor = torch.DoubleTensor(input_)
        print("PYTORCH: FORWARD1")
        input_tensor = input_tensor.view(1, -1)
        print(input_tensor)
        # SHAPE: [K, D]
        print(input_tensor.size())

        print("PYTORCH: FORWARD2")
        temp_1 = F.relu(self.fc1(input_tensor))
        print(temp_1)
        print("PYTORCH: FORWARD3")
        temp_2 = F.relu(self.fc2(temp_1))
        print(temp_2)
        print("PYTORCH: FORWARD4")
        temp_3 = self.fc3(temp_2)
        print(temp_3)

        # SHAPE: [K, D]
        print(temp_3.size())

        # print(output_tensor.dtype)
        # print("PYTORCH: OUTPUT=")
        # print(output_tensor)
        # output_tensor_detached = output_tensor.detach().numpy()
        # print(output_tensor_detached)
        # # print(g)
        # # print(type(g))
        # # z = list(g)
        # # print(type(z))
        # # print(z)
        # print("PYTORCH: OUTPUTTING TO CPP.")
        # return output_tensor;
        return 0


if __name__ == "__main__":
    input_dim = 5;
    L1 = 10;
    L2 = 10;
    output_dim = 3;
    net = MLP(input_dim, L1, L2, output_dim)
    print(net)
    input = torch.randn(input_dim)
    # print(input.to_numpy())
    # input = np.array([1, 2, 3, 4, 5])
    print(input)

    # print(input.size())
    output = net.forwardVector(input)
    print(dir(output))
    print(output.__dict__)
    print(output)
    print(output.detach().numpy())
    # print(input.size())
    # print(output.size())



