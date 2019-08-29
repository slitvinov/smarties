import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.set_default_tensor_type(torch.DoubleTensor)

class MLP(nn.Module):
    def __init__(self, T, D, nnLayerSizes, output_dim):
        super(MLP, self).__init__()
        # Input has the form [K x T x D]
        # where K is the batch-size
        # T is the number of time-steps (POMDP)
        # D is the dimensionality (e.g. state)
        nnLayerSizes=list(nnLayerSizes)
        self.T = T
        self.D = D
        self.nnLayerSizes = nnLayerSizes
        self.output_dim = output_dim

        # Adding the input and output layer
        self.mlp_input_dim = self.T * self.D
        self.nnLayerSizes.insert(0, self.mlp_input_dim)
        self.nnLayerSizes.append(output_dim)

        # Define affine operations: y = Wx + b
        self.fcLayers = []
        for ln in range(len(nnLayerSizes)-1):
            self.fcLayers.append(nn.Linear(nnLayerSizes[ln], nnLayerSizes[ln+1]))

    def forwardCPPDict(self, input_dict):
        print("PYTHON-MLP: Input to forward:")
        print(input_dict)

        vectorMiniBatch = input_dict["vectorMiniBatch"]
        action = input_dict["action"]
        mu = input_dict["mu"]

        print("PYTHON-MLP: vectorMiniBatch = ", vectorMiniBatch)
        print("PYTHON-MLP: action = ", action)
        print("PYTHON-MLP: mu = ", mu)

        # print("|| Episodes x Time steps x Dimensionality = ", vectorMiniBatch[0].S.size(), vectorMiniBatch[0].S[0].size(), vectorMiniBatch[0].S[0][0].size())

        input_ = np.array(vectorMiniBatch[0].S)
        E, T, D = np.shape(input_)
        print("|| Episodes x Time steps x Dimensionality = ", E, T, D)

        input_red = input_[:, -self.T:, :]
        E, T, D = np.shape(input_red)
        print("|| Episodes x Time steps USED x Dimensionality = ", E, T, D)

        output = self.forwardVector(input_red)
        print("PYTHON-MLP: output = ", output)
        
        # DETACHED CANNOT BE USED IN CPP BINDINGS
        # output.detach()
        # return output
        return 0

    def forwardVector(self, input_):
        # import pybind11
        # print(pybind11.__file__)
        # print(pybind11.__version__)

        input_tensor = torch.DoubleTensor(input_)
        # SHAPE: [K, T, D]
        K, T, D = input_tensor.size()
        print("PYTHON-MLP: Input shape = ", input_tensor.size())

        assert(T==self.T)
        assert(D==self.D)

        # In MLP the time-steps do not matter
        var = input_tensor.view(-1, self.T*self.D)
        for nnLayer in self.fcLayers:
            var = F.relu(nnLayer(var))
        print("PYTHON-MLP: Output shape = ", var.size())
        # DETACHED CANNOT BE USED IN CPP BINDINGS
        # var_detached = var.detach().numpy()
        return var


if __name__ == "__main__":

    output_dim = 3
    batch_size = 7
    time_steps = 3
    input_dim = 5
    nnLayerSizes = [10, 20]

    input_ = torch.randn([batch_size, time_steps, input_dim])

    # input_ = torch.randn([1, time_steps, 5])
    # input_ = np.random.randn(1, time_steps, 5)

    net = MLP(time_steps, input_dim, nnLayerSizes, output_dim)
    print(net)

    output = net.forwardVector(input_)

    input_ = torch.DoubleTensor(input_)
    print("Input size KxTxD")
    print(input_.size())
    print("Output size KxN")
    print(output.size())
    
    # print(dir(output))
    # print(output.__dict__)
    # print(output)
    # print(output.detach().numpy())


