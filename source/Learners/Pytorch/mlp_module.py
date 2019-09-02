import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

torch.set_default_tensor_type(torch.DoubleTensor)

PRINT_OUTPUT=False

class MLP(nn.Module):
    def __init__(self, T, D, nnLayerSizes, output_dim, sigma_dim=0):
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
        self.sigma_dim = sigma_dim

        # Adding the input and output layer
        self.mlp_input_dim = self.T * self.D
        self.nnLayerSizes.insert(0, self.mlp_input_dim)
        self.nnLayerSizes.append(output_dim)

        # Define affine operations: y = Wx + b
        self.fcLayers = []
        for ln in range(len(nnLayerSizes)-1):
            self.fcLayers.append(nn.Linear(nnLayerSizes[ln], nnLayerSizes[ln+1]))

        # Define the parameter layer for sigma
        sigma_init = 1.85
        if(sigma_dim>0):
            self.sigma_weights = Variable(torch.Tensor(sigma_init*np.array(sigma_dim*[1.0])), requires_grad=True)

    def forwardBatchId(self, input_dict):
        if(PRINT_OUTPUT): print("PYTHON-MLP: forwardBatchId")
        if(PRINT_OUTPUT): print("PYTHON-MLP: Input to forwardBatchId:")
        if(PRINT_OUTPUT): print(input_dict)

        vectorMiniBatch = input_dict["vectorMiniBatch"]
        bID = input_dict["bID"]
        t = input_dict["t"]
        fval = input_dict["fval"]

        input_ = np.array(vectorMiniBatch[0].S)
        E, T, D = np.shape(input_)
        if(PRINT_OUTPUT): print("|| Episodes x Time steps x Dimensionality = ", E, T, D)
        input_ = input_[bID, :t, :]

        input_red = input_[:, -self.T:, :]
        E, T, D = np.shape(input_red)
        if(PRINT_OUTPUT): print("|| Episodes x Time steps USED x Dimensionality = ", E, T, D)

        output, sigma = self.forwardVector(input_red)
        if(PRINT_OUTPUT): print("PYTHON-MLP: output = ", output)
        output_det = output.detach().numpy()
        sigma_det = sigma.detach().numpy()

        # COMPUTING VALUE FUNCTION
        fval = output_det[0,0]
        return 0


    def trainOnBatch(self, input_dict):
        if(PRINT_OUTPUT): print("PYTHON-MLP: trainOnBatch")
        if(PRINT_OUTPUT): print("PYTHON-MLP: Input to trainOnBatch:")
        if(PRINT_OUTPUT): print(input_dict)
        print(input_dict)

        vectorMiniBatch = input_dict["vectorMiniBatch"]

        input_ = np.array(vectorMiniBatch[0].S)
        E, T, D = np.shape(input_)
        print("|| Episodes x Time steps USED x Dimensionality = ", E, T, D)

        begTimeStep = np.array(vectorMiniBatch[0].begTimeStep)
        endTimeStep = np.array(vectorMiniBatch[0].endTimeStep)
        sampledTimeStep = np.array(vectorMiniBatch[0].sampledTimeStep)

        idx = sampledTimeStep - begTimeStep
        print(idx)




        input_red = input_[:, -self.T:, :]
        E, T, D = np.shape(input_red)
        if(PRINT_OUTPUT): print("|| Episodes x Time steps USED x Dimensionality = ", E, T, D)
        print("|| Episodes x Time steps USED x Dimensionality = ", E, T, D)

        output, sigma_ = self.forwardVector(input_red)
        if(PRINT_OUTPUT): print("PYTHON-MLP: output = ", output)
        output_det = output.detach().numpy()
        action_sigma = sigma_.detach().numpy()

        if(PRINT_OUTPUT): print("Computing action_dim")
        action_dim = (self.output_dim-1)

        if(PRINT_OUTPUT): print("Computing action")
        action_mean = output_det[:,1:1+action_dim]
        action = np.random.normal(loc=action_mean,scale=action_sigma)

        value = output_det[:,0]
        print("VALUES:")
        print(np.shape(action))
        print(np.shape(value))
        print(np.shape(action_sigma))
        mu = np.array(vectorMiniBatch[0].MU)

        mu = mu[:, -1, :]
        idx=np.shape(mu)[1]
        assert(idx%2==0)
        mu_mean = mu[:,:idx//2]
        mu_sigma = mu[:,idx//2:]
        print(np.shape(mu_mean))
        print(np.shape(mu_sigma))
        print("#########")

        CmaxRet = input_dict["CmaxRet"]
        CinvRet = input_dict["CinvRet"]

        print("SIGMAS:")
        # print(mu_sigma)
        # print(action_sigma)

        print("########## PRINTING MU ##############")
        print(mu)
        print(input_red)

        # if np.any(mu_sigma<=0) or np.any(action_sigma<=0):
        #     print("## Something is zero. ###")
        # else:
        #     print("## Something is NOT zero. ###")
        #     impWeight = self.evaluateImportanceWeight(action, action_mean, action_sigma, mu_mean, mu_sigma)
            # print("SHAPE OF impWeight 2")
            # np.shape(impWeight)
        # isOff = self.isFarPolicy(impWeight, CmaxRet, CinvRet)
        # print(isOff)

    def isFarPolicy(self, W, C, invC):
        isOff = (W>C) or (W < invC)
        # If C<=1 assume we never filter far policy samples
        return (C>1.0) and isOff

    def evaluateImportanceWeight(self, action, action_mean, action_sigma, mu_mean, mu_sigma):
        polLogProb = self.evalLogProbability(action, action_mean, action_sigma)
        behaviorLogProb = self.evalLogProbability(action, mu_mean, mu_sigma)
        impWeight = polLogProb - behaviorLogProb
        # print("SHAPE OF impWeight ")
        # print(np.shape(impWeight))
        impWeight = [7.0 if(impWeight_i>7.0) else (-7.0 if impWeight_i < -7.0 else impWeight_i) for impWeight_i in impWeight]
        impWeight = np.array(impWeight)
        return np.exp(impWeight)

    def evalLogProbability(self, var, mean, sigma):
        p=np.zeros((np.shape(mean)[0]))
        for i in range(np.shape(mean)[1]):
            precision = 1.0 / (sigma[:,i]**2)
            # print(precision)
            p -= precision * (var[:,i]-mean[:,i])**2
            p += np.log( precision / 2.0 / np.pi )
            # print(p)
        # print("SHAPE OF P ")
        # print(np.shape(p))
        # print(p)
        return 0.5 * p

    # Rvec grad;
    # if(isOff) grad = offPolCorrUpdate(S, t, O, P, thrID);
    # else grad = compute(S, t, O, P, thrID);

    # if(thrID==0)  profiler->stop_start("BCK");
    # NET.setGradient(grad, bID, t); // place gradient onto output layer

    def selectAction(self, input_dict):
        if(PRINT_OUTPUT): print("PYTHON-MLP: selectAction")
        if(PRINT_OUTPUT): print("PYTHON-MLP: Input to selectAction:")
        if(PRINT_OUTPUT): print(input_dict)

        vectorMiniBatch = input_dict["vectorMiniBatch"]
        action = input_dict["action"]
        mu = input_dict["mu"]

        if(PRINT_OUTPUT): print("PYTHON-MLP: vectorMiniBatch = ", vectorMiniBatch)
        if(PRINT_OUTPUT): print("PYTHON-MLP: action = ", action)
        if(PRINT_OUTPUT): print("PYTHON-MLP: mu = ", mu)

        input_ = np.array(vectorMiniBatch[0].S)
        E, T, D = np.shape(input_)
        if(PRINT_OUTPUT): print("|| Episodes x Time steps x Dimensionality = ", E, T, D)

        input_red = input_[:, -self.T:, :]
        E, T, D = np.shape(input_red)
        if(PRINT_OUTPUT): print("|| Episodes x Time steps USED x Dimensionality = ", E, T, D)

        output, sigma = self.forwardVector(input_red)
        if(PRINT_OUTPUT): print("PYTHON-MLP: output = ", output)
        output_det = output.detach().numpy()
        sigma_det = sigma.detach().numpy()

        if(PRINT_OUTPUT): print("Computing action_dim")
        action_dim = (self.output_dim-1)
        assert(action_dim==np.shape(np.array(action))[0])

        value = output_det[0,0]
        action_mean = output_det[0,1:1+action_dim]
        action_sigma = sigma_det[0]

        if(PRINT_OUTPUT): print("Computing action")
        action_python = np.random.normal(loc=action_mean,scale=action_sigma)

        if(PRINT_OUTPUT): print("Copying action")
        for i in range(action_dim):
            action[i]=action_python[i] 

        if(PRINT_OUTPUT): print("Copying behavior")
        for i in range(action_dim):
            mu[i]=action_python[i] 
        for j in range(action_dim):
            mu[action_dim+j]=action_sigma[i]
        # print("########## PRINTING MU ##############")
        # print(mu)

        if(PRINT_OUTPUT): print("PYTHON-MLP: action = ", action_python)
        return 0

    def forwardVector(self, input_):
        # import pybind11
        # if(PRINT_OUTPUT): print(pybind11.__file__)
        # if(PRINT_OUTPUT): print(pybind11.__version__)

        input_tensor = torch.DoubleTensor(input_)
        # SHAPE: [K, T, D]
        K, T, D = input_tensor.size()
        if(PRINT_OUTPUT): print("PYTHON-MLP: Input shape = ", input_tensor.size())

        assert(T==self.T)
        assert(D==self.D)

        # In MLP the time-steps do not matter
        var = input_tensor.view(-1, self.T*self.D)
        for nnLayer in self.fcLayers:
            var = F.relu(nnLayer(var))
        if(PRINT_OUTPUT): print("PYTHON-MLP: Output shape = ", var.size())
        # DETACHED CANNOT BE USED IN CPP BINDINGS
        # var_detached = var.detach().numpy()

        sp = nn.Softplus()
        sigma = sp(self.sigma_weights)
        sigma = sigma[None]
        sigma = torch.cat(K*[sigma])
        return var, sigma


if __name__ == "__main__":

    output_dim = 3
    sigma_dim = 2
    batch_size = 7
    time_steps = 3
    input_dim = 5
    nnLayerSizes = [10, 20]

    input_ = torch.randn([batch_size, time_steps, input_dim])

    # input_ = torch.randn([1, time_steps, 5])
    # input_ = np.random.randn(1, time_steps, 5)

    net = MLP(time_steps, input_dim, nnLayerSizes, output_dim, sigma_dim)
    if(PRINT_OUTPUT): print(net)

    output, sigma = net.forwardVector(input_)

    input_ = torch.DoubleTensor(input_)
    if(PRINT_OUTPUT): print("Input size KxTxD")
    if(PRINT_OUTPUT): print(input_.size())
    if(PRINT_OUTPUT): print("Output size KxN")
    if(PRINT_OUTPUT): print(output.size())
    if(PRINT_OUTPUT): print("Sigma size KxN")
    if(PRINT_OUTPUT): print(sigma.size())

    # if(PRINT_OUTPUT): print(dir(output))
    # if(PRINT_OUTPUT): print(output.__dict__)
    # if(PRINT_OUTPUT): print(output)
    # if(PRINT_OUTPUT): print(output.detach().numpy())


