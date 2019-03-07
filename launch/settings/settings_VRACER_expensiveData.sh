SETTINGS=
#discount factor in RL
#the closer to 1 it is, the harder it is to learn
#but, the agent might find better long-term strategies
SETTINGS+=" --gamma 0.99 --samplesFile 1"
SETTINGS+=" --restart ."

#size of network layers
SETTINGS+=" --nnl1 32"
SETTINGS+=" --nnl2 32"
#SETTINGS+=" --nnl3 128"

# Activation functions:
#SETTINGS+=" --nnFunc LRelu"
#SETTINGS+=" --nnType GRU"
SETTINGS+=" --nnType LSTM"
SETTINGS+=" --nnBPTTseq 16"
SETTINGS+=" --nnFunc SoftSign"
#SETTINGS+=" --nnFunc Tanh"
# Multiplies initial weights of output layer. Ie U[-.1/sqrt(f), .1/sqrt(f)]
SETTINGS+=" --outWeightsPrefac 0.1"

#whether you are training a policy or testing an already trained network
SETTINGS+=" --bTrain 1"
#RL algorithm: NAF, DPG are continuous actions, NFQ (also accepted DQN) is for discrete actions
SETTINGS+=" --learner VRACER"
#Initialization of the standard deviation for all actions
SETTINGS+=" --explNoise 0.2"
#Number of time steps per gradient step
SETTINGS+=" --obsPerStep 0.5"
#Number of samples before starting gradient steps
SETTINGS+=" --minTotObsNum 16384"
#SETTINGS+=" --minTotObsNum 512"
#Maximum size of the replay memory
#SETTINGS+=" --minTotObsNum 16384"
SETTINGS+=" --maxTotObsNum 262144"
#Number of gradient steps before training ends
SETTINGS+=" --totNumSteps 5000000"

#C in paper. Determines c_max: boundary between (used) near-policy samples and (skipped) far policy ones
SETTINGS+=" --clipImpWeight 4"
SETTINGS+=" --ERoldSeqFilter oldest"

# Here, fraction of far pol samples allowed in memory buffer
SETTINGS+=" --penalTol 0.1"

# Annealing factor for impWeight and learn rate -> 1/(1+epsAnneal*fac)
SETTINGS+=" --epsAnneal 5e-7"
SETTINGS+=" --targetDelay 0.00"
#batch size for network gradients compute
SETTINGS+=" --batchSize 128"
#network update learning rate
SETTINGS+=" --learnrate 0.0001"
