SETTINGS=
#discount factor in RL
#the closer to 1 it is, the harder it is to learn
#but, the agent might find better long-term strategies
SETTINGS+=" --gamma 0.99"
#size of network layers
SETTINGS+=" --nnLayerSizes 64 64"

#subject to changes
#SETTINGS+=" --nnType RNN"
#SETTINGS+=" --nnFunc Tanh"
SETTINGS+=" --nnFunc SoftSign"
#SETTINGS+=" --nnFunc LRelu"

#whether you are training a policy or testing an already trained network
SETTINGS+=" --bTrain 1"

#variables for user-specified environment

#RL algorithm: NAF, DPG are continuous actions, NFQ (also accepted DQN) is for discrete actions
SETTINGS+=" --learner CMA"

#chance of taking random actions
SETTINGS+=" --totNumSteps 10000000"

#SETTINGS+=" --maxTotObsNum 16384"
SETTINGS+=" --maxTotObsNum 64000"

#batch size for network gradients compute
#SETTINGS+=" --batchSize 120"
SETTINGS+=" --batchSize 10"
SETTINGS+=" --ESpopSize 12"
SETTINGS+=" --explNoise 0.1"
#network update learning rate
SETTINGS+=" --learnrate 0.01"
