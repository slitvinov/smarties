SETTINGS=
#file that stores every observtion (log of states and actions)
#if none then no output
SETTINGS+=" --samplesFile none"
#SETTINGS+=" --restart none"

#discount factor in RL
#the closer to 1 it is, the harder it is to learn
#but, the agent might find better long-term strategies
SETTINGS+=" --gamma 0.995"
#size of network layers
SETTINGS+=" --nnl1 64"
SETTINGS+=" --nnl2 64"
#SETTINGS+=" --nnl1 128"
#SETTINGS+=" --nnl2 128"

#subject to changes
#SETTINGS+=" --nnType RNN"
#SETTINGS+=" --nnFunc Tanh"
#SETTINGS+=" --nnFunc SoftSign"
SETTINGS+=" --nnFunc LRelu"
# L2 regularization of the weights
SETTINGS+=" --nnLambda 0.0001"

#whether you are training a policy or testing an already trained network
SETTINGS+=" --bTrain 1"

#variables for user-specified environment

#RL algorithm: NAF, DPG are continuous actions, NFQ (also accepted DQN) is for discrete actions
SETTINGS+=" --learner GAE"

#chance of taking random actions
SETTINGS+=" --greedyEps 0.5"
SETTINGS+=" --klDivConstraint 0.01"
SETTINGS+=" --epsAnneal 1000"
SETTINGS+=" --totNumSteps 10000000"
SETTINGS+=" --lambda 0.97"
SETTINGS+=" --impWeight 0.2"

#SETTINGS+=" --maxTotObsNum 16384"
SETTINGS+=" --maxTotObsNum 2048"
SETTINGS+=" --obsPerStep 6.4" # equivalent to 10 epoch with BS 64
# 4096 / 64 * 10 = 640 steps per epoch
# 4096 / 640 = 6.4 obs per step

#batch size for network gradients compute
SETTINGS+=" --batchSize 64"
#network update learning rate
SETTINGS+=" --learnrate 0.0001"
