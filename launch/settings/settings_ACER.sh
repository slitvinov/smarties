SETTINGS=
#file that stores every observtion (log of states and actions)
#if none then no output
SETTINGS+=" --samplesFile none"
#SETTINGS+=" --restart none"

#discount factor in RL
#the closer to 1 it is, the harder it is to learn
#but, the agent might find better long-term strategies
SETTINGS+=" --gamma 0.99"
#size of network layers
SETTINGS+=" --nnl1 128"
SETTINGS+=" --nnl2 64"

#subject to changes
#SETTINGS+=" --nnFunc Tanh"
SETTINGS+=" --nnFunc SoftSign"
# L2 regularization of the weights
SETTINGS+=" --nnLambda 0.000003"

#whether you are training a policy or testing an already trained network
SETTINGS+=" --bTrain 1"

#variables for user-specified environment

#RL algorithm: NAF, DPG are continuous actions, NFQ (also accepted DQN) is for discrete actions
SETTINGS+=" --learner ACER"

#chance of taking random actions
SETTINGS+=" --greedyEps 0.5"
SETTINGS+=" --epsAnneal 1000"
SETTINGS+=" --nMasters 1"
SETTINGS+=" --bSampleSequences 1"
#SETTINGS+=" --maxTotObsNum 262144"
#SETTINGS+=" --maxTotObsNum 131072"
SETTINGS+=" --maxTotObsNum 65536"
SETTINGS+=" --obsPerStep 1.0"

SETTINGS+=" --totNumSteps 6400000"

#batch size for network gradients compute
SETTINGS+=" --batchSize 32"
#network update learning rate
SETTINGS+=" --learnrate 0.0003"
SETTINGS+=" --targetDelay 0.005"
