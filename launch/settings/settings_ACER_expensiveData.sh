SETTINGS=
#discount factor in RL
#the closer to 1 it is, the harder it is to learn
#but, the agent might find better long-term strategies
SETTINGS+=" --gamma 0.99 --samplesFile 1"
SETTINGS+=" --restart ."
#size of network layers
SETTINGS+=" --nnl1 32"
SETTINGS+=" --nnl2 16"

#subject to changes
#SETTINGS+=" --nnFunc Tanh"
SETTINGS+=" --nnType LSTM"
SETTINGS+=" --nnBPTTseq 16"
SETTINGS+=" --nnFunc SoftSign"
# Multiplies initial weights of output layer. Ie U[-.1/sqrt(f), .1/sqrt(f)]
SETTINGS+=" --outWeightsPrefac 0.1"

#whether you are training a policy or testing an already trained network
SETTINGS+=" --bTrain 1"

#variables for user-specified environment

#RL algorithm: NAF, DPG are continuous actions, NFQ (also accepted DQN) is for discrete actions
SETTINGS+=" --learner ACER"

#Number of time steps per gradient step
SETTINGS+=" --obsPerStep 0.5"
#Number of samples before starting gradient steps
#SETTINGS+=" --minTotObsNum 131072"
SETTINGS+=" --minTotObsNum 32768"
#Maximum size of the replay memory
SETTINGS+=" --maxTotObsNum 131072"
#Number of gradient steps before training ends
SETTINGS+=" --totNumSteps 5000000"

#chance of taking random actions
SETTINGS+=" --explNoise 0.2"
SETTINGS+=" --bSampleSequences 1"

SETTINGS+=" --clipImpWeight 0"
SETTINGS+=" --ERoldSeqFilter oldest"
SETTINGS+=" --penalTol 0"
SETTINGS+=" --epsAnneal 0"

#batch size for network gradients compute
SETTINGS+=" --batchSize 12"
#network update learning rate
SETTINGS+=" --learnrate 0.00001"
SETTINGS+=" --targetDelay 0.001"
