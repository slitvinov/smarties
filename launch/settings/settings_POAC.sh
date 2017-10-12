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
#SETTINGS+=" --nnl1 32"
#SETTINGS+=" --nnl2 32"
#SETTINGS+=" --nnl1 64"
#SETTINGS+=" --nnl2 64"
SETTINGS+=" --nnl1 128"
SETTINGS+=" --nnl2 128"
#SETTINGS+=" --nnl3 128"
#SETTINGS+=" --nnl1 256"
#SETTINGS+=" --nnl2 256"

#subject to changes
#SETTINGS+=" --nnType RNN"
#SETTINGS+=" --nnFunc PRelu"
#SETTINGS+=" --nnFunc Tanh"
#SETTINGS+=" --nnFunc SoftSign"
SETTINGS+=" --nnFunc HardSign"
# L2 regularization of the weights
#SETTINGS+=" --nnLambda 0.0001"

#whether you are training a policy or testing an already trained network
SETTINGS+=" --bTrain 1"

#RL algorithm: NAF, DPG are continuous actions, NFQ (also accepted DQN) is for discrete actions
SETTINGS+=" --learner POAC"

#maximum allowed lenth for a sequence (from first to terminal state)
#if a sequence is longer is just cut after #number of transitions
SETTINGS+=" --maxTotSeqNum 100"

#chance of taking random actions
SETTINGS+=" --greedyEps 1"
SETTINGS+=" --epsAnneal 1000"
#SETTINGS+=" --obsPerStep 6.4"
SETTINGS+=" --obsPerStep 1"
#SETTINGS+=" --obsPerStep 0.01"
#SETTINGS+=" --bSampleSequences 1"
SETTINGS+=" --nMasters 1"
SETTINGS+=" --totNumSteps 1000000"

#lag of target network.
#- if >1 (ie 1000) then weights are copied every dqnT grad descent steps
#- if <1 (ie .001) then every step the target weights are updated as dqnT * w_Target + (1-dqnT)*w
#the first option is markedly safer
SETTINGS+=" --klDivConstraint 0.01"
SETTINGS+=" --targetDelay 0.01"
#batch size for network gradients compute
#SETTINGS+=" --batchSize 256"
#SETTINGS+=" --batchSize 32"
SETTINGS+=" --batchSize 128"
#network update learning rate
SETTINGS+=" --learnrate 0.0001"
