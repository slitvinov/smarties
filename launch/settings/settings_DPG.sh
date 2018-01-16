SETTINGS=
#file that stores every observtion (log of states and actions)
#if none then no output
SETTINGS+=" --samplesFile none"

#discount factor in RL
#the closer to 1 it is, the harder it is to learn
#but, the agent might find better long-term strategies
SETTINGS+=" --gamma 0.99"

#size of network layers
SETTINGS+=" --nnl1 128"
SETTINGS+=" --nnl2 64"

#subject to changes
#SETTINGS+=" --nnType RNN"
#SETTINGS+=" --nnType LSTM"
SETTINGS+=" --nnFunc HardSign"
# L2 regularization of the weights
SETTINGS+=" --nnLambda 0.000001"

#whether you are training a policy or testing an already trained network
SETTINGS+=" --bTrain 1"

#RL algorithm: NAF, DPG are continuous actions, NFQ (also accepted DQN) is for discrete actions
SETTINGS+=" --learner DPG"

#maximum allowed lenth for a sequence (from first to terminal state)
#if a sequence is longer is just cut after #number of transitions
SETTINGS+=" --maxTotObsNum 65536"

#chance of taking random actions
SETTINGS+=" --greedyEps 0.1"
SETTINGS+=" --epsAnneal   100000"
SETTINGS+=" --totNumSteps 5000000"
SETTINGS+=" --obsPerStep 1"
SETTINGS+=" --bSampleSequences 0"

#lag of target network.
#- if >1 (ie 1000) then weights are copied every dqnT grad descent steps
#- if <1 (ie .001) then every step the target weights are updated as dqnT * w_Target + (1-dqnT)*w
#the first option is markedly safer
SETTINGS+=" --targetDelay 0.001"
#batch size for network gradients compute
SETTINGS+=" --batchSize 256"
#network update learning rate
SETTINGS+=" --learnrate 0.0003"
