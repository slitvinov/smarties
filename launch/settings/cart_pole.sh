SETTINGS=
#file that stores every observtion (log of states and actions)
#if none then no output
SETTINGS+=" --samplesFile none"
SETTINGS+=" --restart none"

#discount factor in RL
#the closer to 1 it is, the harder it is to learn
#but, the agent might find better long-term strategies
SETTINGS+=" --gamma 0.99"

#size of network layers
SETTINGS+=" --nnl1 32"
SETTINGS+=" --nnl2 64"
SETTINGS+=" --nnl3 128"

#subject to changes
SETTINGS+=" --nnType FFNN"
#SETTINGS+=" --nnType RNN"
#SETTINGS+=" --nnType LSTM"
SETTINGS+=" --nnFunc PRelu"
# L2 regularization of the weights
SETTINGS+=" --nnLambda 0.0000"

#whether you are training a policy or testing an already trained network
SETTINGS+=" --bTrain 1"

#variables for user-specified environment
SETTINGS+=" --rType 0"
SETTINGS+=" --senses 0"

#RL algorithm: NAF, DPG are continuous actions, NFQ (also accepted DQN) is for discrete actions
#SETTINGS+=" --learner DQN"
#SETTINGS+=" --learner NAF"
SETTINGS+=" --learner DACER"
#SETTINGS+=" --learner NAF"
#SETTINGS+=" --learner RACER"
#SETTINGS+=" --learner DPG"

#number of state vectors received from env to be chained together to form input to net (faux RNN?)
SETTINGS+=" --appendedObs 0"
SETTINGS+=" --splitLayers 0"

#maximum allowed lenth for a sequence (from first to terminal state)
#if a sequence is longer is just cut after #number of transitions
SETTINGS+=" --maxTotSeqNum 5000"

#chance of taking random actions
#SETTINGS+=" --greedyEps 0.5"
SETTINGS+=" --greedyEps 0.1"
SETTINGS+=" --epsAnneal 10000"

#lag of target network.
#- if >1 (ie 1000) then weights are copied every dqnT grad descent steps
#- if <1 (ie .001) then every step the target weights are updated as dqnT * w_Target + (1-dqnT)*w
#the first option is markedly safer
SETTINGS+=" --targetDelay 0.01" #for R(/D) ACER
#SETTINGS+=" --targetDelay 0.001" #for any other algo
#batch size for network gradients compute
SETTINGS+=" --batchSize 24" #for R(/D) ACER or any algo with RNN
# SETTINGS+=" --batchSize 64" #for any other algo with FFNN
#network update learning rate
SETTINGS+=" --learnrate 0.0002"
