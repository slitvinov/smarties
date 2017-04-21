SETTINGS=
#file that stores every observtion (log of states and actions)
#if none then no output
SETTINGS+=" --fileSamp none" 

#discount factor in RL
#the closer to 1 it is, the harder it is to learn
#but, the agent might find better long-term strategies
SETTINGS+=" --gamma 0.99"

#network update learning rate
SETTINGS+=" --learnrate 0.001"

#size of network layers
SETTINGS+=" --nnl1 64"
SETTINGS+=" --nnl2 64"
SETTINGS+=" --nnl3 32"
SETTINGS+=" --nnl4 16"

#0 means feed forward neural nets
#1 means LSTM
#subject to changes
SETTINGS+=" --nnType 0"

# L2 regularization of the weights
SETTINGS+=" --nnL 0.0"

#chance of taking random actions
SETTINGS+=" --greedyeps 0.1"

#whether you are training a policy or testing an already trained network
SETTINGS+=" --bTrain 1"

#variables for user-specified environment
SETTINGS+=" --rType 0"
SETTINGS+=" --senses 0"

#RL algorithm: NAF, DPG are continuous actions, NFQ (also accepted DQN) is for discrete actions
#SETTINGS+=" --learn DQN"
#SETTINGS+=" --learn NAF"
SETTINGS+=" --learn ACER"
#SETTINGS+=" --learn CACER"
#SETTINGS+=" --learn CRACER"
#SETTINGS+=" --learn RACER"
#SETTINGS+=" --learn DPG"

#number of state vectors received from env to be chained together to form input to net (faux RNN?)
SETTINGS+=" --dqnNs 0"

#maximum allowed lenth for a sequence (from first to terminal state)
#if a sequence is longer is just cut after #number of transitions
SETTINGS+=" --dqnSeqMax 1000"
SETTINGS+=" --maxTotSeqNum 1000"

#batch size for network gradients compute
SETTINGS+=" --dqnBatch 32"

SETTINGS+=" --epsAnneal 10000" 
#lag of target network.
#- if >1 (ie 1000) then weights are copied every dqnT grad descent steps
#- if <1 (ie .001) then every step the target weights are updated as dqnT * w_Target + (1-dqnT)*w
#the first option is markedly safer
#SETTINGS+=" --dqnT 1000"
SETTINGS+=" --dqnT 0.01"
