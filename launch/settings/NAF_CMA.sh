SETTINGS=
#file that stores every observtion (log of states and actions)
#if none then no output
SETTINGS+=" --fileSamp none" 
#discount factor in RL
SETTINGS+=" --gamma 0.9"
#network update learning rate
SETTINGS+=" --learnrate 0.0001"
#size of network layers
SETTINGS+=" --nnl1 48"
SETTINGS+=" --nnl2 24"
SETTINGS+=" --nnl3 0"
#0 means feed forward neural nets
#1 means LSTM
#subject to changes
SETTINGS+=" --nnType 0"
# L2 regularization of the weights 
SETTINGS+=" --nnL 0.00"
#chance of taking random actions
SETTINGS+=" --greedyeps 0.1"
#whether you are training a policy or testing an already trained network
SETTINGS+=" --bTrain 1"
#RL algorithm: NAF, DPG are continuous actions, NFQ (also accepted DQN) is for discrete actions
SETTINGS+=" --learn NAF"
#batch size for network gradients compute
SETTINGS+=" --dqnBatch 48"
#lag of target network. 
#if >1 then weights are copied every dqnT grad descent steps
#if <1 then every step the target weights are updated as dqnT * w_Target + (1-dqnT)*w
SETTINGS+=" --dqnT 0.001"
