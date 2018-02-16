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
#SETTINGS+=" --nnl1 12"
SETTINGS+=" --nnl1 128"
SETTINGS+=" --nnl2 128"
#SETTINGS+=" --nnl3 128"

#SETTINGS+=" --nnType RNN"
#SETTINGS+=" --nnFunc PRelu"
SETTINGS+=" --nnFunc SoftSign"
#SETTINGS+=" --nnFunc HardSign"
SETTINGS+=" --nnLambda 0.0001"

#whether you are training a policy or testing an already trained network
SETTINGS+=" --bTrain 1"
#RL algorithm: NAF, DPG are continuous actions, NFQ (also accepted DQN) is for discrete actions
SETTINGS+=" --learner POAC"
#chance of taking random actions
SETTINGS+=" --greedyEps 0.5"

SETTINGS+=" --obsPerStep 1"
#SETTINGS+=" --minTotObsNum 4000" 
#SETTINGS+=" --maxTotObsNum 131072" 
SETTINGS+=" --maxTotObsNum 262144"

SETTINGS+=" --nMasters 1"
SETTINGS+=" --totNumSteps 5000000"
#SETTINGS+=" --impWeight 5"
SETTINGS+=" --impWeight 2"


#SETTINGS+=" --klDivConstraint 0.1"
SETTINGS+=" --klDivConstraint 0.2"
SETTINGS+=" --targetDelay 0"
#batch size for network gradients compute
SETTINGS+=" --batchSize 256"
SETTINGS+=" --bSampleSequences 0"
#SETTINGS+=" --batchSize 32"
#SETTINGS+=" --bSampleSequences 1"
#network update learning rate
SETTINGS+=" --learnrate 0.0001"
