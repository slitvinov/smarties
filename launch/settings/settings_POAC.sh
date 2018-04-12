SETTINGS=
#discount factor in RL
#the closer to 1 it is, the harder it is to learn
#but, the agent might find better long-term strategies
SETTINGS+=" --gamma 0.995"

#size of network layers
SETTINGS+=" --nnl1 128"
SETTINGS+=" --nnl2 128"
#SETTINGS+=" --nnl3 128"
#SETTINGS+=" --nnl2 96"
#SETTINGS+=" --nnl3 64"

#SETTINGS+=" --nnType RNN"
#SETTINGS+=" --nnFunc LRelu"
SETTINGS+=" --nnFunc SoftSign"
#SETTINGS+=" --nnFunc Tanh"

#whether you are training a policy or testing an already trained network
SETTINGS+=" --bTrain 1"
#RL algorithm: NAF, DPG are continuous actions, NFQ (also accepted DQN) is for discrete actions
SETTINGS+=" --learner RACER"
#chance of taking random actions
SETTINGS+=" --greedyEps 0.5"

SETTINGS+=" --obsPerStep 1"
SETTINGS+=" --minTotObsNum 131072"
SETTINGS+=" --maxTotObsNum 524288"

SETTINGS+=" --totNumSteps 5000000"
SETTINGS+=" --impWeight 4"


SETTINGS+=" --klDivConstraint 0.1"
#SETTINGS+=" --outWeightsPrefac 0.0001"
SETTINGS+=" --targetDelay 0"
#batch size for network gradients compute
SETTINGS+=" --batchSize 256"
SETTINGS+=" --bSampleSequences 0"
#network update learning rate
SETTINGS+=" --learnrate 0.0001"
