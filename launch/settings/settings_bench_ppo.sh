
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
SETTINGS+=" --nnFunc SoftSign"
# L2 regularization of the weights
SETTINGS+=" --nnLambda 0.0000003"

#whether you are training a policy or testing an already trained network
SETTINGS+=" --bTrain 1"

#RL algorithm: NAF, DPG are continuous actions, NFQ (also accepted DQN) is for discrete actions
SETTINGS+=" --learner GAE"

#chance of taking random actions
SETTINGS+=" --greedyEps 0.5"
# same as 2048 * 8 agents extrapolated from paper
SETTINGS+=" --maxTotObsNum 16384"
# same as 10 epochs with batch size 64
SETTINGS+=" --obsPerStep 6.4"
# unused:
SETTINGS+=" --totNumSteps 1000000"
SETTINGS+=" --lambda 0.95"

SETTINGS+=" --klDivConstraint 0.01"
#batch size for network gradients compute
SETTINGS+=" --batchSize 64"
#network update learning rate
SETTINGS+=" --learnrate 0.0003"
