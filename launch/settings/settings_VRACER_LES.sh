SETTINGS=
#discount factor in RL
#the closer to 1 it is, the harder it is to learn
#but, the agent might find better long-term strategies
SETTINGS+=" --gamma 0.99"

#size of network layers
SETTINGS+=" --nnLayerSizes 64 64"
#SETTINGS+=" --nnLayerSizes 128 128"

SETTINGS+=" --nnFunc SoftSign"
SETTINGS+=" --outWeightsPrefac 0.1"

#whether you are training a policy or testing an already trained network
SETTINGS+=" --bTrain 1"
#RL algorithm: NAF, DPG are continuous actions, NFQ (also accepted DQN) is for discrete actions
SETTINGS+=" --learner VRACER"
#chance of taking random actions
SETTINGS+=" --explNoise 0.1" # avoid negatives

SETTINGS+=" --obsPerStep 1"
SETTINGS+=" --minTotObsNum 524288"
SETTINGS+=" --maxTotObsNum 524288"

SETTINGS+=" --totNumSteps 5000000"
SETTINGS+=" --clipImpWeight 4" # strict clipping, was 4, bcz small action space and small RM

SETTINGS+=" --penalTol 0.1"  # parameter D of refer
SETTINGS+=" --epsAnneal 0" # no annealing of the learning rate
SETTINGS+=" --targetDelay 0"
#batch size for network gradients compute
SETTINGS+=" --batchSize 256" # we can go to  64 if needed
SETTINGS+=" --bSampleSequences 0"
SETTINGS+=" --ERoldSeqFilter oldest"
#SETTINGS+=" --dataSamplingAlgo PERseq"
#network update learning rate
SETTINGS+=" --learnrate 0.0001" # small net, small learn rate
