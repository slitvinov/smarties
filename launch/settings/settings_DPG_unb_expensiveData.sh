SETTINGS=
#discount factor in RL
#the closer to 1 it is, the harder it is to learn
#but, the agent might find better long-term strategies
SETTINGS+=" --gamma 0.99 --samplesFile 1"

#size of network layers
SETTINGS+=" --nnl1 32"
SETTINGS+=" --nnl2 32"

#subject to changes
SETTINGS+=" --nnType LSTM"
SETTINGS+=" --nnBPTTseq 16"
SETTINGS+=" --nnFunc SoftSign"
#SETTINGS+=" --nnFunc LRelu"
SETTINGS+=" --nnOutputFunc Tanh" #is read by agent
# Multiplies initial weights of output layer. Ie U[-.1/sqrt(f), .1/sqrt(f)]
SETTINGS+=" --outWeightsPrefac 0.1"

#whether you are training a policy or testing an already trained network
SETTINGS+=" --bTrain 1"

#RL algorithm: NAF, DPG are continuous actions, NFQ (also accepted DQN) is for discrete actions
SETTINGS+=" --learner DPG"

#Number of time steps per gradient step
SETTINGS+=" --obsPerStep 0.5"
#chance of taking random actions
SETTINGS+=" --explNoise 0.2"
#Number of samples before starting gradient steps
SETTINGS+=" --minTotObsNum 32768"
#Maximum size of the replay memory
SETTINGS+=" --maxTotObsNum 262144"
#Number of gradient steps before training ends
SETTINGS+=" --totNumSteps 5000000"

SETTINGS+=" --bSampleSequences 0"

#C in paper. Determines c_max: boundary between (used) near-policy samples and (skipped) far policy ones
SETTINGS+=" --clipImpWeight 4"
SETTINGS+=" --ERoldSeqFilter oldest"
# Here, fraction of far pol samples allowed in memory buffer
SETTINGS+=" --penalTol 0.1"

SETTINGS+=" --epsAnneal 0"
#lag of target network.
#- if >1 (ie 1000) then weights are copied every dqnT grad descent steps
#- if <1 (ie .001) then every step the target weights are updated as dqnT * w_Target + (1-dqnT)*w
SETTINGS+=" --targetDelay 0.001"
#batch size for network gradients compute
SETTINGS+=" --batchSize 64"
#network update learning rate
#SETTINGS+=" --learnrate 0.00001"
SETTINGS+=" --learnrate 0.00001"
