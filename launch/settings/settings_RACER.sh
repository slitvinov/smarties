SETTINGS=
#discount factor in RL
#the closer to 1 it is, the harder it is to learn
#but, the agent might find better long-term strategies
SETTINGS+=" --gamma 0.995"

#size of network layers
SETTINGS+=" --nnLayerSizes 128 128"
#SETTINGS+=" --nnl3 128"

# Activation functions:
#SETTINGS+=" --nnFunc LRelu"
SETTINGS+=" --nnFunc SoftSign"
#SETTINGS+=" --nnFunc Tanh"
# Multiplies initial weights of output layer. Ie U[-.1/sqrt(f), .1/sqrt(f)]
SETTINGS+=" --outWeightsPrefac 0.1"

#whether you are training a policy or testing an already trained network
SETTINGS+=" --bTrain 1"
#RL algorithm: NAF, DPG are continuous actions, NFQ (also accepted DQN) is for discrete actions
SETTINGS+=" --learner RACER"
#Initialization of the standard deviation for all actions
SETTINGS+=" --explNoise 0.447214"
#Number of time steps per gradient step
SETTINGS+=" --obsPerStep 1"
#Number of samples before starting gradient steps
SETTINGS+=" --minTotObsNum 524288"
#Maximum size of the replay memory
SETTINGS+=" --maxTotObsNum 524288"
#Number of gradient steps before training ends
SETTINGS+=" --totNumSteps 50000000"

#C in paper. Determines c_max: boundary between (used) near-policy samples and (skipped) far policy ones
SETTINGS+=" --clipImpWeight 4"
# Here, fraction of far pol samples allowed in memory buffer
SETTINGS+=" --penalTol 0.1"
#batch size for network gradients compute
SETTINGS+=" --batchSize 256"
#network update learning rate
SETTINGS+=" --learnrate 0.0001"
SETTINGS+=" --epsAnneal 5e-7"
