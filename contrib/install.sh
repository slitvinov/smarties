#!/bin/sh

: ${PREFIX=$HOME/.local}
me=contirb/install.sh

err () {
    printf >&2 %s\\n "$me: $*"
    exit 2
}

# find source/smarties -name '*.h' | sort
H='
include/smarties_extern.h
include/smarties.h
source/smarties/Communicator.h
source/smarties/Core/Agent.h
source/smarties/Core/Environment.h
source/smarties/Core/Launcher.h
source/smarties/Core/Master.h
source/smarties/Core/StateAction.h
source/smarties/Core/Worker.h
source/smarties/Engine.h
source/smarties/Learners/ACER.h
source/smarties/Learners/AlgoFactory.h
source/smarties/Learners/CMALearner.h
source/smarties/Learners/DPG.h
source/smarties/Learners/DQN.h
source/smarties/Learners/Learner_approximator.h
source/smarties/Learners/Learner.h
source/smarties/Learners/Learner_pytorch.h
source/smarties/Learners/MixedPG.h
source/smarties/Learners/NAF.h
source/smarties/Learners/PPO.h
source/smarties/Learners/RACER.h
source/smarties/Math/Continuous_policy.h
source/smarties/Math/Discrete_advantage.h
source/smarties/Math/Discrete_policy.h
source/smarties/Math/Gaus_advantage.h
source/smarties/Math/Quadratic_advantage.h
source/smarties/Math/Quadratic_term.h
source/smarties/Math/Zero_advantage.h
source/smarties/Network/Approximator.h
source/smarties/Network/Builder.h
source/smarties/Network/CMA_Optimizer.h
source/smarties/Network/Conv2Dfactory.h
source/smarties/Network/Layers/Activation.h
source/smarties/Network/Layers/Functions.h
source/smarties/Network/Layers/Layer_Base.h
source/smarties/Network/Layers/Layer_Conv2D.h
source/smarties/Network/Layers/Layer_GRU.h
source/smarties/Network/Layers/Layer_LSTM.h
source/smarties/Network/Layers/Layers.h
source/smarties/Network/Layers/Parameters.h
source/smarties/Network/Network.h
source/smarties/Network/Optimizer.h
source/smarties/Network/ThreadContext.h
source/smarties/ReplayMemory/DataCoordinator.h
source/smarties/ReplayMemory/Episode.h
source/smarties/ReplayMemory/ExperienceRemovalAlgorithms.h
source/smarties/ReplayMemory/MemoryBuffer.h
source/smarties/ReplayMemory/MemoryProcessing.h
source/smarties/ReplayMemory/MiniBatch.h
source/smarties/ReplayMemory/ReplayStatsCounters.h
source/smarties/ReplayMemory/Sampling.h
source/smarties/Settings/Bund.h
source/smarties/Settings/Definitions.h
source/smarties/Settings/ExecutionInfo.h
source/smarties/Settings/HyperParameters.h
source/smarties/Utils/DelayedReductor.h
source/smarties/Utils/FunctionUtilities.h
source/smarties/Utils/LauncherUtilities.h
source/smarties/Utils/MPIUtilities.h
source/smarties/Utils/ParameterBlob.h
source/smarties/Utils/Profiler.h
source/smarties/Utils/SocketsLib.h
source/smarties/Utils/SstreamUtilities.h
source/smarties/Utils/StatsTracker.h
source/smarties/Utils/TaskQueue.h
source/smarties/Utils/ThreadSafeVec.h
source/smarties/Utils/Warnings.h
'
# find include source/smarties -name '*.h' | xargs -n1 dirname | sort | uniq
D='
include
source/smarties
source/smarties/Core
source/smarties/Learners
source/smarties/Math
source/smarties/Network
source/smarties/Network/Layers
source/smarties/ReplayMemory
source/smarties/Settings
source/smarties/Utils
'
mkdir -p build
(cd build
 cmake .. || err 'cmake failed'
 make || err 'make failed'
) || exit 2

mkdir -p "$PREFIX"/include "$PREFIX"/lib
cp include/smarties.f90 "$PREFIX"/include/smarties.f90 || err 'fail to f90 interface'
cp lib/libsmarties.so "$PREFIX"/lib/libsmarties.so || err 'fail to install library'

for i in $D
do mkdir -p "$PREFIX"/$i
done
for i in $H
do cp $i "$PREFIX"/$i
done
