#!/bin/bash

EXECNAME=rl
RUNFOLDER=$1
BOX=$2

NTHREADS=48
NSLAVESONNODE=1
module load open_mpi
module load gcc/4.9.2

BASEPATH="/cluster/scratch_xp/public/novatig/smarties/"

SETTINGS+=" --gamma 0.9" #Crucial: discount factor
SETTINGS+=" --nnm1 36"
SETTINGS+=" --nnm2 36"
SETTINGS+=" --nnm3 36"
SETTINGS+=" --greedyeps 0.0"
SETTINGS+=" --rType ${BOX}"
SETTINGS+=" --bTrain 1"
SETTINGS+=" --nne 0.00001"
SETTINGS+=" --nnL 0.0"
SETTINGS+=" --nnD 0.5"

OPTIONS=${SETTINGS}${RESTART}

export OMP_NUM_THREADS=2
export OMP_PROC_BIND=true
export OMP_NESTED=true
export OMP_WAIT_POLICY=ACTIVE
mkdir ../run$1
mkdir ../run$1/last_sim
mkdir -p ../run$1/res
mkdir -p ../run$1/restart

cp $0 ../run$1/launch.sh
cp ../makefiles/${EXECNAME} ../run$1/exec
cp ../factory/factory2FMovieSight ../run$1/factory
cp history_bigfish.txt ../run$1/history.txt
cd ../run$1

#valgrind --track-origins=yes
./exec ${OPTIONS}