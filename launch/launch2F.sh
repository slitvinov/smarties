#!/bin/bash

EXECNAME=rl
NTHREADS=$1

SETTINGS+=" --gamma 0.95" #Crucial: discount factor
SETTINGS+=" --nnl1 0" #Neurons in first layer
SETTINGS+=" --nnl2 0" #Neurons in second layer
SETTINGS+=" --nnl3 0" #Neurons in first layer
SETTINGS+=" --nnm1 32"
SETTINGS+=" --nnm2 24"
SETTINGS+=" --nnm3 12"

SETTINGS+=" --nne 0.0001" #(Initial) learning rate
#SETTINGS+=" --nnl1 32" #Neurons in first layer
#SETTINGS+=" --nnl2 16" #Neurons in second layer
#SETTINGS+=" --nnm1 0"
#SETTINGS+=" --nnm2 0"

OPTIONS=${SETTINGS}${RESTART}

export OMP_NUM_THREADS=4
export OMP_PROC_BIND=true
export OMP_NESTED=true
mkdir ../run$2
mkdir ../run$2/last_sim
mkdir -p ../run$2/res
mkdir -p ../run$2/restart

cp $0 ../run$2/launch.sh
cp ../makefiles/${EXECNAME} ../run$2/exec
cp ../factory/factory2F ../run$2/factory
cp history2F.txt ../run$2/history.txt
cd ../run$2

/opt/mpich/bin/mpirun -np $1 ./exec ${OPTIONS}
#mpirun -np $1 ./exec ${OPTIONS}
