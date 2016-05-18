#!/bin/bash

EXECNAME=rl

SETTINGS+=" --gamma 0.9" #Crucial: discount factor
SETTINGS+=" --nnl1 0" #Neurons in first layer
SETTINGS+=" --nnl2 0" #Neurons in second layer
SETTINGS+=" --nnl3 0" #Neurons in first layer
SETTINGS+=" --nnm1 24"
SETTINGS+=" --nnm2 16"
SETTINGS+=" --nnm3 12"

SETTINGS+=" --nne 0.00001" #(Initial) learning rate
#SETTINGS+=" --nnl1 32" #Neurons in first layer
#SETTINGS+=" --nnl2 16" #Neurons in second layer
#SETTINGS+=" --nnm1 0"
#SETTINGS+=" --nnm2 0"

OPTIONS=${SETTINGS}${RESTART}

#export OMP_NUM_THREADS=4
#export OMP_PROC_BIND=true
#export OMP_NESTED=true
mkdir ../run$1
mkdir ../run$1/last_sim
mkdir -p ../run$1/res
mkdir -p ../run$1/restart

cp $0 ../run$1/launch.sh
cp ../makefiles/${EXECNAME} ../run$1/exec
cp ../factory/factory2FMovieSight ../run$1/factory
cp obs_final_final_modified.txt ../run$1/history.txt
cd ../run$1

#valgrind --track-origins=yes
./exec ${OPTIONS}
#mpirun -np $1 ./exec ${OPTIONS}
