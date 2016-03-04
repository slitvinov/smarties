#!/bin/bash

EXECNAME=rl
NTHREADS=$1

SETTINGS+=" --gamma 0.95" #Crucial: discount factor
SETTINGS+=" --nnl1 0" #Neurons in first layer
SETTINGS+=" --nnl2 0" #Neurons in second layer
SETTINGS+=" --nnl3 0" #Neurons in first layer
SETTINGS+=" --nnm1 32"
SETTINGS+=" --nnm2 16"
SETTINGS+=" --nnm3 0"
SETTINGS+=" --nne 0.001" #(Initial) learning rate

RESTART=" --restart res/policy"

RESTARTPOLICY=" -restartPolicy 1"

OPTIONS=${SETTINGS}${RESTART}

export OMP_NUM_THREADS=4
export OMP_PROC_BIND=true
export OMP_NESTED=true

mkdir -p ../run$2
mkdir -p ../run$2/last_sim

if [ "${RESTARTPOLICY}" = " -restartPolicy 1" ]; then
echo "---- launch.sh >> Restart Policy ----"
mkdir -p ../run$2/res
#cp ../launch/policy* ../run$2/res/
#    cp ../factory/policy* ${BASEPATH}${RUNFOLDER}/
fi

cp ../makefiles/${EXECNAME} ../run$2/executable
cp ../factory/factoryCart ../run$2/factory
cp ../apps/cart-pole ../run$2/
cd ../run$2

aprun -n $1 ./executable ${OPTIONS}
#mpirun -mca btl tcp,sm,self -np $1 ./executable ${OPTIONS}
#/opt/mpich/bin/mpirun -np $1 valgrind --leak-check=yes ./executable ${OPTIONS}
