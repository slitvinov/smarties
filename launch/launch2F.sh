#!/bin/bash

EXECNAME=rl
NTHREADS=$1


OPTIONS=${SETTINGS}${RESTART}

export OMP_NUM_THREADS=8
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

#/opt/mpich/bin/mpirun -np $1 ./exec ${OPTIONS}
mpirun -np $1 ./exec ${OPTIONS}
