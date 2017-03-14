#!/bin/bash
NTHREADS=$1
NPROCESS=$2

export OMP_NUM_THREADS=${NTHREADS}
export LD_LIBRARY_PATH=/opt/mpich/lib/:$LD_LIBRARY_PATH

SETTINGSNAME=settings.sh
if [ ! -f $SETTINGSNAME ];then
    echo ${SETTINGSNAME}" not found! - exiting"
    exit -1
fi
source $SETTINGSNAME
SETTINGS+=" --nThreads ${NTHREADS}"
echo $SETTINGS > settings.txt

mpirun -n ${NPROCESS}  ./exec ${SETTINGS}
#mpirun -n ${NPROCESS} valgrind  --num-callers=100  --tool=memcheck  --leak-check=yes  --track-origins=yes --show-reachable=yes ./exec ${SETTINGS}
