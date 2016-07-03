#!/bin/bash
NTHREADS=$1

module load gcc/4.9.2
export OMP_NUM_THREADS=${NTHREADS}
#export OMP_PROC_BIND=true
#export OMP_NESTED=true
#export OMP_WAIT_POLICY=ACTIVE

SETTINGSNAME=settings.sh
if [ ! -f $SETTINGSNAME ];then
    echo ${SETTINGSNAME}" not found! - exiting"
    exit -1
fi
source $SETTINGSNAME
SETTINGS+=" --nThreads ${NTHREADS}"
echo $SETTINGS > settings.txt

#./exec ${SETTINGS}
valgrind  --num-callers=100  --tool=memcheck  --leak-check=yes  --track-origins=yes ./exec ${SETTINGS}
