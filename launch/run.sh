#!/bin/bash
export MPICH_MAX_THREAD_SAFETY=funneled #MPICH
export MV2_ENABLE_AFFINITY=0 #MVAPICH
NPROCESS=$1
NTHREADS=$2
TASKPERN=$3

SETTINGSNAME=settings.sh
if [ ! -f $SETTINGSNAME ];then
    echo ${SETTINGSNAME}" not found! - exiting"
    exit -1
fi
source $SETTINGSNAME
SETTINGS+=" --nThreads ${NTHREADS}"
export OMP_NUM_THREADS=${NTHREADS}
echo $SETTINGS > settings.txt
env > environment.log
echo ${NPROCESS} ${NTHREADS}

#mpirun -n ${NPROCESS} -ppn ${TASKPERN} -bind-to none xterm -e gdb --tui --args ./rl ${SETTINGS}

mpirun -n ${NPROCESS} -ppn ${TASKPERN} -bind-to none ./rl ${SETTINGS} | tee out.log

# mpirun -n ${NPROCESS} -ppn ${TASKPERN} -bind-to none xterm -hold -e gdb -ex run --args ./rl ${SETTINGS}

#mpirun -n ${NPROCESS} -ppn ${TASKPERN} -bind-to none valgrind --num-callers=100  --tool=memcheck  --leak-check=yes  --track-origins=yes --show-reachable=yes  ./rl ${SETTINGS}
