#!/bin/bash
unset LSB_AFFINITY_HOSTFILE
export MPICH_MAX_THREAD_SAFETY=serialized #MPICH
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
#echo ${NPROCESS} ${NTHREADS}

#mpirun -n ${NPROCESS} -ppn ${TASKPERN} -bind-to none xterm -e gdb --tui --args ./rl ${SETTINGS}

if [[ $PATH == *"openmpi"* ]]; then
mpirun -n ${NPROCESS} --map-by ppr:1:socket:pe=12 -report-bindings --mca mpi_cuda_support 0 ./rl ${SETTINGS} | tee out.log
else
mpirun -n ${NPROCESS} -ppn ${TASKPERN} -bind-to none ./rl ${SETTINGS} | tee out.log
fi


# mpirun -n ${NPROCESS} -ppn ${TASKPERN} -bind-to none xterm -hold -e gdb -ex run --args ./rl ${SETTINGS}

#mpirun -n ${NPROCESS} -ppn ${TASKPERN} -bind-to none valgrind --num-callers=100  --tool=memcheck  --leak-check=yes  --track-origins=yes --show-reachable=yes  ./rl ${SETTINGS}
