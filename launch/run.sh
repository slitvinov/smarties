#!/bin/bash
unset LSB_AFFINITY_HOSTFILE
export MPICH_MAX_THREAD_SAFETY=multiple #MPICH
export MV2_ENABLE_AFFINITY=0 #MVAPICH
NPROCESS=$1
NTHREADS=$2
TASKPERN=$3
NMASTERS=$4

SETTINGSNAME=settings.sh
if [ ! -f $SETTINGSNAME ];then
    echo ${SETTINGSNAME}" not found! - exiting"
    exit -1
fi
source $SETTINGSNAME
SETTINGS+=" --nThreads ${NTHREADS}"
SETTINGS+=" --nMasters ${NMASTERS}"
SETTINGS+=" --ppn ${TASKPERN}"
export OMP_NUM_THREADS=${NTHREADS}
export OMP_PROC_BIND=CLOSE
export OMP_PLACES=cores
#export OMP_WAIT_POLICY=active
export OMP_MAX_TASK_PRIORITY=1
#export OMP_DISPLAY_ENV=TRUE
export OMP_DYNAMIC=FALSE

echo $SETTINGS > settings.txt
env > environment.log
echo ${NPROCESS} ${NTHREADS} $TASKPERN $NMASTERS

#mpirun -n ${NPROCESS} -ppn ${TASKPERN} -bind-to none xterm -e gdb --tui --args ./rl ${SETTINGS}

if [[ $PATH == *"openmpi"* ]]; then
#pirun -n ${NPROCESS} --map-by ppr:1:socket:pe=12 -report-bindings --mca mpi_cuda_support 0 ./rl ${SETTINGS} | tee out.log
mpirun -n ${NPROCESS} -oversubscribe --map-by node:PE=24 -report-bindings --mca mpi_cuda_support 0 ./rl ${SETTINGS} | tee out.log
else
 #--leak-check=yes  --track-origins=yes
#mpirun -n ${NPROCESS} -ppn ${TASKPERN} -bind-to core:${NTHREADS} valgrind --num-callers=100  --tool=memcheck  ./rl ${SETTINGS} | tee out.log
mpirun -n ${NPROCESS} -ppn ${TASKPERN} -bind-to core:${NTHREADS} ./rl ${SETTINGS} | tee out.log
fi


# mpirun -n ${NPROCESS} -ppn ${TASKPERN} -bind-to none xterm -hold -e gdb -ex run --args ./rl ${SETTINGS}

#mpirun -n ${NPROCESS} -ppn ${TASKPERN} -bind-to none valgrind --num-callers=100  --tool=memcheck  --leak-check=yes  --track-origins=yes --show-reachable=yes  ./rl ${SETTINGS}
