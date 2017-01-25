#!/bin/bash
NPROCESS=$1
NTHREADS=$2
TASKPERN=$3
export MV2_ENABLE_AFFINITY=0
#module load gcc/4.9.2
#module load open_mpi
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
env > environment.log

mpirun -n ${NPROCESS} -ppn ${TASKPERN} -bind-to none ./exec ${SETTINGS} #--mca btl tcp,self 
#mpirun -n ${NPROCESS} -ppn ${TASKPERN} -bind-to none valgrind  --num-callers=100  --tool=memcheck  --leak-check=yes  --track-origins=yes --show-reachable=yes ./exec ${SETTINGS} #--mca btl tcp,self 
#./exec ${SETTINGS}
