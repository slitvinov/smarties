#!/bin/bash
SOCK=$1

export LD_LIBRARY_PATH=/opt/mpich/lib/:$LD_LIBRARY_PATH

SETTINGSNAME=settings.sh
if [ ! -f $SETTINGSNAME ];then
    echo ${SETTINGSNAME}" not found! - exiting"
    exit -1
fi
source $SETTINGSNAME
SETTINGS+=" --nThreads 1"
SETTINGS+=" --isServer 0"
SETTINGS+=" --sockPrefix ${SOCK}"

echo $SETTINGS > settings.txt
env > environment.log


#module load valgrind
#mpirun -n 1 -launcher fork  valgrind  --tool=memcheck  --leak-check=full --show-reachable=no --show-possibly-lost=no --track-origins=yes ./exec ${SETTINGS}
mpirun -n 1 -launcher fork ./exec ${SETTINGS}

#mpich_run -n ${NPROCESS} -ppn ${TASKPERN} -bind-to none -launcher ssh -f lsf_hostfile valgrind  --tool=memcheck  --leak-check=full --show-reachable=no --show-possibly-lost=no --track-origins=yes ./exec ${SETTINGS}
