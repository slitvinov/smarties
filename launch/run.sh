#!/bin/bash
export MPICH_MAX_THREAD_SAFETY=funneled
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
echo $SETTINGS > settings.txt
env > environment.log
echo ${NPROCESS} ${NTHREADS}


mpirun -n ${NPROCESS} -ppn ${TASKPERN} -bind-to none ./exec ${SETTINGS}

