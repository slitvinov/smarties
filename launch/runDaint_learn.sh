#!/bin/bash
NNODES=$1
NTASK=$2
NTHREADS=$3

NPROCESS=$((${NNODES}*${NTASK}))

SETTINGSNAME=settings.sh
if [ ! -f $SETTINGSNAME ];then
    echo ${SETTINGSNAME}" not found! - exiting"
    exit -1
fi
source $SETTINGSNAME
SETTINGS+=" --nThreads ${NTHREADS}"
echo $SETTINGS > settings.txt
#echo  ${SETTINGS}
srun -n ${NPROCESS} --cpu_bind=none --ntasks-per-node=${NTASK} --cpus-per-task=${NTHREADS} ./exec ${SETTINGS}
