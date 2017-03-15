#!/bin/bash
#export OMP_SCHEDULE=dynamic
SOCK=$1
#export MPICH_NEMESIS_ASYNC_PROGRESS=1
#export MPICH_MAX_THREAD_SAFETY=multiple
export MYROUNDS=10000
export USEMAXTHREADS=1
#needed for thread safety:
#export MV2_ENABLE_AFFINITY=0
export LD_LIBRARY_PATH=/users/novatig/mpich-3.2/build/lib/:$LD_LIBRARY_PATH
export PATH=/users/novatig/mpich-3.2/build/bin/:$PATH

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

echo $SETTINGS > settings.txt

mpich_run -n 1 -launcher fork ./exec ${SETTINGS}
