#!/bin/bash
EXECNAME=rl
RUNFOLDER=$1
NNODES=$2
APP=$3
SETTINGSNAME=$4
BASEPATH="../"

if [ $# -gt 4 ] ; then
    POLICY=$5
    cp $5 ${BASEPATH}${RUNFOLDER}/policy.net
fi
if [ $# -lt 7 ] ; then
    NTASK=4 #n tasks per node
    NTHREADS=12 #n threads per task
else
    NTASK=$6
    NTHREADS=$7
fi
if [ $# -lt 8 ] ; then
    WCLOCK=12:00
else
    WCLOCK=$8
fi
if [ $# -lt 9 ] ; then
    TIMES=1 #chaining
else
    TIMES=$9
fi

NPROCESS=${NNODES}
NPROCESSORS=${NNODES}

mkdir -p ${BASEPATH}${RUNFOLDER}

#this must handle all app-side setup (as well as copying the factory)
source ../apps/${APP}/setup.sh ${BASEPATH}${RUNFOLDER}

cp ../makefiles/${EXECNAME} ${BASEPATH}${RUNFOLDER}/exec
cp ${SETTINGSNAME} ${BASEPATH}${RUNFOLDER}/settings.sh
cp runBrutus_train.sh ${BASEPATH}${RUNFOLDER}/run.sh
cp $0 ${BASEPATH}${RUNFOLDER}/launch.sh

cd ${BASEPATH}${RUNFOLDER}

#bsub -J ${RUNFOLDER} -n ${NPROCESS} -R span[ptile=48] -sp 100 -W ${WCLOCK} ./run.sh ${NPROCESS}
./run.sh ${NPROCESS}

for (( c=1; c<=${TIMES}-1; c++ ))
do
    bsub -J ${RUNFOLDER} -n ${NPROCESS} -R span[ptile=48] -sp 100 -w "ended(${RUNFOLDER})" -W ${WCLOCK} ./run.sh ${NPROCESS}
done

