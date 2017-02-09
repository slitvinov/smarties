#!/bin/bash
EXECNAME=rl
RUNFOLDER=$1
NNODES=$2
APP=$3
SETTINGSNAME=$4

MYNAME=`whoami`
BASEPATH="/cluster/scratch/${MYNAME}/smarties/"
#BASEPATH="../"
mkdir -p ${BASEPATH}${RUNFOLDER}
#lfs setstripe -c 1 ${BASEPATH}${RUNFOLDER}

if [ $# -gt 4 ] ; then
    POLICY=$5
    cp ${POLICY} ${BASEPATH}${RUNFOLDER}/policy.net
fi
if [ $# -lt 7 ] ; then
    NTASK=2 #n tasks per node
    NTHREADS=12 #n threads per task
else
    NTASK=$6
    NTHREADS=$7
fi
if [ $# -lt 8 ] ; then
    WCLOCK=24:00
else
    WCLOCK=$8
fi
if [ $# -lt 9 ] ; then
    TIMES=1 #chaining
else
    TIMES=$9
fi
NTHREADSPERNODE=12
NPROCESS=$((${NNODES}*${NTASK}))
NPROCESSORS=$((${NNODES}*${NTHREADSPERNODE}))

#this must handle all app-side setup (as well as copying the factory)
source ../apps/${APP}/setup.sh ${BASEPATH}${RUNFOLDER}

cp ../makefiles/${EXECNAME} ${BASEPATH}${RUNFOLDER}/exec
cp ${SETTINGSNAME} ${BASEPATH}${RUNFOLDER}/settings.sh
cp ${SETTINGSNAME} ${BASEPATH}${RUNFOLDER}/policy_settings.sh
cp runEuler_learn.sh ${BASEPATH}${RUNFOLDER}/run.sh
cp $0 ${BASEPATH}${RUNFOLDER}/launch.sh

cd ${BASEPATH}${RUNFOLDER}

./run.sh ${NPROCESS} ${NTHREADS} ${NTASK}
#bsub -J ${RUNFOLDER} -n ${NPROCESSORS} -R span[ptile=48] -sp 100 -W ${WCLOCK} ./run.sh  ${NPROCESS} ${NTHREADS} ${NTASK}

for (( c=1; c<=${TIMES}-1; c++ ))
do
    bsub -J ${RUNFOLDER} -n ${NPROCESSORS} -R span[ptile=48] -sp 100 -w "ended(${RUNFOLDER})" -W ${WCLOCK} ./run.sh  ${NPROCESS} ${NTHREADS} ${NTASK}
done

# NTHREADSPERNODE=24
# NPROCESS=$((${NNODES}*${NTASK}))
# NPROCESSORS=$((${NNODES}*${NTHREADSPERNODE}))
