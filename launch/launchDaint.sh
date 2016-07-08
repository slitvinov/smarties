#!/bin/bash
EXECNAME=rl
RUNFOLDER=$1
NNODES=$2
APP=$3
SETTINGSNAME=$4

MYNAME=`whoami`
BASEPATH="/scratch/daint/${MYNAME}/smarties/"
mkdir -p ${BASEPATH}${RUNFOLDER}

if [ $# -gt 4 ] ; then
    POLICY=$5
    cp $5 ${BASEPATH}${RUNFOLDER}/policy.net
fi
if [ $# -lt 7 ] ; then
    NTASK=2 #n tasks per node
    NTHREADS=8 #n threads per task
else
    NTASK=$6
    NTHREADS=$7
fi
if [ $# -lt 8 ] ; then
    WCLOCK=24:00 #chaining
else
    WCLOCK=$8
fi

#this handles app-side setup (incl. copying the factory)
source ../apps/${APP}/setup.sh ${BASEPATH}${RUNFOLDER}

cp ../makefiles/${EXECNAME} ${BASEPATH}${RUNFOLDER}/exec
cp ${SETTINGSNAME} ${BASEPATH}${RUNFOLDER}/settings.sh
cp daint_sbatch ${BASEPATH}${RUNFOLDER}/daint_sbatch
cp runDaint_learn.sh ${BASEPATH}${RUNFOLDER}/run.sh
cp $0 ${BASEPATH}${RUNFOLDER}/launch.sh

cd ${BASEPATH}${RUNFOLDER}

sbatch daint_sbatch ${RUNFOLDER} ${NNODES} ${NTASK} ${NTHREADS} ${WCLOCK}
