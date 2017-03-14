#!/bin/bash
EXECNAME=rl
RUNFOLDER=$1
NTHREADS=$2 # TEMP: here used as nthreads
NTASK=$3
APP=$4
SETTINGSNAME=$5

BASEPATH="../runs/"
mkdir -p ${BASEPATH}${RUNFOLDER}
#lfs setstripe -c 1 ${BASEPATH}${RUNFOLDER}

if [ $# -gt 5 ] ; then
    POLICY=$6
    cp ${POLICY} ${BASEPATH}${RUNFOLDER}/policy.net
fi

#this must handle all app-side setup (as well as copying the factory)
source ../apps/${APP}/setup.sh ${BASEPATH}${RUNFOLDER}

cp ../makefiles/${EXECNAME} ${BASEPATH}${RUNFOLDER}/exec
cp ${SETTINGSNAME} ${BASEPATH}${RUNFOLDER}/settings.sh
cp ${SETTINGSNAME} ${BASEPATH}${RUNFOLDER}/policy_settings.sh
cp runFalcon_train.sh ${BASEPATH}${RUNFOLDER}/run.sh
cp $0 ${BASEPATH}${RUNFOLDER}/launch.sh

cd ${BASEPATH}${RUNFOLDER}

./run.sh ${NTHREADS} ${NTASK}
