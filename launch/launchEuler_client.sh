#!/bin/bash
EXECNAME=rl
RUNFOLDER=$1
APP=$2
#SETTINGSNAME=$3 MUST BE IN THE SAME DIR AS POLICY NAMED policy_settings.sh
POLICY=$3
NNODES=$4
MYNAME=`whoami`
#BASEPATH="../"
BASEPATH="/cluster/scratch/${MYNAME}/smarties/"
echo ${BASEPATH}${RUNFOLDER}
mkdir -p ${BASEPATH}${RUNFOLDER}
mkdir -p ${BASEPATH}${RUNFOLDER}"/simulation"

if [ $# -lt 5 ] ; then
    WCLOCK=24:00
else
    WCLOCK=$6
fi

NTHREADSPERNODE=6
NPROCESSORS=$((${NNODES}*${NTHREADSPERNODE}))

#this must handle all app-side setup (as well as copying the factory)
source ../apps/${APP}/setup.sh ${BASEPATH}${RUNFOLDER}
cp ${POLICY}_net ${BASEPATH}${RUNFOLDER}/policy_net
#cp ${POLICY}_mems ${BASEPATH}${RUNFOLDER}/policy_mems
cp ${POLICY}_data_stats ${BASEPATH}${RUNFOLDER}/policy_data_stats
cp ../makefiles/${EXECNAME} ${BASEPATH}${RUNFOLDER}/exec
#cp ${SETTINGSNAME} ${BASEPATH}${RUNFOLDER}/settings.sh
cp ${POLICY}_settings.sh ${BASEPATH}${RUNFOLDER}/settings.sh
cp runEuler_client.sh ${BASEPATH}${RUNFOLDER}/runClient.sh
cp $0 ${BASEPATH}${RUNFOLDER}/launch.sh

cd ${BASEPATH}${RUNFOLDER}"/simulation"

#../launchSim.sh 0
bsub -J ${RUNFOLDER} -n ${NPROCESSORS} -W ${WCLOCK} ../launchSim.sh 0
