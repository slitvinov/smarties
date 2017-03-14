#!/bin/bash
EXECNAME=rl
RUNFOLDER=$1
APP=$2
#SETTINGSNAME=$3 MUST BE IN THE SAME DIR AS POLICY NAMED policy_settings.sh
POLICY=$3

MYNAME=`whoami`
#BASEPATH="../"
BASEPATH="/scratch/${MYNAME}/smarties/"
echo ${BASEPATH}${RUNFOLDER}
mkdir -p ${BASEPATH}${RUNFOLDER}
mkdir -p ${BASEPATH}${RUNFOLDER}"/simulation"

#this must handle all app-side setup (as well as copying the factory)
source ../apps/${APP}/setup.sh ${BASEPATH}${RUNFOLDER}
cp ${POLICY}_net ${BASEPATH}${RUNFOLDER}/policy_net
#cp ${POLICY}_mems ${BASEPATH}${RUNFOLDER}/policy_mems
cp ${POLICY}_data_stats ${BASEPATH}${RUNFOLDER}/policy_data_stats
cp ${POLICY}.status ${BASEPATH}${RUNFOLDER}/policy.status
cp ../makefiles/${EXECNAME} ${BASEPATH}${RUNFOLDER}/exec
#cp ${SETTINGSNAME} ${BASEPATH}${RUNFOLDER}/settings.sh
cp ${POLICY}_settings.sh ${BASEPATH}${RUNFOLDER}/settings.sh
cp runFalcon_client.sh ${BASEPATH}${RUNFOLDER}/runClient.sh
cp $0 ${BASEPATH}${RUNFOLDER}/launch.sh

cd ${BASEPATH}${RUNFOLDER}"/simulation"

../launchSim.sh 0
