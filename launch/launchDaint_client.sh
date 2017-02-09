#!/bin/bash
RUNFOLDER=$1
APP=$2
POLICY=$3
NNODESX=$4
NNODESY=$5

MYNAME=`whoami`
BASEPATH="/scratch/snx3000/${MYNAME}/smarties/"
echo ${BASEPATH}${RUNFOLDER}
mkdir -p ${BASEPATH}${RUNFOLDER}
mkdir -p ${BASEPATH}${RUNFOLDER}"/simulation"
#lfs setstripe -c 1 ${BASEPATH}${RUNFOLDER}


#this must handle all app-side setup (as well as copying the factory)
source ../apps/${APP}/setup.sh ${BASEPATH}${RUNFOLDER}

cp ${POLICY}_net ${BASEPATH}${RUNFOLDER}/policy_net
#cp ${POLICY}_mems ${BASEPATH}${RUNFOLDER}/policy_mems
cp ${POLICY}_data_stats ${BASEPATH}${RUNFOLDER}/policy_data_stats
cp ${POLICY}_settings.sh ${BASEPATH}${RUNFOLDER}/settings.sh

cp ../makefiles/rl ${BASEPATH}${RUNFOLDER}/exec
cp runDaint_client.sh ${BASEPATH}${RUNFOLDER}/runClient.sh
cp $0 ${BASEPATH}${RUNFOLDER}/launch.sh

cd ${BASEPATH}${RUNFOLDER}"/simulation"
source ../launchSim.sh ${NNODESX} ${NNODESY}
