#!/bin/bash
EXECNAME=rl
RUNFOLDER=$1
NTHREADS=$2
NNODES=$3
APP=$4
SETTINGSNAME=$5

if [ $# -lt 5 ] ; then
echo "Usage: ./launch_openai.sh RUNFOLDER OMP_THREADS MPI_NODES APP SETTINGS_PATH (POLICY_PATH) (N_MPI_TASK_PER_NODE)"
exit 1
fi

MYNAME=`whoami`
BASEPATH="../runs/"
mkdir -p ${BASEPATH}${RUNFOLDER}

if [ $# -gt 5 ] ; then
POLICY=$6
cp ${POLICY}_net* ${BASEPATH}${RUNFOLDER}/policy_net*
cp ${POLICY}_data_stats ${BASEPATH}${RUNFOLDER}/policy_data_stats
cp ${POLICY}.status ${BASEPATH}${RUNFOLDER}/policy.status
fi

if [ $# -gt 6 ] ; then
NTASK=$7
else
NTASK=1 #n tasks per node
fi

NPROCESS=$((${NNODES}*${NTASK}))

#this must handle all app-side setup (as well as copying the factory)
if [ -d ${APP} ]; then
	source ${APP}/setup.sh ${BASEPATH}${RUNFOLDER}
else
	source ../apps/${APP}/setup.sh ${BASEPATH}${RUNFOLDER}
fi

cp ../makefiles/${EXECNAME} ${BASEPATH}${RUNFOLDER}/exec
cp ${SETTINGSNAME} ${BASEPATH}${RUNFOLDER}/settings.sh
cp run.sh ${BASEPATH}${RUNFOLDER}/run.sh
cp $0 ${BASEPATH}${RUNFOLDER}/launch.sh
cp ${SETTINGSNAME} ${BASEPATH}${RUNFOLDER}/policy_settings.sh

cd ${BASEPATH}${RUNFOLDER}
./run.sh ${NPROCESS} ${NTHREADS} ${NTASK}
