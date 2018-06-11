#!/bin/bash
RUNFOLDER=$1
NTHREADS=$2
APP=$3
SETTINGSNAME=$4

if [ $# -lt 4 ] ; then
	echo "Usage: ./launch_openai.sh RUNFOLDER OMP_THREADS APP SETTINGS_PATH (POLICY_PATH) (N_MPI_TASK_PER_NODE)"
	exit 1
fi

source create_rundir.sh

#this must handle all app-side setup (as well as copying the factory)
if [ -d ${APP} ]; then
	if [ -x ${APP}/setup.sh ] ; then
		source ${APP}/setup.sh ${BASEPATH}${RUNFOLDER}
	else
		echo "${APP}/setup.sh does not exist or I cannot execute it"
		exit 1
	fi
else
	if [ -x ../apps/${APP}/setup.sh ] ; then
		source ../apps/${APP}/setup.sh ${BASEPATH}${RUNFOLDER}
	else
		echo "../apps/${APP}/setup.sh does not exist or I cannot execute it"
		exit 1
	fi
fi

source launch_base.sh
