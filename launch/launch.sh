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

if [ ! -f $SETTINGSNAME ] ; then
	echo "Cannot find settings file."
	exit 1
fi

MYNAME=`whoami`
HOST=`hostname`
if [ ${HOST:0:5} == 'euler' ] || [ ${HOST:0:5} == 'eu-lo' ] || [ ${HOST:0:4} == 'eu-c' ] ; then
	BASEPATH="/cluster/scratch/${MYNAME}/smarties/"
else
	BASEPATH="../runs/"
fi
mkdir -p ${BASEPATH}${RUNFOLDER}

if [ $# -gt 5 ] && [ $6 != 'none' ] ; then
	POLICY=$6
	if [ -f ${POLICY}_net.raw ] ; then
		cp ${POLICY}_net.raw ${BASEPATH}${RUNFOLDER}/policy_net.raw
	elif [ -f ${POLICY}_net ] ; then
		cp ${POLICY}_net ${BASEPATH}${RUNFOLDER}/policy_net
	else
		echo "Cannot find saved network file."
		exit 1
	fi
	cp ${POLICY}_data_stats ${BASEPATH}${RUNFOLDER}/policy_data_stats
	cp ${POLICY}.status ${BASEPATH}${RUNFOLDER}/policy.status
fi

if [ $# -gt 6 ] ; then
	NTASK=$7
else
	NTASK=1 #n tasks per node
fi

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

cp ../makefiles/${EXECNAME} ${BASEPATH}${RUNFOLDER}/rl
cp ${SETTINGSNAME} ${BASEPATH}${RUNFOLDER}/settings.sh
cp run.sh ${BASEPATH}${RUNFOLDER}/run.sh
cp $0 ${BASEPATH}${RUNFOLDER}/launch_smarties.sh
cp ${SETTINGSNAME} ${BASEPATH}${RUNFOLDER}/policy_settings.sh
git log | head  > ${BASEPATH}${RUNFOLDER}/gitlog.log
git diff > ${BASEPATH}${RUNFOLDER}/gitdiff.log

cd ${BASEPATH}${RUNFOLDER}

NPROCESS=$((${NNODES}*${NTASK}))
if [ ${HOST:0:5} == 'euler' ] || [ ${HOST:0:5} == 'eu-lo' ] || [ ${HOST:0:4} == 'eu-c' ] ; then
	NTHREADSPERNODE=24
	NPROCESSORS=$((${NNODES}*${NTHREADSPERNODE}))
	bsub -J ${RUNFOLDER} -R "select[model==XeonE5_2680v3]" -n ${NPROCESSORS} -W 24:00 ./run.sh ${NPROCESS} ${NTHREADS} ${NTASK}
else
./run.sh ${NPROCESS} ${NTHREADS} ${NTASK}
fi
