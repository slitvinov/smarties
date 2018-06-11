#!/bin/bash
RUNFOLDER=$1
NTHREADS=$2
SETTINGSNAME=$4

if [ $# -lt 4 ] ; then
	echo "Usage: ./launch_openai.sh RUNFOLDER OMP_THREADS APP SETTINGS_PATH (POLICY_PATH) (N_MPI_TASK_PER_NODE)"
	exit 1
fi
if [ $# -gt 4 ] ; then
NSLAVESPERMASTER=$5
else
NSLAVESPERMASTER=1 #n master ranks
fi
if [ $# -gt 5 ] ; then
NMASTERS=$6
else
NMASTERS=1 #n master ranks
fi
if [ $# -gt 6 ] ; then
NNODES=$7
else
NNODES=1 #n master ranks
fi

NTASKPERMASTER=$((1+${NSLAVESPERMASTER})) # master plus its slaves
NPROCESS=$((${NMASTERS}*$NTASKPERMASTER))
NTASKPERNODE=$((${NPROCESS}/${NNODES}))

cp run.sh ${BASEPATH}${RUNFOLDER}/run.sh
cp ../makefiles/rl ${BASEPATH}${RUNFOLDER}/rl
cp $0 ${BASEPATH}${RUNFOLDER}/launch_smarties.sh
cp ${SETTINGSNAME} ${BASEPATH}${RUNFOLDER}/settings.sh
git log | head  > ${BASEPATH}${RUNFOLDER}/gitlog.log
git diff > ${BASEPATH}${RUNFOLDER}/gitdiff.log

cd ${BASEPATH}${RUNFOLDER}

HOST=`hostname`
# || [ ${HOST:0:4} == 'eu-c' ]
if [ ${HOST:0:5} == 'euler' ] || [ ${HOST:0:5} == 'eu-lo' ] ; then
	NTHREADSPERNODE=24
	NPROCESSORS=$((${NNODES}*${NTHREADSPERNODE}))
	bsub -J ${RUNFOLDER} -R "select[model==XeonE5_2680v3]" -n ${NPROCESSORS} -W 24:00 ./run.sh ${NPROCESS} ${NTHREADS} ${NTASKPERNODE} 1
else
./run.sh ${NPROCESS} ${NTHREADS} ${NTASKPERNODE} ${NMASTERS}
fi
