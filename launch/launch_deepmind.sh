#!/bin/bash
EXECNAME=rl
RUNFOLDER=$1
NTHREADS=$2
ENV=$3
TASK=$4
SETTINGSNAME=$5

if [ $# -lt 5 ] ; then
echo "Usage: ./launch_openai.sh RUNFOLDER OMP_THREADS ENV TASK SETTINGS_PATH (N_MPI_TASK_PER_NODE)"
exit 1
fi

if [ $# -gt 5 ] ; then
NSLAVESPERMASTER=$6
else
NSLAVESPERMASTER=1 #n tasks per node
fi
if [ $# -gt 6 ] ; then
NMASTERS=$7
else
NMASTERS=1 #n master ranks
fi
if [ $# -gt 7 ] ; then
NNODES=$8
else
NNODES=1 #n master ranks
fi

MYNAME=`whoami`
HOSTNAME=`hostname`
BASEPATH="../runs/"
mkdir -p ${BASEPATH}${RUNFOLDER}


NTASKPERMASTER=$((1+${NSLAVESPERMASTER})) # master plus its slaves
NPROCESS=$((${NMASTERS}*$NTASKPERMASTER))
NTASKPERNODE=$((${NPROCESS}/${NNODES}))

export DISABLE_MUJOCO_RENDERING=1

if [ ${HOSTNAME:0:5} == 'falco' ] || [ ${HOSTNAME:0:5} == 'panda' ]
then
cat <<EOF >${BASEPATH}${RUNFOLDER}/launchSim.sh
LD_PRELOAD=libstdc++.so.6 ${HOME}/Python-3.5.2/build/bin/python3.5 ../Communicator_dmc.py \$1 $ENV $TASK
EOF
else
cat <<EOF >${BASEPATH}${RUNFOLDER}/launchSim.sh
python3 ../Communicator_dmc.py \$1 $ENV $TASK
EOF
fi

git log | head  > ${BASEPATH}${RUNFOLDER}/gitlog.log
git diff > ${BASEPATH}${RUNFOLDER}/gitdiff.log

#cat <<EOF >${BASEPATH}${RUNFOLDER}/factory
#Environment exec=../launchSim.sh n=1
#EOF

cp ../source/Communicator*.py ${BASEPATH}${RUNFOLDER}/
chmod +x ${BASEPATH}${RUNFOLDER}/launchSim.sh

cp ../makefiles/${EXECNAME} ${BASEPATH}${RUNFOLDER}/rl
cp ${SETTINGSNAME} ${BASEPATH}${RUNFOLDER}/settings.sh
cp ${SETTINGSNAME} ${BASEPATH}${RUNFOLDER}/policy_settings.sh
cp run.sh ${BASEPATH}${RUNFOLDER}/run.sh
cp $0 ${BASEPATH}${RUNFOLDER}/launch.sh

cd ${BASEPATH}${RUNFOLDER}
./run.sh ${NPROCESS} ${NTHREADS} ${NTASKPERNODE} ${NMASTERS} 
