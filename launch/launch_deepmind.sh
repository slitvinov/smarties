#!/bin/bash
EXECNAME=rl
RUNFOLDER=$1
NTHREADS=$2
NNODES=$3
ENV=$4
TASK=$5
SETTINGSNAME=$6

if [ $# -lt 6 ] ; then
echo "Usage: ./launch_openai.sh RUNFOLDER OMP_THREADS MPI_NODES ENV TASK SETTINGS_PATH (N_MPI_TASK_PER_NODE)"
exit 1
fi

MYNAME=`whoami`
HOSTNAME=`hostname`
BASEPATH="../runs/"
mkdir -p ${BASEPATH}${RUNFOLDER}
#rm /tmp/smarties_sock_

if [ $# -gt 6 ] ; then
NTASK=$7
else
NTASK=1 #n tasks per node
fi

NPROCESS=$((${NNODES}*${NTASK}))
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
./run.sh ${NPROCESS} ${NTHREADS} ${NTASK}
