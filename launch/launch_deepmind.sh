#!/bin/bash
EXECNAME=rl
RUNFOLDER=$1
NTHREADS=$2
ENV=$3
TASK=$4
SETTINGSNAME=$5

if [ $# -lt 5 ] ; then
echo "Usage: ./launch_deepmind.sh RUNFOLDER OMP_THREADS ENV TASK SETTINGS_PATH (N_MPI_TASK_PER_NODE)"
exit 1
fi

source create_rundir.sh

HOSTNAME=`hostname`
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
chmod +x ${BASEPATH}${RUNFOLDER}/launchSim.sh

git log | head  > ${BASEPATH}${RUNFOLDER}/gitlog.log
git diff > ${BASEPATH}${RUNFOLDER}/gitdiff.log

cp ../source/Communicators/Communicator.py     ${BASEPATH}${RUNFOLDER}/
cp ../source/Communicators/Communicator_dmc.py ${BASEPATH}${RUNFOLDER}/

export DISABLE_MUJOCO_RENDERING=1
./launch_base.sh $1 $2 $3 $5 $6 $7 $8
