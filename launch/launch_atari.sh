#!/bin/bash
EXECNAME=rl
RUNFOLDER=$1
NTHREADS=$2 # number of threads per learner
APP=$3 # NoFrameskip-v4 will be added  at the end of the task name
SETTINGSNAME=$4

if [ $# -lt 4 ] ; then
echo "Usage: ./launch_atari.sh RUNFOLDER NTHREADS APP SETTINGS_PATH  (WORKERS PER LEARNER) (N LEARNERS)"
exit 1
fi

source create_rundir.sh

HOSTNAME=`hostname`

if [ ${HOSTNAME:0:5} == 'falco' ] || [ ${HOSTNAME:0:5} == 'panda' ]
then
cat <<EOF >${BASEPATH}${RUNFOLDER}/launchSim.sh
/home/novatig/Python-3.5.2/build/bin/python3.5 ../Communicator_atari.py \$1 $APP
EOF
else
cat <<EOF >${BASEPATH}${RUNFOLDER}/launchSim.sh
python3 ../Communicator_atari.py \$1 $APP
EOF
fi
chmod +x ${BASEPATH}${RUNFOLDER}/launchSim.sh

cp ../source/Communicators/Communicator.py       ${BASEPATH}${RUNFOLDER}/
cp ../source/Communicators/Communicator_atari.py ${BASEPATH}${RUNFOLDER}/

# Atari environment specific settings: glue 4 frames together to compose frame
# and use the Nature paper's CNN architecture specified in AtariEnvironment
# Be careful, this may not be overwritten and may cause bugs if re-using folders
cat <<EOF >${BASEPATH}${RUNFOLDER}/appSettings.sh
SETTINGS+=" --environment AtariEnvironment --appendedObs 3 "
EOF
chmod +x ${BASEPATH}${RUNFOLDER}/appSettings.sh


source launch_base.sh
