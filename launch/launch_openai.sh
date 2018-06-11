#!/bin/bash
EXECNAME=rl
RUNFOLDER=$1
NTHREADS=$2 # number of threads per learner
APP=$3
SETTINGSNAME=$4

if [ $# -lt 4 ] ; then
echo "Usage: ./launch_openai.sh RUNFOLDER NTHREADS APP SETTINGS_PATH  (WORKERS PER LEARNER) (N LEARNERS)"
exit 1
fi

source create_rundir.sh

HOSTNAME=`hostname`
if [ ${HOSTNAME:0:5} == 'falco' ] || [ ${HOSTNAME:0:5} == 'panda' ]
then
cat <<EOF >${BASEPATH}${RUNFOLDER}/launchSim.sh
/home/novatig/Python-3.5.2/build/bin/python3.5 ../Communicator_gym.py \$1 $APP
EOF
else
cat <<EOF >${BASEPATH}${RUNFOLDER}/launchSim.sh
python3 ../Communicator_gym.py \$1 $APP
EOF
fi
chmod +x ${BASEPATH}${RUNFOLDER}/launchSim.sh

cp ../source/Communicators/Communicator.py     ${BASEPATH}${RUNFOLDER}/
cp ../source/Communicators/Communicator_gym.py ${BASEPATH}${RUNFOLDER}/

source launch_base.sh

#python ../openaibot.py \$1 $APP
#xvfb-run -s "-screen $DISPLAY 1400x900x24" -- python ../openaibot.py \$1 $APP
#vglrun -c proxy python3 ../Communicator.py \$1 $APP
