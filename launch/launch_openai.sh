#!/bin/bash
EXECNAME=rl
RUNFOLDER=$1
NTHREADS=$2 # number of threads per learner
APP=$3
SETTINGSNAME=$4

if [ $# -lt 4 ] ; then
echo "Usage: ./launch_openai.sh RUNFOLDER NTHREADS APP SETTINGS_PATH  (SLAVES PER LEARNER) (N LEARNERS)"
exit 1
fi

if [ $# -gt 4 ] ; then
NSLAVESPERMASTER=$5
else
NSLAVESPERMASTER=1 #n tasks per node
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

MYNAME=`whoami`
HOSTNAME=`hostname`
BASEPATH="../runs/"
mkdir -p ${BASEPATH}${RUNFOLDER}

NTASKPERMASTER=$((1+${NSLAVESPERMASTER})) # master plus its slaves
NPROCESS=$((${NMASTERS}*$NTASKPERMASTER))
NTASKPERNODE=$((${NPROCESS}/${NNODES}))
#python ../openaibot.py \$1 $APP
#xvfb-run -s "-screen $DISPLAY 1400x900x24" -- python ../openaibot.py \$1 $APP
#vglrun -c proxy python3 ../Communicator.py \$1 $APP

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
