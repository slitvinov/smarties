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
HOSTNAME=`hostname`
BASEPATH="../runs/"
mkdir -p ${BASEPATH}${RUNFOLDER}
#rm /tmp/smarties_sock_
if [ $# -gt 5 ] ; then
POLICY=$6
cp ${POLICY}_net ${BASEPATH}${RUNFOLDER}/policy_net
cp ${POLICY}_data_stats ${BASEPATH}${RUNFOLDER}/policy_data_stats
cp ${POLICY}.status ${BASEPATH}${RUNFOLDER}/policy.status
fi

if [ $# -gt 6 ] ; then
NTASK=$7
else
NTASK=1 #n tasks per node
fi

NPROCESS=$((${NNODES}*${NTASK}))

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
./run.sh ${NPROCESS} ${NTHREADS} ${NTASK}
