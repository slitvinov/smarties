#!/bin/bash
EXECNAME=rl
RUNFOLDER=$1
NNODES=$2
APP=$3
SETTINGSNAME=$4

MYNAME=`whoami`
BASEPATH="../runs/"
mkdir -p ${BASEPATH}${RUNFOLDER}
#rm /tmp/smarties_sock_
if [ $# -gt 4 ] ; then
    POLICY=$5
    cp $5 ${BASEPATH}${RUNFOLDER}/policy.net
fi
if [ $# -lt 7 ] ; then
    NTASK=1 #n tasks per node
    NTHREADS=4 #n threads per task
else
    NTASK=$6
    NTHREADS=$7
fi

NTHREADSPERNODE=8
NPROCESS=$((${NNODES}*${NTASK}))
NPROCESSORS=$((${NNODES}*${NTHREADSPERNODE}))

cat <<EOF >${BASEPATH}${RUNFOLDER}/launchSim.sh
python ../openaibot.py \$1 $APP
EOF
cp ../apps/openai/factory ${BASEPATH}${RUNFOLDER}/factory
cp ../apps/openai/openaibot.py ${BASEPATH}${RUNFOLDER}/
chmod +x ${BASEPATH}${RUNFOLDER}/launchSim.sh

cp ../makefiles/${EXECNAME} ${BASEPATH}${RUNFOLDER}/exec
cp ${SETTINGSNAME} ${BASEPATH}${RUNFOLDER}/settings.sh
cp run.sh ${BASEPATH}${RUNFOLDER}/run.sh
cp $0 ${BASEPATH}${RUNFOLDER}/launch.sh

cd ${BASEPATH}${RUNFOLDER}
./run.sh ${NPROCESS} ${NTHREADS} ${NTASK}
