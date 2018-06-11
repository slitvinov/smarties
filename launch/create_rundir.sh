#!/bin/bash
EXECNAME=rl
RUNFOLDER=$1

if [ $# -lt 4 ] ; then
echo "Usage: ./launch_openai.sh RUNFOLDER NTHREADS APP SETTINGS_PATH  (SLAVES PER LEARNER) (N LEARNERS)"
exit 1
fi

MYNAME=`whoami`
HOST=`hostname`
if [ ${HOST:0:5} == 'euler' ] || [ ${HOST:0:5} == 'eu-lo' ] || [ ${HOST:0:4} == 'eu-c' ] ; then
	export BASEPATH="/cluster/scratch/${MYNAME}/smarties/"
else
	export BASEPATH="../runs/"
fi
mkdir -p ${BASEPATH}${RUNFOLDER}
