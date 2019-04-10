#!/bin/bash
#
#  smarties
#  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
#  Distributed under the terms of the MIT license.
#
#  Created by Guido Novati (novatig@ethz.ch).
#
EXECNAME=rl
RUNFOLDER=$1
APP=$2 # NoFrameskip-v4 will be added  at the end of the task name

if [ $# -lt 2 ] ; then
echo "Usage: ./launch_atari.sh RUNFOLDER GAMEID( NoFrameskip-v4 will be added internally ) (... optional arguments defined in launch_base.sh )"
exit 1
fi

source create_rundir.sh

HOSTNAME=`hostname`

# Workaround for cselab's headless worstations. If you know that you need to
# modify this for your own setup then you also probably know what to do.
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
export INTERNALAPP=false

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
