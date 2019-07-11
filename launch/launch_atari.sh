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
cat <<EOF >${BASEPATH}${RUNFOLDER}/launchSim.sh
python3 ../Communicator_atari.py $APP
EOF
chmod +x ${BASEPATH}${RUNFOLDER}/launchSim.sh
export INTERNALAPP=false

cp ../makefiles/smarties*                        ${BASEPATH}${RUNFOLDER}/
cp ../source/Communicators/Communicator_atari.py ${BASEPATH}${RUNFOLDER}/

source launch_base.sh
