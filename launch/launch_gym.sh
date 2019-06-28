#!/bin/bash
#
#  smarties
#  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
#  Distributed under the terms of the MIT license.
#
#  Created by Guido Novati (novatig@ethz.ch).
#
RUNFOLDER=$1
APP=$2

if [ $# -lt 2 ] ; then
echo "Usage: ./launch_gym.sh RUNFOLDER ENVIRONMENT_APP (... optional arguments defined in launch_base.sh )"
exit 1
fi

source create_rundir.sh

HOSTNAME=`hostname`
cat <<EOF >${BASEPATH}${RUNFOLDER}/launchSim.sh
python3 ../Communicator_gym.py $APP
EOF
chmod +x ${BASEPATH}${RUNFOLDER}/launchSim.sh
export INTERNALAPP=false

cp ../makefiles/smarties*                      ${BASEPATH}${RUNFOLDER}/
cp ../source/Communicators/Communicator_gym.py ${BASEPATH}${RUNFOLDER}/

source launch_base.sh

#python ../openaibot.py \$1 $APP
#xvfb-run -s "-screen $DISPLAY 1400x900x24" -- python ../openaibot.py \$1 $APP
#vglrun -c proxy python3 ../Communicator.py \$1 $APP
