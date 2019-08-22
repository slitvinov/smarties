#!/bin/bash
#
#  smarties
#  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
#  Distributed under the terms of the MIT license.
#
#  Created by Guido Novati (novatig@ethz.ch).
#
HOST=`hostname`
OS_D=`uname`

unset LSB_AFFINITY_HOSTFILE #euler cluster
export MPICH_MAX_THREAD_SAFETY=multiple #MPICH
export MV2_ENABLE_AFFINITY=0 #MVAPICH
export OPENBLAS_NUM_THREADS=1

SETTINGSNAME=settings.sh
if [ ! -f $SETTINGSNAME ];then
    echo ${SETTINGSNAME}" not found! - exiting"
    exit -1
fi
source $SETTINGSNAME
if [ -x appSettings.sh ]; then
  source appSettings.sh
fi

NTHREADS=1
NPROCESS=2

SETTINGS+=" --nWorkers ${NWORKERS}"
SETTINGS+=" --nMasters ${NMASTERS}"
SETTINGS+=" --nThreads ${NTHREADS}"
export OMP_NUM_THREADS=${NTHREADS}

echo "NTHREADS"
echo ${NTHREADS}

env > environment.log
mpirun -n ${NPROCESS} ./rl ${SETTINGS} | tee out.log
#gdb --args ./rl ${SETTINGS}






