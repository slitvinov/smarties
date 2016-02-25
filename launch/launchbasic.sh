#!/bin/bash

EXECNAME=rl
NTHREADS=$1

RESTART=" --restart res/policy"

RESTARTPOLICY=" -restartPolicy 1"

OPTIONS=${SETTINGS}${RESTART}

export OMP_NUM_THREADS=4

mkdir -p ../run$2
mkdir -p ../run$2/last_sim

if [ "${RESTARTPOLICY}" = " -restartPolicy 1" ]; then
echo "---- launch.sh >> Restart Policy ----"
mkdir -p ../run$2/res
#cp ../launch/policy* ../run$2/res/
#    cp ../factory/policy* ${BASEPATH}${RUNFOLDER}/
fi

cp ../makefiles/${EXECNAME} ../run$2/executable
cp ../factory/factoryCartOld ../run$2/factory
cp ../apps/cart-pole-old ../run$2/
cd ../run$2

./executable ${OPTIONS}
