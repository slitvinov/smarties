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
cp ../factory/factoryCart ../run$2/factory
cp ../apps/cart-pole ../run$2/
cd ../run$2

#aprun -n $1 ./executable ${OPTIONS}
mpirun -np $1 ./executable ${OPTIONS}
#/opt/mpich/bin/mpirun -np $1 ./executable ${OPTIONS}
