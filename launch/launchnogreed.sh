#!/bin/bash

EXECNAME=rl
NTHREADS=$1

SETTINGS+=" --learn_rate 0.1"
SETTINGS+=" --gamma 0.95"
SETTINGS+=" --greedy_eps 0.0" #high because we are restarting from same initial conditions

SETTINGS+=" --save_freq 100"
SETTINGS+=" --debug_lvl 3"
SETTINGS+=" --config factory"

RESTART=" --restart res/policy_backup"

RESTARTPOLICY=" -restartPolicy 1"

OPTIONS=${SETTINGS}${RESTART}

export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=/Users/laskariangeliki/Documents/tbb40_297oss/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/Users/laskariangeliki/Documents/tbb40_297oss/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/novatig/armadillo/usr/lib64/:$LD_LIBRARY_PATH

mkdir ../run$3
mkdir ../run$3/last_sim

if [ "${RESTARTPOLICY}" = " -restartPolicy 1" ]; then
echo "---- launch.sh >> Restart Policy ----"
mkdir -p ../run$3/res
cp ../launch/policy* ../run$3/res/
#    cp ../factory/policy* ${BASEPATH}${RUNFOLDER}/
fi

#cp ../makefiles/${EXECNAME} ../run$3/executable
cp ../factory/factory$2 ../run$3/factory
cd ../run$3

/opt/mpich/bin/mpirun -np $1 ./executable ${OPTIONS}










