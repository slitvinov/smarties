#!/bin/bash

EXECNAME=rl
NTHREADS=$1

SETTINGS+=" --learn_rate 0.1"
SETTINGS+=" --gamma 0.9"
SETTINGS+=" --greedy_eps 0.25" #high because we are restarting from same initial conditions
SETTINGS+=" --save_freq 1"
SETTINGS+=" --debug_lvl 5"
SETTINGS+=" --config factory"
RESTART=" --restart res/policy_backup"
RESTARTPOLICY=" -restartPolicy 1"
OPTIONS=${SETTINGS}${RESTART}
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=/Users/laskariangeliki/Documents/tbb40_297oss/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/Users/laskariangeliki/Documents/tbb40_297oss/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/novatig/armadillo/usr/lib64/:$LD_LIBRARY_PATH

mkdir ../run
mkdir ../run/last_sim

if [ "${RESTARTPOLICY}" = " -restartPolicy 1" ]; then
echo "---- launch.sh >> Restart Policy ----"
mkdir -p ../run/res
cp ../launch/policy* ../run/res/
#    cp ../factory/policy* ${BASEPATH}${RUNFOLDER}/
fi

cp ../makefiles/${EXECNAME} ../run/executable
cp ../factory/factory$2 ../run/factory
cd ../run

/opt/mpich/bin/mpirun -np $1 valgrind ./executable ${OPTIONS}










