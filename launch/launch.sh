#!/bin/bash

EXECNAME=rl
NTHREADS=$1

SETTINGS+=" --learn_rate 0.1"
SETTINGS+=" --gamma 0.8"
SETTINGS+=" --greedy_eps 0.00" #high because we are restarting from same initial conditions

SETTINGS+=" --save_freq 100"
SETTINGS+=" --debug_lvl 9"
SETTINGS+=" --learn NFQNN"
SETTINGS+=" --net WAVE"
SETTINGS+=" --config factory"

RESTART=" --restart res/policy_backup"

RESTARTPOLICY=" -restartPolicy 1"

OPTIONS=${SETTINGS}${RESTART}

export OMP_NUM_THREADS=15
export LD_LIBRARY_PATH=/Users/laskariangeliki/Documents/tbb40_297oss/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/Users/laskariangeliki/Documents/tbb40_297oss/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/novatig/armadillo/usr/lib64/:$LD_LIBRARY_PATH

mkdir ../run$3
mkdir ../run$3/last_sim

if [ "${RESTARTPOLICY}" = " -restartPolicy 1" ]; then
echo "---- launch.sh >> Restart Policy ----"
mkdir -p ../run$3/res
#cp ../launch/policy* ../run$3/res/
#    cp ../factory/policy* ${BASEPATH}${RUNFOLDER}/
fi
mkdir -p ../run$3/restart
cp $HOME/MRAGapps/IF2D_ROCKS/makefiles/hyperion ../run$3/hyperion
cp $HOME/MRAGapps/IF2D_ROCKS/launch/_restart_learn$2/* ../run$3/restart/

cp ../makefiles/${EXECNAME} ../run$3/executable
cp ../factory/factory$2 ../run$3/factory
cp history$2.txt ../run$3/history.txt
cd ../run$3

/opt/mpich/bin/mpirun -np $1 ./executable ${OPTIONS}










