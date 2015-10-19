#!/bin/bash

EXECNAME=rl
NTHREADS=$1

SETTINGS+=" --learn_rate 0.1"
SETTINGS+=" --gamma 0.9"
SETTINGS+=" --greedy_eps 0.01"

SETTINGS+=" --save_freq 1000000"
SETTINGS+=" --learn A"
SETTINGS+=" --net LSTM"
SETTINGS+=" --debug_lvl 0"
SETTINGS+=" --config factory"

RESTART=" --restart res/policy"

RESTARTPOLICY=" -restartPolicy 1"

OPTIONS=${SETTINGS}${RESTART}

export OMP_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
export LD_LIBRARY_PATH=/Users/laskariangeliki/Documents/tbb40_297oss/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/Users/laskariangeliki/Documents/tbb40_297oss/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/novatig/armadillo/usr/lib64/:$LD_LIBRARY_PATH

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
cp ../samples/cart-pole ../run$2/
cd ../run$2

/opt/mpich/bin/mpirun -np $1 ./executable ${OPTIONS}










