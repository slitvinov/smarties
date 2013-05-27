#!/bin/bash

EXECNAME=rl
NTHREADS=1

SETTINGS+=" --scale 0.01"

SETTINGS+=" --learn_rate 0.08"
SETTINGS+=" --gamma 0.9"
SETTINGS+=" --greedy_eps 0.00"

SETTINGS+=" --save_freq 100000"
SETTINGS+=" --config factory"

RESTART=" --restart"

OPTIONS=${SETTINGS}${RESTART}

export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=/Users/laskariangeliki/Documents/tbb40_297oss/lib/:$DYLD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=/Users/laskariangeliki/Documents/tbb40_297oss/lib/:$LD_LIBRARY_PATH

if [[ ${RESTART} != " --restart" ]]; then
rm -fr ../run
mkdir ../run
fi

if [[ ${RESTART} == " --restart" ]]; then
rm ../run/*.txt
fi

cp ../makefiles/${EXECNAME} ../run/executable  
cp ../factory/factoryRL_test1 ../run/factory
cd ../run

./executable ${OPTIONS}










