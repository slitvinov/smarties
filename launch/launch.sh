#!/bin/bash

EXECNAME=rl
NTHREADS=$1

#Learner:
SETTINGS+=" --learn NFQNN" #Which
SETTINGS+=" --learn_rate 0.1" #Mostly unused, except for vanilla Q Learning
SETTINGS+=" --gamma 0.8" #Crucial: discount factor
SETTINGS+=" --greedy_eps 0.00" #Mostly unused, except for vanilla Q Learning
SETTINGS+=" --AL_fac 2." #If <1 then Advantage Learning: detestable piece of crap

#Approximator:
SETTINGS+=" --net LSTM" #Network type: WAVE, ANN, LSTM
SETTINGS+=" --nnl1 36" #Neurons in first layer
SETTINGS+=" --nnl2 24" #Neurons in second layer
SETTINGS+=" --nnl3 12" #Neurons in first layer
SETTINGS+=" --nnl4  6" #Neurons in second layer
SETTINGS+=" --nnm1 24"
SETTINGS+=" --nnm2  0"
SETTINGS+=" --nnout 3" #Either 1 or whatever the number of possible actions are
SETTINGS+=" --nne 0.05" #(Initial) learning rate
SETTINGS+=" --nna 0.3" #Momentum update
SETTINGS+=" --nnk 0.3"  #eligibility trace factor
SETTINGS+=" --nnl 1e-6" #Regularization
SETTINGS+=" --nnS 1e-7" #adaptive learn rate

#Stuff:
SETTINGS+=" --save_freq 100"
SETTINGS+=" --debug_lvl 7"
SETTINGS+=" --config factory"

RESTART=" --restart res/policy"

RESTARTPOLICY=" -restartPolicy 1"

OPTIONS=${SETTINGS}${RESTART}

export OMP_NUM_THREADS=6
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

cp $0 ../run$3/$4
cp ../makefiles/${EXECNAME} ../run$3/$4
cp ../factory/factory$2 ../run$3/factory
cp history$2.txt ../run$3/history.txt
cd ../run$3

/opt/mpich/bin/mpirun -np $1 ./$4 ${OPTIONS}
