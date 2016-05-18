#!/bin/bash

EXECNAME=rl
NNODES=$1
RUNFOLDER=$2

NTHREADS=48
NSLAVESONNODE=1

NPROCESS=$(($NNODES*$NSLAVESONNODE))
NPROCESS=$(($NPROCESS+1))
NPROCESSORS=$(($NNODES*$NTHREADS))

SETTINGS+=" --gamma 0.99" #Crucial: discount factor
SETTINGS+=" --nnl1 0" #Neurons in first layer
SETTINGS+=" --nnl2 16" #Neurons in second layer
SETTINGS+=" --nnl3 12" #Neurons in first layer
SETTINGS+=" --nnm1 24" #Neurons in first layer
SETTINGS+=" --nnm2 0" #Neurons in second layer
SETTINGS+=" --nnm3 0" #Neurons in first layer


SETTINGS+=" --nne 0.000" #(Initial) learning rate
#SETTINGS+=" --nnl1 32" #Neurons in first layer
#SETTINGS+=" --nnl2 16" #Neurons in second layer
#SETTINGS+=" --nnm1 0"
#SETTINGS+=" --nnm2 0"

OPTIONS=${SETTINGS}${RESTART}
export LD_LIBRARY_PATH=/cluster/work/infk/wvanrees/apps/TBB/tbb42_20140122oss/build/linux_intel64_gcc_cc4.7.2_libc2.12_kernel2.6.32_release/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/cluster/work/infk/cconti/VTK5.8_gcc/lib/vtk-5.8/:$LD_LIBRARY_PATH
#export PATH=/cluster/home/mavt/chatzidp/usr/mpich3/bin:$PATH
export LD_LIBRARY_PATH=/cluster/apps/openmpi/1.6.5/x86_64/gcc_4.9.2/lib/:$LD_LIBRARY_PATH

#export OMP_NUM_THREADS=4
#export OMP_PROC_BIND=true
#export OMP_NESTED=true
mkdir ../run$2
mkdir ../run$2/last_sim
mkdir -p ../run$2/res
mkdir -p ../run$2/restart

cp $0 ../run$2/launch.sh
cp ../makefiles/${EXECNAME} ../run$2/exec
cp ../factory/factory2FMovie ../run$2/factory
#cp history2F.txt ../run$2/history.txt
cp policy.net_tmp ../run$2/
cd ../run$2

#mpich_run -n ${NPROCESS} -ppn 4 -launcher ssh -f $LSB_DJOB_HOSTFILE -bind-to none ./exec ${OPTIONS}
mpirun -np ${NPROCESS} --mca btl tcp,self -bynode valgrind ./exec ${OPTIONS}

#/opt/mpich/bin/mpirun -np $1 ./exec ${OPTIONS}
#mpirun -np $1 ./exec ${OPTIONS}
