#!/bin/bash

EXECNAME=rl
NNODES=$1
RUNFOLDER=$2

NTHREADS=48
NSLAVESONNODE=1
module load open_mpi
module load gcc/4.9.2

BASEPATH="/cluster/scratch_xp/public/novatig/smarties/"

SETTINGS+=" --gamma 0.8" #Crucial: discount factor
SETTINGS+=" --nnl1 36" #Neurons in first layer
SETTINGS+=" --nnl2 24" #Neurons in second layer
SETTINGS+=" --nnl3 12" #Neurons in first layer
SETTINGS+=" --nnm1 0" #Neurons in first layer
SETTINGS+=" --nnm2 0" #Neurons in second layer
SETTINGS+=" --nnm3 0" #Neurons in first layer
SETTINGS+=" --greedyeps 0.0"
SETTINGS+=" --bTrain 0"
SETTINGS+=" --nne 0.0001" #(Initial) learning rate

OPTIONS=${SETTINGS}${RESTART}
export LD_LIBRARY_PATH=/cluster/work/infk/wvanrees/apps/TBB/tbb42_20140122oss/build/linux_intel64_gcc_cc4.7.2_libc2.12_kernel2.6.32_release/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/cluster/work/infk/cconti/VTK5.8_gcc/lib/vtk-5.8/:$LD_LIBRARY_PATH
#export PATH=/cluster/home/mavt/chatzidp/usr/mpich3/bin:$PATH
export LD_LIBRARY_PATH=/cluster/apps/openmpi/1.6.5/x86_64/gcc_4.9.2/lib/:$LD_LIBRARY_PATH

export OMP_NUM_THREADS=48
#export OMP_PROC_BIND=true
#export OMP_NESTED=true
mkdir -p ${BASEPATH}${RUNFOLDER}

cp $0 ${BASEPATH}${RUNFOLDER}/launch.sh
cp ../makefiles/${EXECNAME} ${BASEPATH}${RUNFOLDER}/exec
cp ../factory/factory2FMovie ${BASEPATH}${RUNFOLDER}/factory
#cp obs_final_final.txt ../run$2/history.txt
cp policy_ANN_DY_plus17 ${BASEPATH}${RUNFOLDER}/policy.net_tmp

cd ${BASEPATH}${RUNFOLDER}

#mpich_run -n ${NPROCESS} -ppn 4 -launcher ssh -f $LSB_DJOB_HOSTFILE -bind-to none ./exec ${OPTIONS}
mpirun -np 2 --mca btl tcp,self -bynode ./exec ${OPTIONS}

#/opt/mpich/bin/mpirun -np $1 ./exec ${OPTIONS}
#mpirun -np $1 ./exec ${OPTIONS}
