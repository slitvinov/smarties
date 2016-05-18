#!/bin/bash

EXECNAME=rl
NNODES=$1
RUNFOLDER=$2

NTHREADS=48
NSLAVESONNODE=4

NPROCESS=$((${NNODES}*${NSLAVESONNODE}))
NPROCESS=$((${NPROCESS}+1))
NPROCESSORS=$((${NNODES}*${NTHREADS}))

WCLOCK=48:00
TIMES=10 # Job chaining


module load gcc/4.9.2
module load open_mpi
#module load valgrind
BASEPATH="/cluster/scratch_xp/public/novatig/smarties/"

SESETTINGS+=" --gamma 0.8" #Crucial: discount factor
SETTINGS+=" --nnl1 0" #Neurons in first layer
SETTINGS+=" --nnl2 0" #Neurons in second layer
SETTINGS+=" --nnl3 0" #Neurons in first layer
SETTINGS+=" --nnm1 24" #Neurons in first layer
SETTINGS+=" --nnm2 24" #Neurons in second layer
SETTINGS+=" --nnm3 24" #Neurons in first layer
SETTINGS+=" --greedyeps 0.5"
SETTINGS+=" --rType 1"
SETTINGS+=" --bTrain 1"

SETTINGS+=" --nne 0.001" #(Initial) learning rate
OPTIONS=${SETTINGS}${RESTART}

export LD_LIBRARY_PATH=/cluster/work/infk/wvanrees/apps/TBB/tbb42_20140122oss/build/linux_intel64_gcc_cc4.7.2_libc2.12_kernel2.6.32_release/:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/cluster/work/infk/cconti/VTK5.8_gcc/lib/vtk-5.8/:$LD_LIBRARY_PATH
#export PATH=/cluster/home/mavt/chatzidp/usr/mpich3/bin:$PATH
export LD_LIBRARY_PATH=/cluster/apps/openmpi/1.6.5/x86_64/gcc_4.9.2/lib/:$LD_LIBRARY_PATH

#export OMP_NUM_THREADS=${NTHREADS}
mkdir -p ${BASEPATH}${RUNFOLDER}
mkdir -p ${BASEPATH}${RUNFOLDER}/res

cp ../factory/factory2F ${BASEPATH}${RUNFOLDER}/factory
cp ../makefiles/${EXECNAME} ${BASEPATH}${RUNFOLDER}/
cp ${0} ${BASEPATH}${RUNFOLDER}/
#cp obs_final.txt ${BASEPATH}${RUNFOLDER}/history.txt
#cp straight_eff_095_32_24_12_00045 ${BASEPATH}${RUNFOLDER}/policy.net_tmp


cd ${BASEPATH}${RUNFOLDER}
echo ${NPROCESSORS} ${NTHREADS} ${NPROCESS}
echo "Submission 0..."
#bsub -J ${RUNFOLDER} -n ${NPROCESSORS} -R span[ptile=${NTHREADS}] -sp 100 -W ${WCLOCK} mpich_run -n ${NPROCESS} -ppn ${NSLAVESONNODE} -launcher ssh -f $LSB_DJOB_HOSTFILE -bind-to none ./${EXECNAME} ${OPTIONS}
bsub -J ${RUNFOLDER} -n ${NPROCESSORS} -R span[ptile=${NTHREADS}] -sp 100 -W ${WCLOCK} mpirun -np ${NPROCESS} --mca btl tcp,self -bynode ./${EXECNAME} ${OPTIONS}
#
# module load valgrind
#bsub -J ${RUNFOLDER} -n ${NPROCESSORS} -R span[ptile=48] -sp 100 -W ${WCLOCK} mpirun -np ${NPROCESS} --mca btl tcp,self -pernode valgrind  --num-callers=100  --tool=memcheck  --leak-check=yes  --track-origins=yes --show-reachable=yes  --log-file=totoValgrind   ./${EXECNAME} ${OPTIONS}
#valgrind --tool=memcheck --leak-check=yes --log-file=toto%p

# Job Chaining
for (( c=1; c<=${TIMES}-1; c++ ))
do
echo "Submission $c..."
bsub -J ${RUNFOLDER} -n ${NPROCESSORS} -R span[ptile=${NTHREADS}] -sp 100 -w "ended(${RUNFOLDER})" -W ${WCLOCK} mpirun -np ${NPROCESS} --mca btl tcp,self -bynode ./${EXECNAME} ${OPTIONS}
#bsub -J ${RUNFOLDER} -n ${NPROCESSORS} -R span[ptile=${NTHREADS}] -sp 100 -w "ended(${RUNFOLDER})" -W ${WCLOCK} mpich_run -n ${NPROCESS} -ppn ${NSLAVESONNODE} -launcher ssh -f $LSB_DJOB_HOSTFILE -bind-to none ./${EXECNAME} ${OPTIONS}
done
