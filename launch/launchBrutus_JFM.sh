#!/bin/bash

EXECNAME=rl
NNODES=$1
RUNFOLDER=$2
PTILE=48
NTHREADS=48
NSLAVESONNODE=4
PPN=5
export OMP_NUM_THREADS=12

NPROCESS=$((${NNODES}*${NSLAVESONNODE}))
NPROCESS=$((${NPROCESS}+1))
NPROCESSORS=$((${NNODES}*${NTHREADS}))

WCLOCK=12:00
TIMES=3 # Job chaining


#module load gcc/4.9.2
#module load open_mpi
#export PATH=/cluster/home/mavt/chatzidp/usr/mpich3/bin:$PATH
#export LD_LIBRARY_PATH=/cluster/home/mavt/chatzidp/usr/mpich3/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/cluster/apps/openmpi/1.6.5/x86_64/gcc_4.9.2/lib/:$LD_LIBRARY_PATH
#module load valgrind
BASEPATH="/cluster/scratch_xp/public/novatig/smarties/"

SESETTINGS+=" --gamma 0.99" #Crucial: discount factor
SETTINGS+=" --nnl1 0" #Neurons in first layer
SETTINGS+=" --nnl2 0" #Neurons in second layer
SETTINGS+=" --nnl3 0" #Neurons in first layer
SETTINGS+=" --nnm1 12" #Neurons in first layer
SETTINGS+=" --nnm2 12" #Neurons in second layer
SETTINGS+=" --nnm3 12" #Neurons in first layer
SETTINGS+=" --greedyeps 0.0"

SETTINGS+=" --nne 0.0" #(Initial) learning rate
OPTIONS=${SETTINGS}${RESTART}

export LD_LIBRARY_PATH=/cluster/work/infk/wvanrees/apps/TBB/tbb42_20140122oss/build/linux_intel64_gcc_cc4.7.2_libc2.12_kernel2.6.32_release/:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/cluster/work/infk/cconti/VTK5.8_gcc/lib/vtk-5.8/:$LD_LIBRARY_PATH

mkdir -p ${BASEPATH}${RUNFOLDER}
mkdir -p ${BASEPATH}${RUNFOLDER}/res

cp ../factory/factory2FMovie ${BASEPATH}${RUNFOLDER}/factory
cp ../makefiles/${EXECNAME} ${BASEPATH}${RUNFOLDER}/
cp ${0} ${BASEPATH}${RUNFOLDER}/
cp restart_45mb_Eff ${BASEPATH}${RUNFOLDER}/policy.net_tmp
#cp restart_45mb_Y ${BASEPATH}${RUNFOLDER}/policy.net_tmp

cd ${BASEPATH}${RUNFOLDER}
echo ${NPROCESSORS} ${NTHREADS} ${NPROCESS}

echo "Submission 0..."
bsub -J ${RUNFOLDER} -n ${NPROCESSORS} -R span[ptile=${PTILE}] -sp 100 -W ${WCLOCK} mpirun -np ${NPROCESS} --mca btl tcp,self -bynode ./${EXECNAME} ${OPTIONS}
#bsub -J ${RUNFOLDER} -n ${NPROCESSORS} -R span[ptile=${PTILE}] -sp 100 -W ${WCLOCK} mpich_run -n ${NPROCESS} -ppn ${PPN} -launcher ssh -f $LSB_DJOB_HOSTFILE -bind-to none ./${EXECNAME} ${OPTIONS}

# Job Chaining
for (( c=1; c<=${TIMES}-1; c++ ))
do
echo "Submission $c..."
bsub -J ${RUNFOLDER} -n ${NPROCESSORS} -R span[ptile=${PTILE}] -sp 100 -w "ended(${RUNFOLDER})" -W ${WCLOCK} mpirun -np ${NPROCESS} --mca btl tcp,self -bynode ./${EXECNAME} ${OPTIONS}
#bsub -J ${RUNFOLDER} -n ${NPROCESSORS} -R span[ptile=${PTILE}] -sp 100 -w "ended(${RUNFOLDER})" -W ${WCLOCK} mpich_run -n ${NPROCESS} -ppn ${PPN} -launcher ssh -f $LSB_DJOB_HOSTFILE -bind-to none ./${EXECNAME} ${OPTIONS}
done
