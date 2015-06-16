#!/bin/bash

EXECNAME=rl
NNODES=$1
RUNFOLDER=$2

NTHREADS=48
NSLAVESONNODE=4

NPROCESS=$(($NNODES*$NSLAVESONNODE))
NPROCESS=$(($NPROCESS+1))
NPROCESSORS=$(($NNODES*$NTHREADS))

WCLOCK=08:00
TIMES=5 # Job chaining


module load gcc
module load open_mpi
module load openblas

BASEPATH="/cluster/scratch_xp/public/novatig/smarties/"

SETTINGS+=" --learn_rate 0.1"
SETTINGS+=" --gamma 0.9"
SETTINGS+=" --lambda 0.0"
SETTINGS+=" --greedy_eps 0.05"

SETTINGS+=" --save_freq 1"
SETTINGS+=" --config factory"

RESTART=" --restart res/policy_backup"

RESTARTPOLICY=" -restartPolicy 1" # create a new simulation with a given policy (the policy to load has to be in ../launch/policy_backup)

OPTIONS=${SETTINGS}${RESTART}

export LD_LIBRARY_PATH=/cluster/work/infk/wvanrees/apps/TBB/tbb42_20140122oss/build/linux_intel64_gcc_cc4.7.2_libc2.12_kernel2.6.32_release/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/cluster/apps/openmpi/1.6.2/x86_64/gcc_4.8.2/lib/libmpi_cxx/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/cluster/work/infk/cconti/VTK5.8_gcc/lib/vtk-5.8/:$LD_LIBRARY_PATH


export OMP_NUM_THREADS=${NTHREADS}
mkdir -p ${BASEPATH}${RUNFOLDER}
if [ "${RESTARTPOLICY}" = " -restartPolicy 1" ]; then
    echo "---- launch.sh >> Restart Policy ----"
    mkdir -p ${BASEPATH}${RUNFOLDER}/res
#cp ../factory/policy* ${BASEPATH}${RUNFOLDER}/res/
#    cp ../factory/policy* ${BASEPATH}${RUNFOLDER}/
fi
cp ../factory/factoryExt ${BASEPATH}${RUNFOLDER}/factory
cp ../makefiles/${EXECNAME} ${BASEPATH}${RUNFOLDER}/
cp ${0} ${BASEPATH}${RUNFOLDER}/
cd ${BASEPATH}${RUNFOLDER}
echo "Submission 0..."
bsub -J ${RUNFOLDER} -n ${NPROCESSORS} -R span[ptile=${NTHREADS}] -sp 100 -W ${WCLOCK} mpirun -np ${NPROCESS} --mca btl tcp,self -bynode ./${EXECNAME} ${OPTIONS}
# module load valgrind
# bsub -J ${RUNFOLDER} -n ${NPROCESSORS} -R span[ptile=48] -sp 100 -W ${WCLOCK} mpirun -np ${NPROCESS} --mca btl tcp,self -pernode valgrind  --num-callers=100  --tool=memcheck  --leak-check=yes  --track-origins=yes --show-reachable=yes  --log-file=totoValgrind   ./${EXECNAME} ${OPTIONS}
#valgrind --tool=memcheck --leak-check=yes --log-file=toto%p

# Job Chaining

RESTART=" --restart res/policy_backup"
OPTIONS=${SETTINGS}${RESTART}${RESTARTPOLICY}

for (( c=1; c<=${TIMES}-1; c++ ))
do
echo "Submission $c..."
bsub -J ${RUNFOLDER} -n ${NPROCESSORS} -R span[ptile=${NTHREADS}] -sp 100 -w "ended(${RUNFOLDER})" -W ${WCLOCK} mpirun -np ${NPROCESS} --mca btl tcp,self -bynode ./${EXECNAME} ${OPTIONS}
done
