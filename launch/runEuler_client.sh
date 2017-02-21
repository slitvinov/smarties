#!/bin/bash
ulimit -c unlimited
module load gcc/4.9.2

#export LD_LIBRARY_PATH=/cluster/home03/mavt/novatig/mpich-3.2/lib/:$LD_LIBRARY_PATH
#export PATH=/cluster/home03/mavt/novatig/mpich-3.2/bin/:$PATH
export MV2_ENABLE_AFFINITY=0
#export LD_LIBRARY_PATH=/cluster/home/mavt/chatzidp/usr/mpich3/lib/:$LD_LIBRARY_PATH
#export PATH=/cluster/home/mavt/chatzidp/usr/mpich3/bin:$PATH
#export LD_LIBRARY_PATH=/cluster/apps/openmpi/1.6.5/x86_64/gcc_4.9.2/lib/:$LD_LIBRARY_PATH
#export OMP_NUM_THREADS=${NTHREADS}
#export OMP_PROC_BIND=true
#export OMP_NESTED=true
#export OMP_WAIT_POLICY=ACTIVE
#export TMPDIR=/cluster/scratch_xp/public/novatig/

SETTINGSNAME=settings.sh
if [ ! -f $SETTINGSNAME ];then
    echo ${SETTINGSNAME}" not found! - exiting"
    exit -1
fi
source $SETTINGSNAME
SETTINGS+=" --nThreads 1"
SETTINGS+=" --isServer 0"

echo $SETTINGS > settings.txt
env > environment.log


#module load valgrind
#mpirun -n 1 -launcher fork  valgrind  --tool=memcheck  --leak-check=full --show-reachable=no --show-possibly-lost=no --track-origins=yes ./exec ${SETTINGS}
mpirun -n 1 -launcher fork ./exec ${SETTINGS}

#mpich_run -n ${NPROCESS} -ppn ${TASKPERN} -bind-to none -launcher ssh -f lsf_hostfile valgrind  --tool=memcheck  --leak-check=full --show-reachable=no --show-possibly-lost=no --track-origins=yes ./exec ${SETTINGS}

