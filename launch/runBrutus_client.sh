#!/bin/bash
ulimit -c unlimited
module load gcc/4.9.2

export LD_LIBRARY_PATH=/cluster/home03/mavt/novatig/mpich-3.2/lib/:$LD_LIBRARY_PATH
export PATH=/cluster/home03/mavt/novatig/mpich-3.2/bin/:$PATH

#export LD_LIBRARY_PATH=/cluster/home/mavt/chatzidp/usr/mpich3/lib/:$LD_LIBRARY_PATH
#export PATH=/cluster/home/mavt/chatzidp/usr/mpich3/bin:$PATH
#export LD_LIBRARY_PATH=/cluster/apps/openmpi/1.6.5/x86_64/gcc_4.9.2/lib/:$LD_LIBRARY_PATH
#export OMP_NUM_THREADS=${NTHREADS}
#export OMP_PROC_BIND=true
#export OMP_NESTED=true
#export OMP_WAIT_POLICY=ACTIVE
export TMPDIR=/cluster/scratch_xp/public/novatig/

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


#mpirun -np ${NPROCESS} --mca btl tcp,self -bynode ./exec ${SETTINGS} #openmpi

sort $LSB_DJOB_HOSTFILE | uniq  > lsf_hostfile
module load valgrind
mpich_run -n 1 -ppn 1 -bind-to none -launcher ssh -f lsf_hostfile ./exec ${SETTINGS}

#module load valgrind
#mpich_run -n ${NPROCESS} -ppn ${TASKPERN} -bind-to none -launcher ssh -f lsf_hostfile valgrind  --tool=memcheck  --leak-check=full --show-reachable=no --show-possibly-lost=no --track-origins=yes ./exec ${SETTINGS}

#valgrind  --num-callers=100  --tool=memcheck  --leak-check=yes  --track-origins=yes --show-reachable=yes
#/opt/mpich/bin/mpirun -np ${NPROCESS} ./exec ${SETTINGS} #falcon/panda
