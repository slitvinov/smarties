#!/bin/bash
NPROCESS=$1
NTHREADS=$2
TASKPERN=$3
ulimit -c unlimited
module load gcc/4.9.2
#module load open_mpi/1.6.5

export LD_LIBRARY_PATH=/cluster/work/infk/cconti/VTK5.8_gcc/lib/vtk-5.8/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/cluster/home/mavt/chatzidp/usr/mpich3/lib/:$LD_LIBRARY_PATH
export PATH=/cluster/home/mavt/chatzidp/usr/mpich3/bin:$PATH
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
SETTINGS+=" --nThreads ${NTHREADS}"
echo $SETTINGS > settings.txt
env > environment.log
echo ${NPROCESS} ${NTHREADS}
module load valgrind
#mpirun -np ${NPROCESS} --mca btl tcp,self -bynode ./exec ${SETTINGS} #openmpi

sort $LSB_DJOB_HOSTFILE | uniq  > lsf_hostfile
mpich_run -n ${NPROCESS} -ppn ${TASKPERN} -bind-to none -launcher ssh -f lsf_hostfile ./exec ${SETTINGS}
#module load valgrind
#mpich_run -n ${NPROCESS} -ppn ${TASKPERN} -bind-to none -launcher ssh -f lsf_hostfile valgrind  --tool=memcheck  --leak-check=yes  --track-origins=yes ./exec ${SETTINGS}

#valgrind  --num-callers=100  --tool=memcheck  --leak-check=yes  --track-origins=yes --show-reachable=yes
#/opt/mpich/bin/mpirun -np ${NPROCESS} ./exec ${SETTINGS} #falcon/panda
