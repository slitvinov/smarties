#!/bin/bash
NTHREADS=$1

export LD_LIBRARY_PATH=/cluster/work/infk/cconti/VTK5.8_gcc/lib/vtk-5.8/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/cluster/home03/mavt/novatig/mpich-3.2/lib/:$LD_LIBRARY_PATH
export PATH=/cluster/home03/mavt/novatig/mpich-3.2/bin/:$PATH

module load gcc/4.9.2
export OMP_NUM_THREADS=${NTHREADS}
#export OMP_PROC_BIND=true
#export OMP_NESTED=true
#export OMP_WAIT_POLICY=ACTIVE

SETTINGSNAME=settings.sh
if [ ! -f $SETTINGSNAME ];then
    echo ${SETTINGSNAME}" not found! - exiting"
    exit -1
fi
source $SETTINGSNAME
SETTINGS+=" --nThreads ${NTHREADS}"
echo $SETTINGS > settings.txt
sort $LSB_DJOB_HOSTFILE | uniq  > lsf_hostfile

#./exec ${SETTINGS}
mpich_run -n 1 -ppn 1 -launcher ssh -f lsf_hostfile  ./exec ${SETTINGS}
#valgrind  --num-callers=100  --tool=memcheck  --leak-check=yes  --track-origins=yes ./exec ${SETTINGS}
