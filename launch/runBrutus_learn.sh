#!/bin/bash
NPROCESS=$1
NTHREADS=$2

NPROCESS=$((${NNODES}*${NTASK}))

module load gcc/4.9.2
export LD_LIBRARY_PATH=/cluster/work/infk/wvanrees/apps/TBB/tbb42_20140122oss/build/linux_intel64_gcc_cc4.7.2_libc2.12_kernel2.6.32_release/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/cluster/work/infk/cconti/VTK5.8_gcc/lib/vtk-5.8/:$LD_LIBRARY_PATH
export PATH=/cluster/home/mavt/chatzidp/usr/mpich3/bin:$PATH
#export LD_LIBRARY_PATH=/cluster/apps/openmpi/1.6.5/x86_64/gcc_4.9.2/lib/:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=${NTHREADS}
#export OMP_PROC_BIND=true
#export OMP_NESTED=true
#export OMP_WAIT_POLICY=ACTIVE

SETTINGSNAME=settings.sh
if [ ! -f $SETTINGSNAME ];then
    echo ${SETTINGSNAME}" not found! - exiting"
    exit -1
fi
source SETTINGSNAME
SETTINGS+=" -nThreads ${NTHREADS}"
echo $SETTINGS > settings.txt

mpich_run -n ${NPROCESS} -launcher ssh -f $LSB_DJOB_HOSTFILE -bind-to none ./exec ${SETTINGS}

#valgrind  --num-callers=100  --tool=memcheck  --leak-check=yes  --track-origins=yes --show-reachable=yes
#mpirun -np ${NPROCESS} --mca btl tcp,self -bynode ./exec ${SETTINGS} #openmpi
#/opt/mpich/bin/mpirun -np ${NPROCESS} ./exec ${SETTINGS} #falcon/panda