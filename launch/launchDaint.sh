#!/bin/bash
EXECNAME=rl
RUNFOLDER=$1
NNODES=$2
APP=$3
SETTINGSNAME=$4

MYNAME=`whoami`
BASEPATH="/scratch/snx3000/${MYNAME}/smarties/"
mkdir -p ${BASEPATH}${RUNFOLDER}
lfs setstripe -c 1 ${BASEPATH}${RUNFOLDER}

if [ $# -gt 4 ] ; then
    POLICY=$5
    cp $5 ${BASEPATH}${RUNFOLDER}/policy.net
fi
if [ $# -lt 7 ] ; then
    NTASK=2 #n tasks per node
    NTHREADS=12 #n threads per task
else
    NTASK=$6
    NTHREADS=$7
fi
if [ $# -lt 8 ] ; then
WCLOCK=00:30:00 #chaining
else
    WCLOCK=$8
fi
NPROCESS=$((${NNODES}*${NTASK}))

#this handles app-side setup (incl. copying the factory)
source ../apps/${APP}/setup.sh

cp ../makefiles/${EXECNAME} ${BASEPATH}${RUNFOLDER}/exec
cp ${SETTINGSNAME} ${BASEPATH}${RUNFOLDER}/settings.sh
cp ${SETTINGSNAME} ${BASEPATH}${RUNFOLDER}/policy_settings.sh
#cp daint_sbatch ${BASEPATH}${RUNFOLDER}/daint_sbatch
#cp runDaint_learn.sh ${BASEPATH}${RUNFOLDER}/run.sh
cp $0 ${BASEPATH}${RUNFOLDER}/launch.sh

cd ${BASEPATH}${RUNFOLDER}
if [ ! -f settings.sh ];then
    echo ${SETTINGSNAME}" not found! - exiting"
    exit -1
fi
source settings.sh
SETTINGS+=" --nThreads ${NTHREADS}"
echo $SETTINGS > settings.txt
echo  ${SETTINGS}
echo ${NPROCESS} ${NNODES} ${NTASK} ${NTHREADS}

cat <<EOF >daint_sbatch
#!/bin/bash -l

#SBATCH --account=s658
#SBATCH --job-name="${RUNFOLDER}"
#SBATCH --output=${RUNFOLDER}_%j.txt
#SBATCH --error=${RUNFOLDER}_%j.txt
#SBATCH --time=${WCLOCK}
#SBATCH --nodes=${NNODES}
# #SBATCH --partition=viz
#SBATCH --ntasks-per-node=${NTASK}
#SBATCH --cpus-per-task=${NTHREADS}
#SBATCH --constraint=gpu
module load daint-gpu 
#module load slurm
#module swap PrgEnv-cray PrgEnv-gnu
#module swap gcc/4.8.2 gcc/4.9.2
export OMP_NUM_THREADS=${NTHREADS}
export CRAY_CUDA_MPS=1

srun -n ${NPROCESS} --cpu_bind=none --ntasks-per-node=${NTASK} --cpus-per-task=${NTHREADS} ./exec ${SETTINGS}
EOF

chmod 755 daint_sbatch

sbatch daint_sbatch
