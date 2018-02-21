#!/bin/bash
EXECNAME=rl
RUNFOLDER=$1
NNODES=$2
ENV=$3
TASK=$4
SETTINGSNAME=$5

MYNAME=`whoami`
BASEPATH="/scratch/snx3000/${MYNAME}/smarties/"
#BASEPATH="/scratch/snx1600/${MYNAME}/smarties/"
mkdir -p ${BASEPATH}${RUNFOLDER}
ulimit -c unlimited
#lfs setstripe -c 1 ${BASEPATH}${RUNFOLDER}

if [ $# -lt 7 ] ; then
    NTASK=2 #n tasks per node
    NTHREADS=12 #n threads per task
else
    NTASK=$6
    NTHREADS=$7
fi
if [ $# -lt 8 ] ; then
  WCLOCK=24:00:00 #chaining
else
  WCLOCK=$8
fi
NPROCESS=$((${NNODES}*${NTASK}))

cat <<EOF >${BASEPATH}${RUNFOLDER}/launchSim.sh
LD_PRELOAD=${HOME}/glew-2.1.0/install/lib64/libGLEW.so python3 ../Communicator_dmc.py \$1 $ENV $TASK
EOF

cat <<EOF >${BASEPATH}${RUNFOLDER}/factory
Environment exec=../launchSim.sh n=1
EOF

#this handles app-side setup (incl. copying the factory)
#cp ../apps/openai/factory ${BASEPATH}${RUNFOLDER}/factory
#cp ../apps/openai/openaibot.py ${BASEPATH}${RUNFOLDER}/
cp ../source/Communicator*.py ${BASEPATH}${RUNFOLDER}/
chmod +x ${BASEPATH}${RUNFOLDER}/launchSim.sh

cp ../makefiles/${EXECNAME} ${BASEPATH}${RUNFOLDER}/exec
cp ${SETTINGSNAME} ${BASEPATH}${RUNFOLDER}/settings.sh
cp ${SETTINGSNAME} ${BASEPATH}${RUNFOLDER}/policy_settings.sh
cp $0 ${BASEPATH}${RUNFOLDER}/launch.sh
git log | head  > ${BASEPATH}${RUNFOLDER}/gitlog.log
git diff > ${BASEPATH}${RUNFOLDER}/gitdiff.log

cd ${BASEPATH}${RUNFOLDER}
if [ ! -f settings.sh ];then
    echo ${SETTINGSNAME}" not found! - exiting"
    exit -1
fi
source settings.sh
SETTINGS+=" --nThreads 12"
echo $SETTINGS > settings.txt
echo ${SETTINGS}
echo ${NPROCESS} ${NNODES} ${NTASK} ${NTHREADS}
#s658
cat <<EOF >daint_sbatch
#!/bin/bash -l

#SBATCH --account=eth2 
#SBATCH --job-name="${RUNFOLDER}"
#SBATCH --output=${RUNFOLDER}_out_%j.txt
#SBATCH --error=${RUNFOLDER}_err_%j.txt
#SBATCH --time=${WCLOCK}
#SBATCH --nodes=${NNODES}
#SBATCH --ntasks-per-node=${NTASK}
#SBATCH --constraint=gpu
# #SBATCH --threads-per-core=1

# #SBATCH --partition=debug
# #SBATCH --time=00:30:00
# #SBATCH --cpus-per-task=$((${NTHREADS}/2)) # Hyperthreaded
# #SBATCH --mail-user="${MYNAME}@ethz.ch"
# #SBATCH --mail-type=ALL

module load daint-gpu
export OMP_NUM_THREADS=12
export CRAY_CUDA_MPS=1
export OMP_PROC_BIND=CLOSE
export OMP_PLACES=cores
srun --ntasks ${NPROCESS} --threads-per-core=1 --cpu_bind=none --cpus-per-task=12 --ntasks-per-node=${NTASK} ./exec ${SETTINGS}

#srun --ntasks ${NPROCESS} --cpu_bind=none --ntasks-per-node=${NTASK} --threads-per-core=2 valgrind  --tool=memcheck  --leak-check=full --show-reachable=no --show-possibly-lost=no --track-origins=yes ./exec ${SETTINGS}
EOF

chmod 755 daint_sbatch


sbatch daint_sbatch
cd -