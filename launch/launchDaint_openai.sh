#!/bin/bash
EXECNAME=rl
RUNFOLDER=$1
NNODES=$2
APP=$3
SETTINGSNAME=$4

MYNAME=`whoami`
BASEPATH="/scratch/snx3000/${MYNAME}/smarties/"
#BASEPATH="/scratch/snx1600/${MYNAME}/smarties/"
mkdir -p ${BASEPATH}${RUNFOLDER}
ulimit -c unlimited
#lfs setstripe -c 1 ${BASEPATH}${RUNFOLDER}

if [ $# -gt 4 ] ; then
    POLICY=$5
    cp ${POLICY}_net ${BASEPATH}${RUNFOLDER}/policy_net
    #cp ${POLICY}_mems ${BASEPATH}${RUNFOLDER}/policy_mems
    cp ${POLICY}_data_stats ${BASEPATH}${RUNFOLDER}/policy_data_stats
    cp ${POLICY}.status ${BASEPATH}${RUNFOLDER}/policy.status
fi
if [ $# -lt 7 ] ; then
    NTASK=12 #n tasks per node
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
python ../Communicator.py \$1 $APP
EOF

cat <<EOF >${BASEPATH}${RUNFOLDER}/factory
Environment exec=../launchSim.sh n=1
EOF

#this handles app-side setup (incl. copying the factory)
#cp ../apps/openai/factory ${BASEPATH}${RUNFOLDER}/factory
#cp ../apps/openai/openaibot.py ${BASEPATH}${RUNFOLDER}/
cp ../source/Communicator.py ${BASEPATH}${RUNFOLDER}/
chmod +x ${BASEPATH}${RUNFOLDER}/launchSim.sh

cp ../makefiles/${EXECNAME} ${BASEPATH}${RUNFOLDER}/exec
cp ${SETTINGSNAME} ${BASEPATH}${RUNFOLDER}/settings.sh
cp ${SETTINGSNAME} ${BASEPATH}${RUNFOLDER}/policy_settings.sh
cp $0 ${BASEPATH}${RUNFOLDER}/launch.sh

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

cat <<EOF >daint_sbatch
#!/bin/bash -l

#SBATCH --account=s658
#SBATCH --job-name="${RUNFOLDER}"
#SBATCH --output=${RUNFOLDER}_out_%j.txt
#SBATCH --error=${RUNFOLDER}_err_%j.txt
#SBATCH --time=${WCLOCK}
#SBATCH --nodes=${NNODES}
#SBATCH --ntasks-per-node=${NTASK}
#SBATCH --constraint=gpu

# #SBATCH --partition=debug
# #SBATCH --time=00:30:00
# #SBATCH --cpus-per-task=$((${NTHREADS}/2)) # Hyperthreaded
# #SBATCH --mail-user="${MYNAME}@ethz.ch"
# #SBATCH --mail-type=ALL

module load daint-gpu
export OMP_NUM_THREADS=12
export CRAY_CUDA_MPS=1

srun --ntasks ${NPROCESS} --cpu_bind=none --ntasks-per-node=${NTASK} ./exec ${SETTINGS}

#srun --ntasks ${NPROCESS} --cpu_bind=none --ntasks-per-node=${NTASK} --threads-per-core=2 valgrind  --tool=memcheck  --leak-check=full --show-reachable=no --show-possibly-lost=no --track-origins=yes ./exec ${SETTINGS}
EOF

chmod 755 daint_sbatch


sbatch daint_sbatch
cd -
