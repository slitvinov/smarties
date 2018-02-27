#!/bin/bash
EXECNAME=rl
RUNFOLDER=$1
APP=$2
SETTINGSNAME=$3

MYNAME=`whoami`
BASEPATH="/scratch/snx3000/${MYNAME}/smarties/"
#BASEPATH="/scratch/snx1600/${MYNAME}/smarties/"
mkdir -p ${BASEPATH}${RUNFOLDER}
ulimit -c unlimited

if [ $# -gt 3 ] ; then
NSLAVESPERMASTER=$4
else
NSLAVESPERMASTER=1 #n tasks per node
fi
if [ $# -gt 4 ] ; then
NMASTERS=$5
else
NMASTERS=1 #n master ranks
fi
if [ $# -gt 5 ] ; then
NTHREADS=$6
else
NTHREADS=12 #threads per master
fi
NTASKPERNODE=$((1+${NSLAVESPERMASTER})) # master plus its slaves
NPROCESS=$((${NMASTERS}*$NTASKPERNODE))

cat <<EOF >${BASEPATH}${RUNFOLDER}/launchSim.sh
python3 ../Communicator_gym.py \$1 $APP
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
SETTINGS+=" --nMasters ${NMASTERS}"
SETTINGS+=" --nThreads ${NTHREADS}"
echo $SETTINGS > settings.txt
echo ${SETTINGS}
#eth2
cat <<EOF >daint_sbatch
#!/bin/bash -l

#SBATCH --account=s658
#SBATCH --job-name="${RUNFOLDER}"
#SBATCH --output=${RUNFOLDER}_out_%j.txt
#SBATCH --error=${RUNFOLDER}_err_%j.txt
#SBATCH --time=12:00:00
#SBATCH --nodes=${NMASTERS}
#SBATCH --ntasks-per-node=${NTASKPERNODE}
#SBATCH --constraint=gpu

# #SBATCH --constraint=mc
# #SBATCH --partition=debug
# #SBATCH --time=00:30:00
# #SBATCH --mail-user="${MYNAME}@ethz.ch"
# #SBATCH --mail-type=ALL

export OMP_NUM_THREADS=${NTHREADS}
export CRAY_CUDA_MPS=1
export OMP_PROC_BIND=CLOSE
export OMP_PLACES=cores

srun --ntasks ${NPROCESS} --threads-per-core=1 --cpu_bind=none --cpus-per-task=${NTHREADS} --ntasks-per-node=${NTASK} ./exec ${SETTINGS}
EOF

chmod 755 daint_sbatch

sbatch daint_sbatch
cd -
