#!/bin/bash
EXECNAME=rl
RUNFOLDER=$1
APP=$2
SETTINGSNAME=$3

MYNAME=`whoami`
BASEPATH="${SCRATCH}/smarties/"

mkdir -p ${BASEPATH}${RUNFOLDER}
ulimit -c unlimited

NTHREADS=12 #n master ranks

cat <<EOF >${BASEPATH}${RUNFOLDER}/launchSim.sh
python3 ../Communicator_gym.py \$1 $APP
EOF
chmod +x ${BASEPATH}${RUNFOLDER}/launchSim.sh

cp ../../makefiles/${EXECNAME} ${BASEPATH}${RUNFOLDER}/exec
cp ../../source/Communicators/Communicator.py     ${BASEPATH}${RUNFOLDER}/
cp ../../source/Communicators/Communicator_gym.py ${BASEPATH}${RUNFOLDER}/
cp ${SETTINGSNAME} ${BASEPATH}${RUNFOLDER}/settings.sh
cp $0 ${BASEPATH}${RUNFOLDER}/launch.sh
git log | head  > ${BASEPATH}${RUNFOLDER}/gitlog.log
git diff HEAD > ${BASEPATH}${RUNFOLDER}/gitdiff.log

cd ${BASEPATH}${RUNFOLDER}
if [ ! -f settings.sh ];then
    echo ${SETTINGSNAME}" not found! - exiting"
    exit -1
fi
source settings.sh
SETTINGS+=" --nMasters 1"
SETTINGS+=" --nWorkers 1"
SETTINGS+=" --nThreads 12"

echo $SETTINGS > settings.txt
echo ${SETTINGS}

cat <<EOF >daint_sbatch
#!/bin/bash -l

#SBATCH --account=s658
# #SBATCH --account=eth2
#SBATCH --job-name="${RUNFOLDER}"
#SBATCH --output=${RUNFOLDER}_out_%j.txt
#SBATCH --error=${RUNFOLDER}_err_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu

#SBATCH --time=24:00:00
# #SBATCH --partition=debug
# #SBATCH --time=00:30:00
# #SBATCH --mail-user="${MYNAME}@ethz.ch"
# #SBATCH --mail-type=ALL

export MPICH_MAX_THREAD_SAFETY=multiple
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=12
export OMP_PROC_BIND=CLOSE
export OMP_PLACES=cores
export CRAY_CUDA_MPS=1

srun --n 1 --nodes=1 --ntasks-per-node=1 ./exec ${SETTINGS}

EOF

chmod 755 daint_sbatch

sbatch daint_sbatch
cd -


