#!/bin/bash
RUNFOLDER=$1
NNODES=$2
APP=$3
SETTINGSNAME=$4

if [ $# -lt 4 ] ; then
	echo "Usage: ./launch_openai.sh RUNFOLDER MPI_NODES APP SETTINGS_PATH (POLICY_PATH) (N_MPI_TASK_PER_NODE OMP_THREADS)"
	exit 1
fi

MYNAME=`whoami`
BASEPATH="/scratch/snx3000/${MYNAME}/smarties/"
mkdir -p ${BASEPATH}${RUNFOLDER}
ulimit -c unlimited
#lfs setstripe -c 1 ${BASEPATH}${RUNFOLDER}

if [ $# -gt 4 ] ; then
    POLICY=$5
		if [ -f ${POLICY}_net.raw ] ; then
			cp ${POLICY}_net.raw ${BASEPATH}${RUNFOLDER}/policy_net.raw
		elif [ -f ${POLICY}_net ] ; then
			cp ${POLICY}_net ${BASEPATH}${RUNFOLDER}/policy_net
		else
			echo "Cannot find saved network file."
			exit 1
		fi
    cp ${POLICY}_data_stats ${BASEPATH}${RUNFOLDER}/policy_data_stats
		cp ${POLICY}.status ${BASEPATH}${RUNFOLDER}/policy.status
fi
if [ $# -lt 7 ] ; then
    NTASK=1 #n tasks per node
    NTHREADS=24 #n threads per task
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

#this handles app-side setup (incl. copying the factory)
#this must handle all app-side setup (as well as copying the factory)
if [ -d ${APP} ]; then
	if [ -x ${APP}/setup.sh ] ; then
		source ${APP}/setup.sh ${BASEPATH}${RUNFOLDER}
	else
		echo "${APP}/setup.sh does not exist or I cannot execute it"
		exit 1
	fi
else
	if [ -x ../apps/${APP}/setup.sh ] ; then
		source ../apps/${APP}/setup.sh ${BASEPATH}${RUNFOLDER}
	else
		echo "../apps/${APP}/setup.sh does not exist or I cannot execute it"
		exit 1
	fi
fi

cp ../makefiles/rl ${BASEPATH}${RUNFOLDER}/rl
if [ ! -x ../makefiles/rl ] ; then
	echo "../makefiles/rl not found! - exiting"
	exit 1
fi
cp ${SETTINGSNAME} ${BASEPATH}${RUNFOLDER}/settings.sh
cp ${SETTINGSNAME} ${BASEPATH}${RUNFOLDER}/policy_settings.sh
cp $0 ${BASEPATH}${RUNFOLDER}/launch_smarties.sh

cd ${BASEPATH}${RUNFOLDER}
if [ ! -f settings.sh ] ; then
    echo ${SETTINGSNAME}" not found! - exiting"
    exit 1
fi
source settings.sh
SETTINGS+=" --nThreads ${NTHREADS}"
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
# #SBATCH --time=00:30:00
# #SBATCH --partition=debug
#SBATCH --nodes=${NNODES}
#SBATCH --ntasks-per-node=${NTASK}
#SBATCH --cpus-per-task=$((${NTHREADS}/2)) # Hyperthreaded
#SBATCH --constraint=gpu
#SBATCH --mail-user="${MYNAME}@ethz.ch"
#SBATCH --mail-type=ALL

module load daint-gpu
export OMP_NUM_THREADS=${NTHREADS}
export CRAY_CUDA_MPS=1

srun --ntasks ${NPROCESS} --cpu_bind=none --ntasks-per-node=${NTASK} --cpus-per-task=$((${NTHREADS}/2)) --threads-per-core=2 ./exec ${SETTINGS}

#srun --ntasks ${NPROCESS} --cpu_bind=none --ntasks-per-node=${NTASK} --cpus-per-task=$((${NTHREADS}/2)) --threads-per-core=2 valgrind  --tool=memcheck  --leak-check=full --show-reachable=no --show-possibly-lost=no --track-origins=yes ./exec ${SETTINGS}
EOF

chmod 755 daint_sbatch
sbatch daint_sbatch
