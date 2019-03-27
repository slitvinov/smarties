#!/bin/bash
#
#  smarties
#  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
#  Distributed under the terms of the MIT license.
#
#  Created by Guido Novati (novatig@ethz.ch).
#
RUNFOLDER=$1
NNODES=$2
APP=$3
SETTINGSNAME=$4
if [ $# -gt 4 ] ; then #n worker ranks
export NWORKERS=$5
else
export NWORKERS=$(($NNODES-1))
fi
if [ $# -gt 5 ] ; then
export NMASTERS=$6
else
export NMASTERS=1 #n master ranks
fi

if [ $# -lt 5 ] ; then
	echo "Usage: ./launch_openai.sh RUNFOLDER NDAINT_NODES APP SETTINGS_PATH (NWORKERS) (NMASTERS)"
	exit 1
fi

NPROCESSES=$(( $NWORKERS + $NMASTERS ))
ZERO=$(( $NPROCESSES % $NNODES ))
if [ $ZERO != 0 ] ; then
echo "ERROR: unable to map NWORKERS and NMASTERS onto NNODES"
exit 1
fi
NPROCESSPERNODE=$(( $NPROCESSES / $NNODES ))
echo "NWORKERS:"$NWORKERS "NMASTERS:"$NMASTERS "NNODES:"$NNODES "NPROCESSPERNODE:"$NPROCESSPERNODE

WCLOCK=${WCLOCK:-24:00:00}

MYNAME=`whoami`
BASEPATH="${SCRATCH}/smarties/"
mkdir -p ${BASEPATH}${RUNFOLDER}

NTHREADS=12

#this handles app-side setup (incl. copying the factory)
#this must handle all app-side setup (as well as copying the factory)
if [ -d ${APP} ]; then
	if [ -x ${APP}/setup.sh ] ; then
		source ${APP}/setup.sh ${BASEPATH}${RUNFOLDER}
	else
		echo "${APP}/setup.sh does not exist or I cannot execute it"
		exit 1
	fi
elif [ -f ../apps/${APP}/setup.sh ]; then
	if [ -x ../apps/${APP}/setup.sh ] ; then
		source ../apps/${APP}/setup.sh ${BASEPATH}${RUNFOLDER}
	else
		echo "../apps/${APP}/setup.sh does not exist or I cannot execute it"
		exit 1
	fi
else
echo "Assumed" ${APP} "is an OpenAI Gym application"
cp ../source/Communicators/Communicator.py     ${BASEPATH}${RUNFOLDER}/
cp ../source/Communicators/Communicator_gym.py ${BASEPATH}${RUNFOLDER}/

cat <<EOF >${BASEPATH}${RUNFOLDER}/launchSim.sh
python3 ../Communicator_gym.py \$1 ${APP}
EOF

chmod +x ${BASEPATH}${RUNFOLDER}/launchSim.sh
fi

cp ../makefiles/rl ${BASEPATH}${RUNFOLDER}/rl
if [ ! -x ../makefiles/rl ] ; then
	echo "../makefiles/rl not found! - exiting"
	exit 1
fi
cp ${SETTINGSNAME} ${BASEPATH}${RUNFOLDER}/settings.sh
cp ${SETTINGSNAME} ${BASEPATH}${RUNFOLDER}/policy_settings.sh
cp $0 ${BASEPATH}${RUNFOLDER}/launch_smarties.sh
git log | head  > ${BASEPATH}${RUNFOLDER}/gitlog.log
git diff > ${BASEPATH}${RUNFOLDER}/gitdiff.log

cd ${BASEPATH}${RUNFOLDER}
if [ ! -f settings.sh ] ; then
    echo ${SETTINGSNAME}" not found! - exiting"
    exit 1
fi
source settings.sh
if [ -x appSettings.sh ]; then
source appSettings.sh
fi

SETTINGS+=" --nWorkers ${NWORKERS}"
SETTINGS+=" --nThreads ${NTHREADS}"
SETTINGS+=" --nMasters ${NMASTERS}"
#export OMP_PROC_BIND=TRUE
#export OMP_PLACES=cores

echo $SETTINGS > settings.txt
echo ${SETTINGS}

cat <<EOF >daint_sbatch
#!/bin/bash -l

#SBATCH --account=ch7
#SBATCH --job-name="${RUNFOLDER}"
#SBATCH --output=${RUNFOLDER}_out_%j.txt
#SBATCH --error=${RUNFOLDER}_err_%j.txt
#SBATCH --time=${WCLOCK}
#SBATCH --nodes=${NNODES}
#SBATCH --constraint=gpu

# #SBATCH --time=24:00:00
# #SBATCH --partition=debug
# #SBATCH --constraint=mc

# #SBATCH --mail-user="${MYNAME}@ethz.ch"
# #SBATCH --mail-type=ALL

export MPICH_MAX_THREAD_SAFETY=multiple
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=12
export CRAY_CUDA_MPS=1

srun -n ${NNODES} --nodes=${NNODES}  --ntasks-per-node=${NPROCESSPERNODE} ./rl ${SETTINGS}
EOF

chmod 755 daint_sbatch
sbatch daint_sbatch
