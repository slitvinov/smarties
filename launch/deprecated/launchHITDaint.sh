#!/bin/bash
#
#  smarties
#  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
#  Distributed under the terms of the MIT license.
#
#  Created by Guido Novati (novatig@ethz.ch).
#
BPDX=$1
RUNFOLDER=$2
APP=$3
SETTINGSNAME=$4

if [ $# -lt 4 ] ; then
    echo "Usage: ./launchChannelDaint.sh BPDX RUNFOLDER APP SETTINGS_PATH"
    echo "        BPDX must be set to {4, 8, 16, 32 ...}"
	exit 1
fi

WCLOCK=${WCLOCK:-24:00:00}

MYNAME=`whoami`
BASEPATH="${SCRATCH}/smarties/"
mkdir -p ${BASEPATH}${RUNFOLDER}

##################### CubismUP3D settings #####################
# BPD={BPDX, BPDX, BPDX/2}
BPDY=${BPDX}
BPDZ=${BPDX}
# NNODES_CUP={BPDX/2, BPDX/2, BPDX/4}
NNODEX=1
NNODEY=1
NNODEZ=1

###################### Smarties settings ######################
NWORKERS=$(($NNODEX * $NNODEY * $NNODEZ))
NMASTERS=1

if [ $NWORKERS -eq 0 ] ; then
    echo "Usage: ./launchChannelDaint.sh BPDX RUNFOLDER APP SETTINGS_PATH"
    exit 1
fi

######################## Daint settings #######################
NTASKPERNODE=1
NTHREADS=12
NNODES=$(($NMASTERS + $NWORKERS))

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
    echo "app does not exist or I cannot execute it"
    exit 1
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

echo $SETTINGS > settings.txt
echo ${SETTINGS}

cat <<EOF >daint_sbatch
#!/bin/bash -l

#SBATCH --account=s929
#SBATCH --job-name="${RUNFOLDER}"
#SBATCH --output=${RUNFOLDER}_out_%j.txt
#SBATCH --error=${RUNFOLDER}_err_%j.txt
#SBATCH --time=${WCLOCK}
#SBATCH --nodes=${NNODES}
#SBATCH --constraint=gpu

# #SBATCH --partition=debug
# #SBATCH --constraint=mc
# #SBATCH --mail-user="{MYNAME}@ethz.ch"
# #SBATCH --mail-type=ALL

export MPICH_MAX_THREAD_SAFETY=multiple
export OPENBLAS_NUM_THREADS=1
export OMP_PROC_BIND=TRUE
export OMP_NUM_THREADS=${NTHREADS}
export OMP_PLACES=cores
export CRAY_CUDA_MPS=1
export GASNET_USE_UDREG=0

srun -n $((${NNODES}*${NTASKPERNODE})) --nodes=${NNODES} --ntasks-per-node=${NTASKPERNODE} ./rl ${SETTINGS}
EOF

chmod 755 daint_sbatch
sbatch daint_sbatch
