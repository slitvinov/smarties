#!/bin/bash
#
#  smarties
#  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
#  Distributed under the terms of the MIT license.
#
#  Created by Guido Novati (novatig@ethz.ch).
#
RUNFOLDER=$1

if [ $# -lt 2 ] ; then
	echo "Usage: ./launch_base.sh RUNFOLDER ENVIRONMENT_APP (SETTINGS_PATH default is 'settings/settings_VRACER.sh' ) (NTHREADS default read from system, but unrealiable on clusters) (NNODES default 1) (NMASTERS default 1) (NWORKERS default 1)"
	exit 1
fi
HOST=`hostname`

if [ $# -gt 2 ] ; then export SETTINGSNAME=$3
else export SETTINGSNAME=settings/settings_VRACER.sh
fi

################################################################################
########### FIRST, read cores per node and available number of nodes ###########
################################################################################
if [ $# -gt 3 ] ; then export NTHREADS=$4
else
################################################################################
if [ ${HOST:0:5} == 'euler' ] || [ ${HOST:0:3} == 'eu-' ] ; then
export NTHREADS=36
elif [ ${HOST:0:5} == 'daint' ] ; then
export NTHREADS=12
else
export NTHREADS=$([[ $(uname) = 'Darwin' ]] && sysctl -n hw.physicalcpu_max || lscpu -p | egrep -v '^#' | sort -u -t, -k 2,4 | wc -l)
fi
################################################################################
fi

if [ $# -gt 4 ] ; then export NNODES=$5
else export NNODES=1
fi

################################################################################
######## SECOND, depending on environment type, specify distribution of ########
####### resources between learning (master) ranks and env (worker) ranks #######
################################################################################

################################################################################
if [[ "${INTERNALAPP}" == "true" ]] ; then # WORKERS RUN ON DEDICATED MPI RANKS
################################################################################

# if distribution not specified, assume we want as many workers as possible
if [ $# -gt 5 ] ; then export NMASTERS=$6
else export NMASTERS=1
fi
if [ $# -gt 6 ] ; then export NWORKERS=$7
elif [ $NNODES -gt 1 ] ; then export NWORKERS=$(( $NNODES - $NMASTERS ))
else export NWORKERS=1
fi

# BOTH MASTERS AND WORKERS ARE CREATED DURING INITIAL MPIRUN CALL:
NPROCESSES=$(( $NWORKERS + $NMASTERS ))

################################################################################
else # THEN WORKERS ARE FORKED PROCESSES RUNNING ON SAME CORES AS MASTERS
################################################################################

# if distribution not specified, assume we want as many masters as possible:
if [ $# -gt 5 ] ; then export NMASTERS=$6
else export NMASTERS=$NNODES
fi
# assume we fork one env process per master mpi rank:
if [ $# -gt 6 ] ; then export NWORKERS=$7
else export NWORKERS=$NNODES
fi

# ONLY MASTERS ARE INCLUDED AMONG PROCESSES IN INITIAL MPIRUN CALL:
NPROCESSES=$NMASTERS

################################################################################
fi # END OF LOGIC ON ${INTERNALAPP} AND RESOURCES DISTRIBUTION
################################################################################

# Compute number of processes running on each node:
ZERO=$(( $NPROCESSES % $NNODES ))
if [ $ZERO != 0 ] ; then
echo "ERROR: unable to map NWORKERS and NMASTERS onto NNODES"
exit 1
fi
export NPROCESSPERNODE=$(( $NPROCESSES / $NNODES ))
echo "NWORKERS:"$NWORKERS "NMASTERS:"$NMASTERS "NNODES:"$NNODES "NPROCESSPERNODE:"$NPROCESSPERNODE

################################################################################
############################## PREPARE RUNFOLDER ###############################
################################################################################
if [ ! -x ../makefiles/rl ] ; then
	echo "../makefiles/rl not found! - exiting"
	exit 1
fi
cp ../makefiles/rl ${BASEPATH}${RUNFOLDER}/rl
cp $0 ${BASEPATH}${RUNFOLDER}/launch_smarties.sh
cp ${SETTINGSNAME} ${BASEPATH}${RUNFOLDER}/settings.sh
git log | head  > ${BASEPATH}${RUNFOLDER}/gitlog.log
git diff > ${BASEPATH}${RUNFOLDER}/gitdiff.log

################################################################################
############################### PREPARE HPC ENV ################################
################################################################################
unset LSB_AFFINITY_HOSTFILE #euler cluster
export MPICH_MAX_THREAD_SAFETY=multiple #MPICH
export MV2_ENABLE_AFFINITY=0 #MVAPICH
export OMP_NUM_THREADS=${NTHREADS}
export OPENBLAS_NUM_THREADS=1
export CRAY_CUDA_MPS=1
################################################################################

cd ${BASEPATH}${RUNFOLDER}

################################################################################
############################ READ SMARTIES SETTINGS ############################
################################################################################
SETTINGSNAME=settings.sh
if [ ! -f $SETTINGSNAME ] ; then
    echo ${SETTINGSNAME}" not found! - exiting" ; exit -1
fi
source $SETTINGSNAME
if [ -x appSettings.sh ]; then source appSettings.sh ; fi
SETTINGS+=" --nWorkers ${NWORKERS}"
SETTINGS+=" --nMasters ${NMASTERS}"
SETTINGS+=" --nThreads ${NTHREADS}"
if [ "${INTERNALAPP}" == "true" ] ; then SETTINGS+=" --runInternalApp 1"
else SETTINGS+=" --runInternalApp 0"
fi

################################################################################
#################################### EULER #####################################
################################################################################
if [ ${HOST:0:5} == 'euler' ] || [ ${HOST:0:3} == 'eu-' ] ; then

# override trick to run without calling bsub:
if [ "${RUNLOCAL}" == "true" ] ; then
mpirun -n ${NPROCESSES} --map-by ppr:${NPROCESSPERNODE}:node ./rl ${SETTINGS} | tee out.log
fi

WCLOCK=${WCLOCK:-24:00}
# compute the number of CPU CORES to ask euler:
export NPROCESSORS=$(( ${NNODES} * ${NTHREADS} ))

bsub -n ${NPROCESSORS} -J ${RUNFOLDER} -R "select[model==XeonGold_6150] span[ptile=${NTHREADS}]" -W ${WCLOCK} mpirun -n ${NPROCESSES} --map-by ppr:${NPROCESSPERNODE}:node ./rl ${SETTINGS} | tee out.log

################################################################################
#################################### DAINT #####################################
################################################################################
elif [ ${HOST:0:5} == 'daint' ] ; then

WCLOCK=${WCLOCK:-24:00:00}
# did we allocate a node?
srun hostname &> /dev/null
if [[ "$?" -gt "0" ]] ; then # no we did not. call sbatch:

cat <<EOF >daint_sbatch
#!/bin/bash -l
#SBATCH --account=s929 --job-name="${RUNFOLDER}" --time=${WCLOCK}
#SBATCH --output=${RUNFOLDER}_out_%j.txt --error=${RUNFOLDER}_err_%j.txt
#SBATCH --nodes=${NNODES} --constraint=gpu
srun -n ${NPROCESSES} --nodes=${NNODES}  --ntasks-per-node=${NPROCESSPERNODE} ./rl ${SETTINGS}
EOF

chmod 755 daint_sbatch
sbatch daint_sbatch

else

srun -n ${NPROCESSES} --nodes ${NNODES} --ntasks-per-node ${NPROCESSPERNODE} ./rl ${SETTINGS}

fi

################################################################################
############################## LOCAL WORKSTATION ###############################
################################################################################
else

mpirun -n ${NPROCESSES} --map-by ppr:${NPROCESSPERNODE}:node ./rl ${SETTINGS} | tee out.log

fi
