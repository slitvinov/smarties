#!/bin/bash -l

COMMNAME=Racer_revisit_clip5

SETTINGS=" --gamma 0.995 --nnl1 128 --nnl2 128 --nnFunc Tanh --learner POAC --minTotObsNum 131072 --targetDelay 0 --greedyEps 0.5 --totNumSteps 10000000 --learnrate 0.0001 --nMasters 1 --nThreads 12 --ppn 2 "

MYNAME=`whoami`
BASEPATH="/scratch/snx3000/${MYNAME}/smarties/"

#SBATCH --job-name=${COMMNAME}
#SBATCH --time=24:00:00
#SBATCH --output=out.%j.%a.o
#SBATCH --error=out.%j.%a.e
#SBATCH --constraint=gpu

# #SBATCH --account=ch7
#SBATCH --account=s658
# #SBATCH --account=eth2

#SBATCH --array=0-179

# ======START=====
BUFFSIZE=(1048576 524288 262144)
IMPSAMPR=(2 4 8)
DKLPARAM=(0.10)
BATCHNUM=(256)
EPERSTEP=(1)
RUNTRIAL=(1 2 3 4 5)

#TASKNAME=(Humanoid-v2 Ant-v2 Walker2d-v2 HalfCheetah-v2 Swimmer-v2 Reacher-v2 Hopper-v2 HumanoidStandup-v2 InvertedPendulum-v2 InvertedDoublePendulum-v2)
#SHORTID=(humanw spider walker cheeta swimmr reachr hopper standu invpnd dblpnd)
TASKNAME=(Humanoid-v2 Ant-v2 Walker2d-v2 HalfCheetah-v2)
SHORTID=(humanw spider walker cheeta)

cases=()
rundir=()
appspec=()
#approximatively sorted by strongest effect on runtime
for it in `seq 0 ${#TASKNAME[@]}`; do

for b in ${BATCHNUM[@]}; do
for d in ${DKLPARAM[@]}; do
for n in ${BUFFSIZE[@]}; do
for o in ${EPERSTEP[@]}; do
for c in ${IMPSAMPR[@]}; do
for r in ${RUNTRIAL[@]}; do

cases+=("--maxTotObsNum $n --impWeight $c --klDivConstraint $d --batchSize $b --obsPerStep $o")
rundir+=("${SHORTID[$i]}_${COMMNAME}_R${c}_N${n}_D${d}_TRIAL${r}")
appspec+=("${TASKNAME[$i]}")

done
done
done
done
done
done

done

# stuff that depends on SLURM_ARRAY_TASK_ID
RUNFOLDER=${rundir[$SLURM_ARRAY_TASK_ID]}
mkdir -p ${BASEPATH}${RUNFOLDER}
RUNSET=${SETTINGS}${cases[$SLURM_ARRAY_TASK_ID]}
cat <<EOF >${BASEPATH}${RUNFOLDER}/launchSim.sh
python3 ../Communicator_gym.py \$1 ${appspec[$SLURM_ARRAY_TASK_ID]}
EOF
cat <<EOF >${BASEPATH}${RUNFOLDER}/factory
Environment exec=../launchSim.sh n=1
EOF
chmod +x ${BASEPATH}${RUNFOLDER}/launchSim.sh

# copy over stuff
cp $0                         ${BASEPATH}${RUNFOLDER}/launch.sh
cp ../makefiles/rl            ${BASEPATH}${RUNFOLDER}/exec
cp ../source/Communicator*.py ${BASEPATH}${RUNFOLDER}/
git log | head  > ${BASEPATH}${RUNFOLDER}/gitlog.log
git diff > ${BASEPATH}${RUNFOLDER}/gitdiff.log

cd ${BASEPATH}${RUNFOLDER}

export OMP_NUM_THREADS=${NTHREADS}
export OMP_PROC_BIND=CLOSE
export OMP_PLACES=cores
export CRAY_CUDA_MPS=1

srun --ntasks 2 --threads-per-core=2 --cpu_bind=sockets --cpus-per-task=12 --ntasks-per-node=2 ./exec ${RUNSET}

#srun  -n 2 --cpu_bind=map_cpu:0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17 ./run1 & srun  -n 2 --cpu_bind=map_cpu:18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35 ./run2 & wait

# =====END====
