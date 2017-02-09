#!/bin/bash
NNODESX=$1
NNODESY=$2
NNODES=$((${1}*${2}))
cp ../factory2Stefans ./factory
echo "starting job "${RUNFOLDER}" with "${NNODES}" processors"
if [ ! -f ../settings2Stefans.sh ];then
    echo "../settings2Stefans.sh not found! - exiting"
    exit -1
fi
source ../settings2Stefans.sh
OPTIONS+=" -sock 0"
OPTIONS+=" -nprocsx ${NNODESX}"
OPTIONS+=" -nprocsy ${NNODESY}"

echo $OPTIONS > settings.txt

cat <<EOF >daint_sbatch
#!/bin/bash -l

#SBATCH --account=s658
#SBATCH --job-name="${RUNFOLDER}"
#SBATCH --output=${RUNFOLDER}_%j.txt
#SBATCH --error=${RUNFOLDER}_%j.txt
#SBATCH --time=24:00:00
#SBATCH --nodes=${NNODES}
# #SBATCH --partition=viz
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --constraint=gpu

#module swap PrgEnv-cray PrgEnv-gnu
#export LD_LIBRARY_PATH=/users/novatig/accfft/build_dbg/:$LD_LIBRARY_PATH
module load daint-gpu GSL/2.1-CrayGNU-2016.11 cray-hdf5-parallel/1.10.0
module load cudatoolkit/8.0.44_GA_2.2.7_g4a6c213-2.1 fftw/3.3.4.10
export OMP_NUM_THREADS=24
export MYROUNDS=10000
export USEMAXTHREADS=1
export CRAY_CUDA_MPS=1

srun -n ${NNODES} --ntasks-per-node=1 --cpus-per-task=24 ../execSim ${OPTIONS}
EOF

chmod 755 daint_sbatch
sbatch daint_sbatch
