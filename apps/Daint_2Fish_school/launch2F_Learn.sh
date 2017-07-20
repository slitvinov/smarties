SOCK=$1
PROGNAME=hyperion
SETTINGSNAME=settings2F_Learn.sh
EXECNAME=LearningSim
NTHREADS=12

#module load gcc/4.9.2
#module purge
#module new gcc/5.2.0

MYNAME=`whoami`
BASEPATH="./"

    RESTART=" -restart 0"
mkdir -p ${BASEPATH}${EXECNAME}

echo "Setting up simulation."

source ../$SETTINGSNAME

factoryFile=`echo $SETTINGS | awk 'BEGIN{FS="-factory "} {print $2}' | awk '{print $1}'`
if [ -n "$factoryFile" ]; then
    if [ ! -f $factoryFile ];then
        echo "factoryFile "$factoryFile" not found! - exiting"
        exit -1
    fi
    echo "factoryFile detected: "$factoryFile
fi

cp ../launchHere.sh ${BASEPATH}${EXECNAME}/
cp ../$SETTINGSNAME ${BASEPATH}${EXECNAME}/
cp ../hyperion ${BASEPATH}${EXECNAME}/

if [ -n "$factoryFile" ]; then
cp $factoryFile ${BASEPATH}${EXECNAME}/
fi
env > ${BASEPATH}${EXECNAME}/environment.log


SETTINGS+=" -nthreads ${NTHREADS}"
SETTINGS+=" -sock ${SOCK}"
OPTIONS=${SETTINGS}${RESTART}

cd ${BASEPATH}${EXECNAME}
export TMPDIR=/cluster/scratch_xp/public/novatig/

#export LD_LIBRARY_PATH=${HOME}/2d-treecodes-ispc/p6_f12_mp/:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=${HOME}/2d-treecodes-ispc/:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/cluster/work/infk/wvanrees/apps/TBB/tbb42_20140416oss/build/linux_intel64_gcc_cc4.7.2_libc2.12_kernel2.6.32_release/:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/cluster/home03/mavt/novatig/tbb2017_20160916oss/build/linux_intel64_gcc_cc5.2.0_libc2.12_kernel2.6.32_release/:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/cluster/home03/mavt/novatig/tbb2017_20160916oss/build/linux_intel64_gcc_cc4.9.2_libc2.12_kernel2.6.32_release/:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/cluster/work/infk/cconti/VTK5.8_gcc/lib/vtk-5.8/:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=${HOME}/2d-treecodes-ispc/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${HOME}/tbb2017/build/linux_intel64_gcc_cc5.3.0_libc2.19_kernel3.12.60_release:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/users/sverma/usr/VTK-7.1.1/lib/:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=${NTHREADS}
./hyperion ${OPTIONS}

#valgrind  --tool=memcheck  --track-origins=yes ./hyperion ${OPTIONS}
