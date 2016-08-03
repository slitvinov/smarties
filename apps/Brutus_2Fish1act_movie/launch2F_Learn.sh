SOCK=$1
PROGNAME=hyperion
SETTINGSNAME=settings2F_Learn.sh
EXECNAME=LearningSim
NTHREADS=48

module load gcc/4.9.2

MYNAME=`whoami`
BASEPATH="./"

settingsfile=${BASEPATH}${EXECNAME}"/"${SETTINGSNAME}
if [ ! -f $settingsfile ];then
    echo "Deleting folder."
    rm -rf ${BASEPATH}${EXECNAME}
fi

# check if restart file exist
restartfile=${BASEPATH}${EXECNAME}"/restart.mrg"
if [ ! -f $restartfile ];then
    echo "Deleting folder."
    rm -rf ${BASEPATH}${EXECNAME}
fi

if [ ! -d ${BASEPATH}${EXECNAME} ]; then
    echo "Directory does not exist yet! Setting up simulation."
    RESTART=" -restart 0"
    mkdir -p ${BASEPATH}${EXECNAME}
else
    echo "Directory already exists! Restarting."
    RESTART=" -restart 1"
fi

echo "Setting up simulation."

source ../$SETTINGSNAME

factoryFile=`echo $SETTINGS | awk 'BEGIN{FS="-factory "} {print $2}' | awk '{print $1}'`
if [ -n "$factoryFile" ]; then
    if [ ! -f ../$factoryFile ];then
        echo "factoryFile "$factoryFile" not found! - exiting"
        exit -1
    fi
    echo "factoryFile detected: "../$factoryFile
fi

cp ../launch2F_Learn.sh ${BASEPATH}${EXECNAME}/
cp ../$SETTINGSNAME ${BASEPATH}${EXECNAME}/
cp ../hyperion ${BASEPATH}${EXECNAME}/

if [ -n "$factoryFile" ]; then
cp ../$factoryFile ${BASEPATH}${EXECNAME}/
fi
env > ${BASEPATH}${EXECNAME}/environment.log


SETTINGS+=" -nthreads ${NTHREADS}"
SETTINGS+=" -sock ${SOCK}"
OPTIONS=${SETTINGS}${RESTART}

cd ${BASEPATH}${EXECNAME}


export LD_LIBRARY_PATH=${HOME}/2d-treecodes-ispc/p6_f12_mp/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/cluster/work/infk/wvanrees/apps/TBB/tbb42_20140416oss/build/linux_intel64_gcc_cc4.7.2_libc2.12_kernel2.6.32_release/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/cluster/work/infk/cconti/VTK5.8_gcc/lib/vtk-5.8/:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=${NTHREADS}

./hyperion ${OPTIONS}
