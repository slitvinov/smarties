SOCK=$1
PROGNAME=hyperion
SETTINGSNAME=deadSettings.sh
EXECNAME=LearningSim
NTHREADS=24

MYNAME=`whoami`
BASEPATH="./"

RESTART=" -restart 0"
mkdir -p ${BASEPATH}${EXECNAME}

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

cp ../launchSim.sh ${BASEPATH}${EXECNAME}/
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
export LD_LIBRARY_PATH=/cluster/home/novatig/2d-treecodes/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/cluster/home/novatig/tbb2017/build/linux_intel64_gcc_cc4.9.2_libc2.12_kernel2.6.32_release:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/cluster/home/novatig/VTK-7.1.0/Build/lib/:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=${NTHREADS}

./hyperion ${OPTIONS}
module load valgrind
#valgrind  --num-callers=100  --tool=memcheck  --leak-check=yes  --track-origins=yes --show-reachable=yes ./hyperion ${OPTIONS}
