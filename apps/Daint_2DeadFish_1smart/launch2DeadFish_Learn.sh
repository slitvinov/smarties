SOCK=$1
PROGNAME=hyperion
SETTINGSNAME=deadSettings.sh
EXECNAME=LearningSim
NTHREADS=12

MYNAME=`whoami`
BASEPATH="./"

RESTART=" -restart 0"
mkdir -p ${BASEPATH}${EXECNAME}
source ../$SETTINGSNAME

factoryFile=`echo $SETTINGS | awk 'BEGIN{FS="-factory "} {print $2}' | awk '{print $1}'`
if [ -n "$factoryFile" ]; then
if [ ! -f $factoryFile ];then
echo "factoryFile "$factoryFile" not found! - exiting"
exit -1
fi
echo "factoryFile detected: "$factoryFile
fi

#cp $HOME/MRAGapps/IF2D_ROCKS/launch/restart_learn/* ${BASEPATH}${EXECNAME}/
cp ../launchSim.sh ${BASEPATH}${EXECNAME}/
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
export LD_LIBRARY_PATH=${HOME}/2d-treecodes/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${HOME}/tbb2017/build/linux_intel64_gcc_cc5.3.0_libc2.19_kernel3.12.60_release:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=${NTHREADS}

./hyperion ${OPTIONS}
