PROGNAME=hyperion
SETTINGSNAME=settingsLearn.sh
EXECNAME=ProvaLearn
NTHREADS=24

MYNAME=`whoami`
BASEPATH="$HOME/Runs/"

if [ $3 = "kill" ] ; then
echo "Deleting folder."
rm -rf ${BASEPATH}${EXECNAME}
fi

if [ ! -d ${BASEPATH}${EXECNAME} ]; then
echo "Directory does not exist yet! Setting up simulation."
RESTART=" -restart 0"
source $SETTINGSNAME

factoryFile=`echo $SETTINGS | awk 'BEGIN{FS="-factory "} {print $2}' | awk '{print $1}'`
if [ -n "$factoryFile" ]; then
if [ ! -f $factoryFile ];then
echo "factoryFile "$factoryFile" not found! - exiting"
exit -1
fi
echo "factoryFile detected: "$factoryFile
fi

mkdir -p ${BASEPATH}${EXECNAME}
cp $0 ${BASEPATH}${EXECNAME}/
cp $SETTINGSNAME ${BASEPATH}${EXECNAME}/
#git fetch && git diff FETCH_HEAD > ${BASEPATH}${EXECNAME}/gitDiff.patch
cp ../makefiles/${PROGNAME} ${BASEPATH}${EXECNAME}/
if [ -n "$factoryFile" ]; then
cp $factoryFile ${BASEPATH}${EXECNAME}/
fi
env > ${BASEPATH}${EXECNAME}/environment.log

else
echo "Directory already exists! Restarting."
RESTART=" -restart 1"

# check if the settingsfile exist
settingsfile=${BASEPATH}${EXECNAME}"/"${SETTINGSNAME}
if [ ! -f $settingsfile ];then
echo ${settingsfile}" not found! - exiting"
exit -1
fi

echo "Using settings in "${settingsfile}
source $settingsfile

# check if executable exist
progfile=${BASEPATH}${EXECNAME}"/"${PROGNAME}
if [ ! -f $progfile ];then
echo ${progfile}" not found! - exiting"
exit -1
fi

# check if restart file exist
restartfile=${BASEPATH}${EXECNAME}"/restart.mrg"
if [ ! -f $restartfile ];then
echo ${restartfile}" not found! - exiting"
exit -1
fi
fi

SETTINGS+=" -nthreads ${NTHREADS}"
OPTIONS=${SETTINGS}${RESTART}

cd ${BASEPATH}${EXECNAME}

TBB_INSTALL_DIR=$HOME/tbbsrc
TBB_LIB_DIR=$TBB_INSTALL_DIR/build/linux_intel64_gcc_cc4.8.3_libc2.17_kernel3.10.0_release
VTK_LIB_DIR=/usr/lib64/vtk/

export LD_LIBRARY_PATH=$TBB_LIB_DIR/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$VTK_LIB_DIR/:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=${NTHREADS}

./hyperion ${OPTIONS}