PROGNAME=rl
SETTINGSNAME=settingsLearn.sh
TIMES=1

NPROCESS=1
EXECNAME=$2

WCLOCK=08:00

if [ $# -lt 6 ] ; then
NTHREADS=48
echo "Setting NTHREADS to ${NTHREADS}"
else
NTHREADS=$6
fi

MYNAME=`whoami`
BASEPATH="/cluster/scratch_xp/public/"${MYNAME}"/MRAG2D/"

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

export LD_LIBRARY_PATH=/cluster/work/infk/wvanrees/apps/TBB/tbb42_20140416oss/build/linux_intel64_gcc_cc4.7.2_libc2.12_kernel2.6.32_release/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/cluster/work/infk/cconti/VTK5.8_gcc/lib/vtk-5.8/:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=${NTHREADS}

echo "Submission 0..."
bsub -J ${EXECNAME} -n ${NTHREADS} -W ${WCLOCK} -o out ${NUMALINE} time ./${PROGNAME} ${OPTIONS}

RESTART=" -restart 1"
OPTIONS=${SETTINGS}${RESTART}
for (( c=1; c<=${TIMES}-1; c++ ))
do
echo "Submission $c..."
bsub -J ${EXECNAME} -n ${NTHREADS} -w "ended(${EXECNAME})" -W ${WCLOCK} -o out ${NUMALINE} time ./${PROGNAME} ${OPTIONS}
done