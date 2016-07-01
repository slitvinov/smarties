SOCK=$1
PROGNAME=hyperion
SETTINGSNAME=settingsDcyl_Fish.sh
EXECNAME=LearningSim
NTHREADS=8

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

#cp $HOME/MRAGapps/IF2D_ROCKS/launch/restart_learn/* ${BASEPATH}${EXECNAME}/
cp ../launchHere.sh ${BASEPATH}${EXECNAME}/
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

export LD_LIBRARY_PATH=/opt/intel/13.0.1.117/composer_xe_2013.1.117/tbb/lib/intel64/:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=${NTHREADS}

./hyperion ${OPTIONS}
