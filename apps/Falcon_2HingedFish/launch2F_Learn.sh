SOCK=$1
PROGNAME=hyperion
SETTINGSNAME=settings2F_Learn.sh
EXECNAME=LearningSim
NTHREADS=1

MYNAME=`whoami`
BASEPATH="./"

RESTART="-restart\n0"
mkdir -p ${BASEPATH}${EXECNAME}
echo "Setting up simulation."

source ../$SETTINGSNAME

#factoryFile=`echo $SETTINGS | awk 'BEGIN{FS="-factory "} {print $2}' | awk '{print $1}'`
factoryFile=`echo $SETTINGS | awk 'BEGIN{FS="\n-factory\n"} {print $2}' | awk '{print $1}'`
if [ -n "$factoryFile" ]; then
    if [ ! -f $factoryFile ];then
        echo "factoryFile "$factoryFile" not found! - exiting"
        exit -1
    fi
    echo "factoryFile detected: "$factoryFile
fi

cp ../launchSim.sh ${BASEPATH}${EXECNAME}/
cp ../$SETTINGSNAME ${BASEPATH}${EXECNAME}/
cp ../hyperion ${BASEPATH}${EXECNAME}/

if [ -n "$factoryFile" ]; then
cp $factoryFile ${BASEPATH}${EXECNAME}/
fi
env > ${BASEPATH}${EXECNAME}/environment.log


SETTINGS+="\n-nthreads\n${NTHREADS}"
SETTINGS+="\n-sock\n${SOCK}"
#OPTIONS=${SETTINGS}${RESTART}
OPTIONS=${RESTART}${SETTINGS}

cp ../factory2F_Learn .

echo -e ${OPTIONS} > argList.txt
cd ${BASEPATH}${EXECNAME}

cp ../factory2F_Learn .

export LD_LIBRARY_PATH=/opt/tbb/lib/intel64/gcc4.4/:${HOME}/2d-treecodes/:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=${NTHREADS}

echo -e ${OPTIONS} > argList.txt

#./hyperion ${OPTIONS}
