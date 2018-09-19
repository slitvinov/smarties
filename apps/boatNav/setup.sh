make -C ../apps/boatNav clean
make -C ../apps/boatNav

cp ../apps/boatNav/launch.sh ${BASEPATH}${RUNFOLDER}/launchSim.sh
cp ../apps/boatNav/boatNav   ${BASEPATH}${RUNFOLDER}/
cp ../source/Communicators/Communicator.py ${BASEPATH}${RUNFOLDER}/

#SETTINGS+=" --appendedObs 1"
cat <<EOF >${BASEPATH}${RUNFOLDER}/appSettings.sh
SETTINGS+=" --bSharedPol 0"
EOF
chmod +x ${BASEPATH}${RUNFOLDER}/appSettings.sh
