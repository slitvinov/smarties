make -C ../apps/predator_prey clean
make -C ../apps/predator_prey

cp ../apps/predator_prey/launch.sh ${BASEPATH}${RUNFOLDER}/launchSim.sh
cp ../apps/predator_prey/pp        ${BASEPATH}${RUNFOLDER}/
cp ../apps/predator_prey/pp.py     ${BASEPATH}${RUNFOLDER}/
cp ../source/Communicators/Communicator.py ${BASEPATH}${RUNFOLDER}/

cat <<EOF >${BASEPATH}${RUNFOLDER}/appSettings.sh
SETTINGS+=" --bSharedPol 0 --appendedObs 1"
EOF
chmod +x ${BASEPATH}${RUNFOLDER}/appSettings.sh
