cp ../apps/Daint_2Fish_3D/factory             ${BASEPATH}${RUNFOLDER}/
cp ../apps/Daint_2Fish_3D/settings_32.txt     ${BASEPATH}${RUNFOLDER}/
cp ../apps/Daint_2Fish_3D/settings_64.txt     ${BASEPATH}${RUNFOLDER}/
cp ../apps/Daint_2Fish_3D/factory2Stefans     ${BASEPATH}${RUNFOLDER}/
cp ../apps/Daint_2Fish_3D/runDaint.sh         ${BASEPATH}${RUNFOLDER}/launchSim.sh
cp ../apps/Daint_2Fish_3D/settings2Stefans.sh ${BASEPATH}${RUNFOLDER}/
mkdir -p ${BASEPATH}${RUNFOLDER}/bin
cp ../apps/Daint_2Fish_3D/factory2Stefans     ${BASEPATH}${RUNFOLDER}/bin/factory

cp ${HOME}/CubismUP_3D/makefiles/simulation   ${BASEPATH}${RUNFOLDER}/execSim
