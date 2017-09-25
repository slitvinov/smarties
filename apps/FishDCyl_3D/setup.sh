cp ../apps/FishDCyl_3D/factory             ${BASEPATH}${RUNFOLDER}/
cp ../apps/FishDCyl_3D/settings_32.txt     ${BASEPATH}${RUNFOLDER}/
cp ../apps/FishDCyl_3D/settings_64.txt     ${BASEPATH}${RUNFOLDER}/
cp ../apps/FishDCyl_3D/factory2Stefans     ${BASEPATH}${RUNFOLDER}/

mkdir -p ${BASEPATH}${RUNFOLDER}/bin
cp ../apps/FishDCyl_3D/factory2Stefans     ${BASEPATH}${RUNFOLDER}/bin/factory

#cp ${HOME}/CubismUP_3D/makefiles/simulation   ${BASEPATH}${RUNFOLDER}/execSim
