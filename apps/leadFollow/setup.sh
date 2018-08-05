#cp ../apps/leadFollow/launchSim.sh ${BASEPATH}${RUNFOLDER}/
cp ../apps/leadFollow/runArguments.sh ${BASEPATH}${RUNFOLDER}/

make -C ../makefiles/ clean
make -C ../makefiles/ app=leadFollow -j
#config=segf 
cat <<EOF >${BASEPATH}${RUNFOLDER}/appSettings.sh
SETTINGS+=" --appSettings runArguments.sh "
EOF
chmod +x ${BASEPATH}${RUNFOLDER}/appSettings.sh

#export LD_LIBRARY_PATH=/cluster/home/novatig/VTK-7.1.0/Build/lib/:$LD_LIBRARY_PATH
#cp ${HOME}/CubismUP_2D/makefiles/blowfish ${BASEPATH}${RUNFOLDER}/
