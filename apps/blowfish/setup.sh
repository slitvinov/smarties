cp ../apps/blowfish/runArguments* ${BASEPATH}${RUNFOLDER}/

make -C ../makefiles/ clean
make -C ../makefiles/  app=blowfish -j4

cat <<EOF >${BASEPATH}${RUNFOLDER}/appSettings.sh
SETTINGS+=" --appSettings runArguments00.sh,runArguments01.sh,runArguments02.sh,runArguments03.sh "
SETTINGS+=" --nStepPappSett 2097152,1048576,524288,0 "
EOF
chmod 755 ${BASEPATH}${RUNFOLDER}/appSettings.sh

#SETTINGS+=" --nStepPappSett 2097152,1048576,524288,0 "
#export LD_LIBRARY_PATH=/cluster/home/novatig/VTK-7.1.0/Build/lib/:$LD_LIBRARY_PATH
#cp ${HOME}/CubismUP_2D/makefiles/blowfish ${BASEPATH}${RUNFOLDER}/
