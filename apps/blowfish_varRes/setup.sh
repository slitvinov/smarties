cp ../apps/blowfish_varRes/runArguments* ${BASEPATH}${RUNFOLDER}/

make -C ../makefiles/ clean
make -C ../makefiles/ config=prod app=blowfish -j

cat <<EOF >${BASEPATH}${RUNFOLDER}/appSettings.sh
SETTINGS+=" --appSettings runArguments00.sh;runArguments01.sh;runArguments02.sh;runArguments03.sh "
SETTINGS+=" --nStepPappSett 1048576;1048576;1048576;0 "
EOF
chmod +x ${BASEPATH}${RUNFOLDER}/appSettings.sh

#export LD_LIBRARY_PATH=/cluster/home/novatig/VTK-7.1.0/Build/lib/:$LD_LIBRARY_PATH
#cp ${HOME}/CubismUP_2D/makefiles/blowfish ${BASEPATH}${RUNFOLDER}/
