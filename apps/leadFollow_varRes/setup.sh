cp ../apps/leadFollow_varRes/runArguments* ${BASEPATH}${RUNFOLDER}/

make -C ../makefiles/ clean
make -C ../makefiles/ app=leadFollow_varRes -j4
#config=segf 
cat <<EOF >${BASEPATH}${RUNFOLDER}/appSettings.sh
SETTINGS+=" --appSettings runArguments00.sh;runArguments01.sh;runArguments02.sh;runArguments03.sh "
SETTINGS+=" --nStepPappSett 4194304;2097152;1048576;0 "
EOF
chmod +x ${BASEPATH}${RUNFOLDER}/appSettings.sh

