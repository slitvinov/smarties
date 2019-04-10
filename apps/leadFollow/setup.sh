export INTERNALAPP=true
cp ../apps/leadFollow/runArguments* ${BASEPATH}${RUNFOLDER}/

if [[ "${SKIPMAKE}" != "true" ]] ; then
make -C ../makefiles/ clean
make -C ../makefiles/ app=leadFollow -j4 #config=segf
fi

cat <<EOF >${BASEPATH}${RUNFOLDER}/appSettings.sh
SETTINGS+=" --appSettings runArguments00.sh,runArguments01.sh,runArguments02.sh,runArguments03.sh "
SETTINGS+=" --nStepPappSett 4194304,2097152,1048576,0 "
EOF
chmod 755 ${BASEPATH}${RUNFOLDER}/appSettings.sh
