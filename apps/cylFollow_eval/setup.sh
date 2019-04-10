export INTERNALAPP=true
cp ../apps/cylFollow_eval/runArguments* ${BASEPATH}${RUNFOLDER}/
cp ../apps/cylFollow_eval/agent_00* ${BASEPATH}${RUNFOLDER}/

if [[ "${SKIPMAKE}" != "true" ]] ; then
make -C ../makefiles/ clean
make -C ../makefiles/ app=cylFollow -j4 #config=segf
fi

cat <<EOF >${BASEPATH}${RUNFOLDER}/appSettings.sh
SETTINGS+=" --appSettings runArguments00.sh "
SETTINGS+=" --nStepPappSett 0 "
EOF
chmod 755 ${BASEPATH}${RUNFOLDER}/appSettings.sh

#SETTINGS+=" --nStepPappSett 2097152,1048576,524288,0 "
#SETTINGS+=" --nStepPappSett 4194304,2097152,1048576,0 "
