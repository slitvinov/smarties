export INTERNALAPP=true
cp ../apps/leadFollowDiffSize/runArguments* ${BASEPATH}${RUNFOLDER}/

if [[ "${SKIPMAKE}" != "true" ]] ; then
make -C ../makefiles/ clean
make -C ../makefiles/ app=leadFollow -j #config=segf
fi

cat <<EOF >${BASEPATH}${RUNFOLDER}/appSettings.sh
SETTINGS+=" --appSettings runArguments00.sh "
SETTINGS+=" --nStepPappSett 4194304 "
EOF
chmod 755 ${BASEPATH}${RUNFOLDER}/appSettings.sh
