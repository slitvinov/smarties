cp ../apps/smartCyl/runArguments* ${BASEPATH}${RUNFOLDER}/

if [[ "${SKIPMAKE}" != "true" ]] ; then
make -C ../makefiles/ clean
rm ../makefiles/libsimulation.a
rm ../makefiles/rl
make -C ../makefiles/ app=smartCyl -j4 #config=segf
fi

cat <<EOF >${BASEPATH}${RUNFOLDER}/appSettings.sh
SETTINGS+=" --appSettings runArguments01.sh"
SETTINGS+=" --nStepPappSett 0 "
EOF
chmod 755 ${BASEPATH}${RUNFOLDER}/appSettings.sh
