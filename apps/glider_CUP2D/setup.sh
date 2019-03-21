cp ../apps/glider_CUP2D/runArguments* ${BASEPATH}${RUNFOLDER}/

if [[ "${SKIPMAKE}" != "true" ]] ; then
make -C ../makefiles/ clean
make -C ../makefiles/ app=glider_CUP2D  -j4
fi

cat <<EOF >${BASEPATH}${RUNFOLDER}/appSettings.sh
SETTINGS+=" --appSettings runArguments00.sh"
SETTINGS+=" --nStepPappSett 0 "
EOF
chmod 755 ${BASEPATH}${RUNFOLDER}/appSettings.sh
