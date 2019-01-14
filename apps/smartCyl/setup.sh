cp ../apps/smartCyl/runArguments* ${BASEPATH}${RUNFOLDER}/

#make -C ../makefiles/ clean
rm ../makefiles/libsimulation.a
rm ../makefiles/rl
make -C ../makefiles/ app=smartCyl precision=single -j4
#config=segf
cat <<EOF >${BASEPATH}${RUNFOLDER}/appSettings.sh
SETTINGS+=" --appSettings runArguments00.sh"
SETTINGS+=" --nStepPappSett 0 "
EOF
chmod 755 ${BASEPATH}${RUNFOLDER}/appSettings.sh
