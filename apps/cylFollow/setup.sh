cp ../apps/cylFollow/runArguments* ${BASEPATH}${RUNFOLDER}/

rm ../makefiles/libsimulation.a
rm ../makefiles/rl
make -C ../makefiles/ app=cylFollow -j4 #config=debug
#config=segf
cat <<EOF >${BASEPATH}${RUNFOLDER}/appSettings.sh
SETTINGS+=" --appSettings runArguments00.sh,runArguments01.sh,runArguments02.sh,runArguments03.sh "
SETTINGS+=" --nStepPappSett 4194304,2097152,1048576,0 "
EOF
chmod 755 ${BASEPATH}${RUNFOLDER}/appSettings.sh
