cp ../apps/smartCyl_eval/runArguments* ${BASEPATH}${RUNFOLDER}/
#cp ../apps/smartCyl_re500_eval/agent_00* ${BASEPATH}${RUNFOLDER}/

#make -C ../makefiles/ clean
#rm ../makefiles/libsimulation.a
#rm ../makefiles/rl
#make -C ../makefiles/ app=smartCyl precision=single config=prod -j4
#config=segf

cat <<EOF >${BASEPATH}${RUNFOLDER}/appSettings.sh
SETTINGS+=" --appSettings runArguments00.sh"
SETTINGS+=" --nStepPappSett 0 "
EOF
chmod 755 ${BASEPATH}${RUNFOLDER}/appSettings.sh
