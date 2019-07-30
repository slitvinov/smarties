export INTERNALAPP=true

if [[ "${SKIPMAKE}" != "true" ]] ; then
rm ../makefiles/libsimulation.a
make -C ../makefiles/ app=glider_CUP2D  -j4
fi

cat <<EOF >${BASEPATH}${RUNFOLDER}/runArguments00.sh
../launchSim.sh -CFL 0.1 -DLM 1 -lambda 1e5 -iterativePenalization 1 -poissonType cosine -muteAll 1 -bpdx 32 -bpdy 32 -tdump 1 -nu 0.0004 -tend 0 -shapes 'glider_semiAxisX=.125_semiAxisY=.025_rhoS=2_xpos=.6_ypos=.4_bFixed=1_bForced=0'
EOF
#cp ../apps/glider_CUP2D/ODE_timeOpt_rho1.01_ar0.2/agent* ${BASEPATH}${RUNFOLDER}/
cp ../apps/glider_CUP2D/glider_timeopt_rho2_noise005/agent* ${BASEPATH}${RUNFOLDER}/

cat <<EOF >${BASEPATH}${RUNFOLDER}/appSettings.sh
SETTINGS+=" --appSettings runArguments00.sh"
SETTINGS+=" --nStepPappSett 0 "
EOF
chmod 755 ${BASEPATH}${RUNFOLDER}/appSettings.sh
